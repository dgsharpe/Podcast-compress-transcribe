package org.joshjob.mimi.bridge;

import android.content.ContentProvider;
import android.content.ContentValues;
import android.database.Cursor;
import android.database.MatrixCursor;
import android.net.Uri;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.ParcelFileDescriptor;
import android.os.ProxyFileDescriptorCallback;
import android.os.storage.StorageManager;
import android.provider.OpenableColumns;
import android.system.ErrnoException;
import android.system.OsConstants;
import android.util.Log;
import androidx.annotation.Nullable;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import org.joshjob.mimi.MimiFileDecoder;

/**
 * Serves {@code content://org.joshjob.mimi.bridge.wav/<name>.wav} for every `.mimi` in the app's
 * library dir as a seekable 16-bit 24 kHz mono WAV, decoded on demand as the reader asks for bytes.
 */
public final class WavProvider extends ContentProvider {
  static final String AUTHORITY = "org.joshjob.mimi.bridge.wav";
  private static final String TAG = "MimiWavProvider";
  private static final int HDR = 44;

  static Uri uriFor(File mimi) {
    String name = mimi.getName().replaceAll("\\.mimi$", "") + ".wav";
    return new Uri.Builder().scheme("content").authority(AUTHORITY).appendPath(name).build();
  }

  private File sourceFor(Uri uri) throws FileNotFoundException {
    String seg = uri.getLastPathSegment();
    if (seg == null || !seg.endsWith(".wav") || seg.contains("/")) throw new FileNotFoundException(uri.toString());
    File f = new File(Library.dir(getContext()), seg.substring(0, seg.length() - 4) + ".mimi");
    if (!f.isFile()) throw new FileNotFoundException(uri.toString());
    return f;
  }

  @Override
  public boolean onCreate() {
    return true;
  }

  @Override
  public String getType(Uri uri) {
    return "audio/wav";
  }

  @Override
  public Cursor query(Uri uri, @Nullable String[] projection, @Nullable String selection,
      @Nullable String[] selectionArgs, @Nullable String sortOrder) {
    String[] cols = projection != null ? projection
        : new String[] {OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE};
    MatrixCursor c = new MatrixCursor(cols);
    try {
      File src = sourceFor(uri);
      long size = HDR + Library.numSamples(src) * 2;
      Object[] row = new Object[cols.length];
      for (int i = 0; i < cols.length; i++) {
        if (OpenableColumns.DISPLAY_NAME.equals(cols[i])) row[i] = uri.getLastPathSegment();
        else if (OpenableColumns.SIZE.equals(cols[i])) row[i] = size;
      }
      c.addRow(row);
    } catch (IOException e) {
      Log.w(TAG, "query " + uri, e);
    }
    return c;
  }

  @Override
  public ParcelFileDescriptor openFile(Uri uri, String mode) throws FileNotFoundException {
    if (!"r".equals(mode)) throw new SecurityException("read-only");
    File src = sourceFor(uri);
    HandlerThread thread = new HandlerThread("mimi-wav-" + src.getName());
    thread.start();
    try {
      FileInputStream in = new FileInputStream(src);
      MimiFileDecoder dec = new MimiFileDecoder(getContext(), in.getChannel());
      StorageManager sm = getContext().getSystemService(StorageManager.class);
      return sm.openProxyFileDescriptor(
          ParcelFileDescriptor.MODE_READ_ONLY, new WavCallback(dec, thread), new Handler(thread.getLooper()));
    } catch (Exception e) {
      thread.quitSafely();
      Log.e(TAG, "open " + uri, e);
      throw new FileNotFoundException(uri + ": " + e);
    }
  }

  /** Byte-level view of the decoded audio: a 44-byte WAV header, then little-endian PCM16. */
  private static final class WavCallback extends ProxyFileDescriptorCallback {
    private final MimiFileDecoder dec;
    private final HandlerThread thread;
    private final byte[] header;
    private final long size;

    WavCallback(MimiFileDecoder dec, HandlerThread thread) {
      this.dec = dec;
      this.thread = thread;
      long data = dec.numSamples() * 2;
      this.size = HDR + data;
      this.header = wavHeader(data);
    }

    @Override
    public long onGetSize() {
      return size;
    }

    @Override
    public int onRead(long offset, int len, byte[] out) throws ErrnoException {
      if (offset >= size) return 0;
      len = (int) Math.min(len, size - offset);
      int done = 0;
      try {
        while (done < len) {
          long pos = offset + done;
          if (pos < HDR) {
            int n = (int) Math.min(len - done, HDR - pos);
            System.arraycopy(header, (int) pos, out, done, n);
            done += n;
            continue;
          }
          long byteInData = pos - HDR;
          long sample = byteInData / 2;
          int k = (int) (sample / MimiFileDecoder.SAMPLES_PER_BLOCK);
          short[] blk = dec.block(k);
          long blockStartByte = (long) k * MimiFileDecoder.SAMPLES_PER_BLOCK * 2;
          int inBlock = (int) (byteInData - blockStartByte);
          int avail = blk.length * 2 - inBlock;
          if (avail <= 0) break;
          int n = Math.min(len - done, avail);
          for (int i = 0; i < n; i++) {
            int b = inBlock + i;
            short s = blk[b >> 1];
            out[done + i] = (byte) ((b & 1) == 0 ? s : s >> 8);
          }
          done += n;
        }
      } catch (IOException e) {
        Log.e(TAG, "read at " + offset, e);
        throw new ErrnoException("onRead", OsConstants.EIO);
      }
      return done;
    }

    @Override
    public void onRelease() {
      try {
        dec.close();
      } catch (IOException ignored) {
      }
      thread.quitSafely();
    }
  }

  static byte[] wavHeader(long dataBytes) {
    byte[] h = new byte[HDR];
    put(h, 0, "RIFF");
    le32(h, 4, (int) Math.min(0xffffffffL, 36 + dataBytes));
    put(h, 8, "WAVEfmt ");
    le32(h, 16, 16);
    h[20] = 1;  // PCM
    h[22] = 1;  // mono
    le32(h, 24, 24000);
    le32(h, 28, 48000);
    h[32] = 2;
    h[34] = 16;
    put(h, 36, "data");
    le32(h, 40, (int) Math.min(0xffffffffL, dataBytes));
    return h;
  }

  private static void put(byte[] b, int o, String s) {
    for (int i = 0; i < s.length(); i++) b[o + i] = (byte) s.charAt(i);
  }

  private static void le32(byte[] b, int o, int v) {
    b[o] = (byte) v;
    b[o + 1] = (byte) (v >> 8);
    b[o + 2] = (byte) (v >> 16);
    b[o + 3] = (byte) (v >> 24);
  }

  @Override
  public Uri insert(Uri uri, @Nullable ContentValues values) {
    throw new UnsupportedOperationException();
  }

  @Override
  public int delete(Uri uri, @Nullable String selection, @Nullable String[] selectionArgs) {
    throw new UnsupportedOperationException();
  }

  @Override
  public int update(Uri uri, @Nullable ContentValues values, @Nullable String selection,
      @Nullable String[] selectionArgs) {
    throw new UnsupportedOperationException();
  }
}
