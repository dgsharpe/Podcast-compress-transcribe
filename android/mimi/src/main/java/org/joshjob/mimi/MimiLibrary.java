package org.joshjob.mimi;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.system.Os;
import android.util.Log;
import androidx.annotation.Nullable;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;

/**
 * Process-wide access to the native Mimi decoder.
 *
 * <p>Creating the decoder compiles the two LiteRT graphs (about 2–3 s with a warm NPU cache), so one
 * instance is shared by every {@link MimiDecoder}; call {@link #preloadAsync} early (e.g. in {@code
 * Application.onCreate}) to keep that off the playback path. Backends are tried in order: NPU
 * transformer + GPU conv decoder, CPU + GPU, then CPU only.
 */
public final class MimiLibrary {
  private static final String TAG = "MimiLibrary";
  public static final String MIME_TYPE = "audio/x-mimi";
  static final String[] MODEL_FILES = {
    "mimi_dec_tx_fp16.tflite", "mimi_deconly_fp16.tflite", "mimi_rvq.bin"
  };
  private static final int[][] BACKENDS = {
    {MimiNative.NPU, MimiNative.GPU}, {MimiNative.CPU, MimiNative.GPU}, {MimiNative.CPU, MimiNative.CPU}
  };
  private static final String[] BACKEND_NAMES = {"CPU", "GPU", "NPU"};

  private static final Object LOAD_LOCK = new Object();
  private static boolean loadAttempted;
  @Nullable private static Throwable loadError;
  @Nullable private static volatile Handle shared;

  private MimiLibrary() {}

  /** Whether the native library could be loaded (64-bit ARM only). */
  public static boolean isAvailable() {
    synchronized (LOAD_LOCK) {  // not the class lock: acquire() holds that for seconds
      if (!loadAttempted) {
        loadAttempted = true;
        try {
          System.loadLibrary("mimi_jni");
        } catch (Throwable t) {
          loadError = t;
          Log.w(TAG, "libmimi_jni unavailable", t);
        }
      }
      return loadError == null;
    }
  }

  /** Starts creating the shared decoder on a background thread. */
  public static void preloadAsync(Context context) {
    Context app = context.getApplicationContext();
    new Thread(
            () -> {
              try {
                acquire(app);
              } catch (MimiDecoderException e) {
                Log.w(TAG, "preload failed", e);
              }
            },
            "mimi-preload")
        .start();
  }

  /** Returns the shared decoder, creating it on first use (blocking). */
  public static synchronized Handle acquire(Context context) throws MimiDecoderException {
    if (shared != null) return shared;
    if (!isAvailable()) throw new MimiDecoderException("native library unavailable", loadError);
    Context app = context.getApplicationContext();
    String libDir = app.getApplicationInfo().nativeLibraryDir;
    try {
      // FastRPC loads the Hexagon skel from here; must be set before QNN initializes.
      Os.setenv(
          "ADSP_LIBRARY_PATH", libDir + ";/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/vendor/dsp", true);
    } catch (Exception e) {
      Log.w(TAG, "setenv ADSP_LIBRARY_PATH failed", e);
    }
    File models;
    try {
      models = ensureModels(app);
    } catch (IOException e) {
      throw new MimiDecoderException("cannot unpack models", e);
    }
    File cache = new File(app.getCodeCacheDir(), "mimi_npu");
    cache.mkdirs();
    StringBuilder errors = new StringBuilder();
    for (int[] be : BACKENDS) {
      long t0 = System.nanoTime();
      long h =
          MimiNative.nCreate(
              models.getAbsolutePath(), libDir, libDir, cache.getAbsolutePath(), be[0], be[1], 2);
      String name = "tx=" + BACKEND_NAMES[be[0]] + " conv=" + BACKEND_NAMES[be[1]];
      if (h != 0) {
        shared = new Handle(h, name, (System.nanoTime() - t0) / 1e6);
        Log.i(TAG, "decoder ready: " + name + " in " + Math.round(shared.initMs) + " ms");
        return shared;
      }
      String err = MimiNative.nLastError();
      Log.w(TAG, name + " failed: " + err);
      errors.append(name).append(": ").append(err).append("; ");
    }
    throw new MimiDecoderException("no usable backend: " + errors);
  }

  /** Stats of the shared decoder if it has been created, else a placeholder (never blocks). */
  public static String acquireIfReady() throws MimiDecoderException {
    Handle h = shared;  // volatile read; acquire() holds the class lock while creating
    return h == null ? "decoder not created yet" : h.stats();
  }

  /** Copies the bundled models from assets/mimi/ to files/mimi/ (once; re-copied if sizes differ). */
  static File ensureModels(Context context) throws IOException {
    File dir = new File(context.getFilesDir(), "mimi");
    dir.mkdirs();
    AssetManager am = context.getAssets();
    byte[] buf = new byte[1 << 20];
    for (String name : MODEL_FILES) {
      File dst = new File(dir, name);
      long want = -1;
      try (AssetFileDescriptor afd = am.openFd("mimi/" + name)) {
        want = afd.getLength();
      } catch (IOException compressed) {
        // openFd only works for uncompressed assets (app sets noCompress "tflite", "bin");
        // otherwise trust an existing copy, which is only ever created by rename after a full write.
      }
      if (want >= 0 ? dst.length() == want : dst.exists()) continue;
      File tmp = new File(dir, name + ".tmp");
      try (InputStream in = am.open("mimi/" + name);
          OutputStream out = new FileOutputStream(tmp)) {
        int n;
        while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
      }
      if (!tmp.renameTo(dst)) throw new IOException("rename failed for " + dst);
    }
    return dir;
  }

  /** The shared native decoder. Calls are serialized. */
  public static final class Handle {
    private final long ptr;
    public final String backends;
    public final double initMs;
    private long blocks;
    private double totalMs, worstMs;

    Handle(long ptr, String backends, double initMs) {
      this.ptr = ptr;
      this.backends = backends;
      this.initMs = initMs;
    }

    // The native decoder keeps one crossfade tail; it is only valid for the stream that produced it.
    @Nullable private Object lastOwner;
    private int lastBlock = -2;

    synchronized void reset() {
      MimiNative.nReset(ptr);
      lastBlock = -2;
    }

    /** True if {@code owner}'s previous decode was block k−1 and nobody decoded since. */
    synchronized boolean follows(Object owner, int k) {
      return lastOwner == owner && lastBlock == k - 1;
    }

    synchronized int decode(
        Object owner, int block, ByteBuffer in, int inOffset, int nq, int frameBytes, int valid,
        ByteBuffer out) {
      if (owner != lastOwner) {
        MimiNative.nReset(ptr);
        lastOwner = owner;
      }
      lastBlock = block;
      long t0 = System.nanoTime();
      int n = MimiNative.nDecode(ptr, block, in, inOffset, nq, frameBytes, valid, out);
      double ms = (System.nanoTime() - t0) / 1e6;
      blocks++;
      totalMs += ms;
      worstMs = Math.max(worstMs, ms);
      return n;
    }

    /** One-line summary: backends, init time, blocks decoded, mean / worst ms per 1.44 s block. */
    public synchronized String stats() {
      return String.format(
          java.util.Locale.US,
          "%s, init %.0f ms, %d blocks, mean %.1f ms, worst %.1f ms per 1.44 s block",
          backends, initMs, blocks, blocks > 0 ? totalMs / blocks : 0.0, worstMs);
    }
  }
}
