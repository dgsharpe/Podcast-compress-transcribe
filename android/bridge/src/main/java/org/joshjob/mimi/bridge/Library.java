package org.joshjob.mimi.bridge;

import android.content.Context;
import android.database.Cursor;
import android.net.Uri;
import android.provider.OpenableColumns;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.RandomAccessFile;
import java.util.Arrays;

/** The bridge's own copies of imported `.mimi` files (they are small: ~2 MB per hour). */
final class Library {
  private Library() {}

  static File dir(Context c) {
    File d = new File(c.getFilesDir(), "library");
    d.mkdirs();
    return d;
  }

  static File[] list(Context c) {
    File[] fs = dir(c).listFiles((d, n) -> n.endsWith(".mimi"));
    if (fs == null) return new File[0];
    Arrays.sort(fs, (a, b) -> Long.compare(b.lastModified(), a.lastModified()));
    return fs;
  }

  /** Copies {@code uri} into the library (validating the header) and returns the stored file. */
  static File importUri(Context c, Uri uri) throws IOException {
    String name = displayName(c, uri);
    if (name == null) name = "episode";
    name = name.replaceAll("\\.mimi$", "").replaceAll("[^A-Za-z0-9._-]+", "_");
    File tmp = new File(dir(c), name + ".mimi.tmp"), dst = new File(dir(c), name + ".mimi");
    try (InputStream in = c.getContentResolver().openInputStream(uri);
        OutputStream out = new FileOutputStream(tmp)) {
      if (in == null) throw new IOException("cannot open " + uri);
      byte[] buf = new byte[1 << 16];
      int n;
      while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
    }
    try {
      numSamples(tmp);
    } catch (IOException e) {
      tmp.delete();
      throw e;
    }
    if (!tmp.renameTo(dst)) throw new IOException("rename failed");
    return dst;
  }

  /** Decoded length in samples, from the header; throws if this is not a v1 `.mimi`. */
  static long numSamples(File f) throws IOException {
    try (RandomAccessFile r = new RandomAccessFile(f, "r")) {
      byte[] h = new byte[64];
      r.readFully(h);
      if (h[0] != 'M' || h[1] != 'I' || h[2] != 'M' || h[3] != 'I' || h[4] != 1) {
        throw new IOException(f.getName() + " is not a .mimi v1 file");
      }
      long frames = (h[16] & 0xff) | (h[17] & 0xff) << 8 | (h[18] & 0xff) << 16 | (long) (h[19] & 0xff) << 24;
      return frames * 1920;
    }
  }

  /** True if the content starts with a v1 `.mimi` header. */
  static boolean isMimi(Context c, Uri uri) {
    try (InputStream in = c.getContentResolver().openInputStream(uri)) {
      byte[] h = new byte[5];
      return in != null && in.read(h) == 5 && h[0] == 'M' && h[1] == 'I' && h[2] == 'M' && h[3] == 'I'
          && h[4] == 1;
    } catch (Exception e) {
      return false;
    }
  }

  static String displayNameOr(Context c, Uri uri, String fallback) {
    String n = displayName(c, uri);
    return n == null ? fallback : n;
  }

  private static String displayName(Context c, Uri uri) {
    if ("file".equals(uri.getScheme())) return new File(uri.getPath()).getName();
    try (Cursor cur = c.getContentResolver().query(uri, new String[] {OpenableColumns.DISPLAY_NAME}, null, null, null)) {
      if (cur != null && cur.moveToFirst()) return cur.getString(0);
    } catch (Exception ignored) {
    }
    return uri.getLastPathSegment();
  }
}
