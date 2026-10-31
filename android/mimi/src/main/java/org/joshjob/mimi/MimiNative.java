package org.joshjob.mimi;

import java.nio.ByteBuffer;

/** JNI entry points into libmimi (src/main/cpp/mimi_jni.c). Use {@link MimiLibrary} instead. */
final class MimiNative {
  static final int CPU = 0;
  static final int GPU = 1;
  static final int NPU = 2;

  private MimiNative() {}

  static native long nCreate(
      String modelDir, String runtimeDir, String npuDir, String cacheDir, int tx, int conv, int threads);

  static native String nLastError();

  static native void nDestroy(long handle);

  static native void nReset(long handle);

  /** Returns the number of PCM16 bytes written to {@code out}, or -1. */
  static native int nDecode(
      long handle, int block, ByteBuffer in, int inOffset, int nq, int frameBytes, int valid,
      ByteBuffer out);

  /** {rvq, tx, conv} milliseconds of the last decode. */
  static native double[] nTiming(long handle);
}
