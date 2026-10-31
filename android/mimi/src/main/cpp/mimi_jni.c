// JNI bridge for org.joshjob.mimi.MimiNative -> libmimi.
#include <jni.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "mimi.h"

static char g_err[512];

static const char* cstr(JNIEnv* env, jstring s) {
  return s ? (*env)->GetStringUTFChars(env, s, NULL) : NULL;
}
static void rel(JNIEnv* env, jstring s, const char* c) {
  if (s && c) (*env)->ReleaseStringUTFChars(env, s, c);
}

JNIEXPORT jlong JNICALL Java_org_joshjob_mimi_MimiNative_nCreate(
    JNIEnv* env, jclass cls, jstring modelDir, jstring runtimeDir, jstring npuDir, jstring cacheDir,
    jint tx, jint conv, jint threads) {
  const char *m = cstr(env, modelDir), *r = cstr(env, runtimeDir), *n = cstr(env, npuDir),
             *c = cstr(env, cacheDir);
  MimiConfig cfg = {.model_dir = m, .runtime_dir = r, .npu_dir = n, .cache_dir = c,
                    .tx = (MimiBackend)tx, .conv = (MimiBackend)conv, .threads = threads};
  g_err[0] = 0;
  MimiDecoder* d = mimi_decoder_create(&cfg, g_err, sizeof g_err);
  rel(env, modelDir, m); rel(env, runtimeDir, r); rel(env, npuDir, n); rel(env, cacheDir, c);
  return (jlong)(intptr_t)d;
}

JNIEXPORT jstring JNICALL Java_org_joshjob_mimi_MimiNative_nLastError(JNIEnv* env, jclass cls) {
  return (*env)->NewStringUTF(env, g_err);
}

JNIEXPORT void JNICALL Java_org_joshjob_mimi_MimiNative_nDestroy(JNIEnv* env, jclass cls, jlong h) {
  mimi_decoder_destroy((MimiDecoder*)(intptr_t)h);
}

JNIEXPORT void JNICALL Java_org_joshjob_mimi_MimiNative_nReset(JNIEnv* env, jclass cls, jlong h) {
  mimi_decoder_reset((MimiDecoder*)(intptr_t)h);
}

// Decode one block. `in` holds MIMI_W packed frames (frame_bytes each) starting at inOffset; writes
// valid * MIMI_SPF 16-bit PCM samples to `out` (direct buffer) and returns the byte count, or -1.
JNIEXPORT jint JNICALL Java_org_joshjob_mimi_MimiNative_nDecode(
    JNIEnv* env, jclass cls, jlong h, jint block, jobject in, jint inOffset, jint nq, jint frameBytes,
    jint valid, jobject out) {
  MimiDecoder* d = (MimiDecoder*)(intptr_t)h;
  uint8_t* src = (*env)->GetDirectBufferAddress(env, in);
  int16_t* dst = (*env)->GetDirectBufferAddress(env, out);
  if (!d || !src || !dst || valid < 0 || valid > MIMI_H) return -1;
  if ((*env)->GetDirectBufferCapacity(env, out) < (jlong)valid * MIMI_SPF * 2) return -1;
  MimiInfo info = {.nq = nq, .bits = 11, .frame_bytes = (uint32_t)frameBytes};
  int16_t codes[MIMI_W * MIMI_NQMAX];
  mimi_unpack_frames(src + inOffset, MIMI_W, &info, codes);
  static float pcm[MIMI_H * MIMI_SPF];  // decoder calls are serialized by MimiLibrary
  int n = mimi_decode_block(d, block, codes, nq, valid, pcm);
  if (n < 0) return -1;
  for (int i = 0; i < n; i++) {
    float v = pcm[i] * 32767.f;
    dst[i] = v > 32767.f ? 32767 : v < -32768.f ? -32768 : (int16_t)lrintf(v);
  }
  return n * 2;
}

JNIEXPORT jdoubleArray JNICALL Java_org_joshjob_mimi_MimiNative_nTiming(JNIEnv* env, jclass cls,
                                                                         jlong h) {
  double t[3];
  mimi_last_timing((MimiDecoder*)(intptr_t)h, &t[0], &t[1], &t[2]);
  jdoubleArray a = (*env)->NewDoubleArray(env, 3);
  (*env)->SetDoubleArrayRegion(env, a, 0, 3, t);
  return a;
}
