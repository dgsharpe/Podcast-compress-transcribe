// mimi_dec — Mimi (kyutai) decoder for Android via the LiteRT C API (CompiledModel).
//
// Uses the litert-community/Mimi graphs: mimi_dec_tx_fp16.tflite (RVQ embedding [1,512,25] ->
// upsample + 8-layer transformer -> [1,512,50]) and mimi_deconly_fp16.tflite (SEANet decoder
// [1,512,50] -> audio [1,1,48000]), i.e. fixed 2 s windows of 25 frames @ 12.5 Hz. The RVQ
// codebooks/projections (mimi_rvq.bin) are applied on the host.
//
// Streaming: windows of W=25 frames with hop H=W-C; the first C frames of every window after
// the first are left context whose output is discarded, except the last XF samples before the
// seam, which are crossfaded with the previous window's tail.
//
// usage: mimi_dec --models DIR --codes FILE.i16 --out OUT.wav [--nq 32] [--tx cpu|gpu|gpu32|npu]
//                 [--conv cpu|gpu|gpu32|npu] [--threads 4] [--ctx 6] [--xf 480] [--max-windows N]
//                 [--libdir DIR] [--npudir DIR] [--cache DIR] [--pace SECONDS]
// NPU: --npudir holds libLiteRtDispatch_Qualcomm.so + libLiteRtCompilerPlugin_Qualcomm.so + the
// QNN libs (libQnnHtp*.so, libQnnSystem.so); set ADSP_LIBRARY_PATH to the dir with the V81 skels.
// codes file: int16, frame-major [F][32] (Mimi codebook order, semantic first).

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"

#define W 25           // frames per window (graph is fixed)
#define SPF 1920       // samples per frame (24 kHz / 12.5 Hz)
#define WS (W * SPF)   // samples per window
#define D 256          // codebook dim
#define H512 512       // embedding dim
#define CB 2048        // codebook size
#define NQMAX 32

#define CK(x)                                                              \
  do {                                                                     \
    LiteRtStatus s_ = (x);                                                 \
    if (s_ != kLiteRtStatusOk) {                                           \
      fprintf(stderr, "%s:%d %s -> status %d\n", __FILE__, __LINE__, #x, s_); \
      exit(1);                                                             \
    }                                                                      \
  } while (0)

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

static void* slurp(const char* path, size_t* n) {
  FILE* f = fopen(path, "rb");
  if (!f) { perror(path); exit(1); }
  fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
  void* p = malloc(*n);
  if (fread(p, 1, *n, f) != *n) { perror("read"); exit(1); }
  fclose(f);
  return p;
}

typedef struct {
  LiteRtModel model;
  LiteRtCompiledModel cm;
  LiteRtTensorBuffer in, out;
  size_t in_n, out_n;  // float counts
  const char* name;
} Net;

static char* dupstr(const char* s) { char* p = malloc(strlen(s) + 1); strcpy(p, s); return p; }

static void add_opaque(LiteRtOptions opt, const char* id, const char* toml) {
  LiteRtOpaqueOptions oo;
  CK(LiteRtCreateOpaqueOptions(id, dupstr(toml), free, &oo));
  CK(LiteRtAddOpaqueOptions(opt, oo));
}

static size_t numel(LiteRtRankedTensorType* t) {
  size_t n = 1;
  for (int i = 0; i < (int)t->layout.rank; i++) n *= t->layout.dimensions[i];
  return n;
}

static LiteRtTensorBuffer make_buf(LiteRtEnvironment env, LiteRtCompiledModel cm, LiteRtModel m,
                                   int is_input, size_t* nfloat) {
  LiteRtSignature sig;
  CK(LiteRtGetModelSignature(m, 0, &sig));
  LiteRtTensor t;
  if (is_input) CK(LiteRtGetSignatureInputTensorByIndex(sig, 0, &t));
  else CK(LiteRtGetSignatureOutputTensorByIndex(sig, 0, &t));
  LiteRtRankedTensorType tt;
  CK(LiteRtGetRankedTensorType(t, &tt));
  if (tt.element_type != kLiteRtElementTypeFloat32) {
    fprintf(stderr, "unexpected element type %d\n", tt.element_type); exit(1);
  }
  *nfloat = numel(&tt);
  LiteRtTensorBufferRequirements req;
  if (is_input) CK(LiteRtGetCompiledModelInputBufferRequirements(cm, 0, 0, &req));
  else CK(LiteRtGetCompiledModelOutputBufferRequirements(cm, 0, 0, &req));
  LiteRtTensorBuffer b;
  CK(LiteRtCreateManagedTensorBufferFromRequirements(env, &tt, req, &b));
  return b;
}

static void load_net(Net* n, LiteRtEnvironment env, const char* path, const char* acc, int threads) {
  n->name = path;
  CK(LiteRtCreateModelFromFile(env, path, &n->model));
  LiteRtOptions opt;
  CK(LiteRtCreateOptions(&opt));
  char toml[64];
  if (!strcmp(acc, "cpu")) {
    CK(LiteRtSetOptionsHardwareAccelerators(opt, kLiteRtHwAcceleratorCpu));
    snprintf(toml, sizeof toml, "num_threads = %d\n", threads);
    add_opaque(opt, "xnnpack", toml);
  } else if (!strncmp(acc, "gpu", 3)) {
    CK(LiteRtSetOptionsHardwareAccelerators(opt, kLiteRtHwAcceleratorGpu));
    int prec = !strcmp(acc, "gpu32") ? kLiteRtDelegatePrecisionFp32 : kLiteRtDelegatePrecisionFp16;
    snprintf(toml, sizeof toml, "precision = %d\n", prec);
    add_opaque(opt, "gpu_options", toml);
  } else if (!strcmp(acc, "npu")) {
    CK(LiteRtSetOptionsHardwareAccelerators(opt, kLiteRtHwAcceleratorNpu | kLiteRtHwAcceleratorCpu));
  } else {
    fprintf(stderr, "bad accelerator %s\n", acc); exit(1);
  }
  double t0 = now();
  CK(LiteRtCreateCompiledModel(env, n->model, opt, &n->cm));
  bool full = false;
  LiteRtCompiledModelIsFullyAccelerated(n->cm, &full);
  fprintf(stderr, "compiled %s on %s in %.0f ms (fully accelerated: %s)\n", path, acc,
          1e3 * (now() - t0), full ? "yes" : "no");
  n->in = make_buf(env, n->cm, n->model, 1, &n->in_n);
  n->out = make_buf(env, n->cm, n->model, 0, &n->out_n);
}

static void run_net(Net* n, const float* x, float* y) {
  void* p;
  CK(LiteRtLockTensorBuffer(n->in, &p, kLiteRtTensorBufferLockModeWrite));
  memcpy(p, x, n->in_n * sizeof(float));
  CK(LiteRtUnlockTensorBuffer(n->in));
  CK(LiteRtRunCompiledModel(n->cm, 0, 1, &n->in, 1, &n->out));
  CK(LiteRtLockTensorBuffer(n->out, &p, kLiteRtTensorBufferLockModeRead));
  memcpy(y, p, n->out_n * sizeof(float));
  CK(LiteRtUnlockTensorBuffer(n->out));
}

// RVQ decode for one window: emb[512][W] (channel-major) from codes[W][32].
static void rvq_window(const float* rvq, const int16_t* codes, int nq, float* emb) {
  const float* sem_Wout = rvq + 2 * D * H512;
  const float* aco_Wout = sem_Wout + H512 * D;
  const float* sem_CB = aco_Wout + H512 * D;
  const float* aco_CB = sem_CB + CB * D;
  float s[D], a[D];
  for (int t = 0; t < W; t++) {
    const int16_t* c = codes + t * NQMAX;
    memcpy(s, sem_CB + (size_t)c[0] * D, sizeof s);
    memset(a, 0, sizeof a);
    for (int i = 1; i < nq; i++) {
      const float* v = aco_CB + ((size_t)(i - 1) * CB + c[i]) * D;
      for (int k = 0; k < D; k++) a[k] += v[k];
    }
    for (int h = 0; h < H512; h++) {
      const float* ws = sem_Wout + (size_t)h * D;
      const float* wa = aco_Wout + (size_t)h * D;
      float acc = 0.f;
      for (int k = 0; k < D; k++) acc += ws[k] * s[k] + wa[k] * a[k];
      emb[h * W + t] = acc;
    }
  }
}

static void write_wav(const char* path, const float* x, size_t n) {
  FILE* f = fopen(path, "wb");
  if (!f) { perror(path); exit(1); }
  uint32_t sr = 24000, br = sr * 2, sz = (uint32_t)(n * 2);
  uint16_t one = 1, bits = 16, ba = 2;
  uint32_t fmt = 16, riff = 36 + sz;
  fwrite("RIFF", 1, 4, f); fwrite(&riff, 4, 1, f); fwrite("WAVEfmt ", 1, 8, f);
  fwrite(&fmt, 4, 1, f); fwrite(&one, 2, 1, f); fwrite(&one, 2, 1, f); fwrite(&sr, 4, 1, f);
  fwrite(&br, 4, 1, f); fwrite(&ba, 2, 1, f); fwrite(&bits, 2, 1, f);
  fwrite("data", 1, 4, f); fwrite(&sz, 4, 1, f);
  for (size_t i = 0; i < n; i++) {
    float v = x[i] * 32767.f;
    int16_t s = v > 32767.f ? 32767 : v < -32768.f ? -32768 : (int16_t)lrintf(v);
    fwrite(&s, 2, 1, f);
  }
  fclose(f);
}

static int cmpd(const void* a, const void* b) {
  double x = *(const double*)a, y = *(const double*)b;
  return (x > y) - (x < y);
}

static void stats(const char* name, double* v, int n, double audio_per_item) {
  qsort(v, n, sizeof *v, cmpd);
  double s = 0;
  for (int i = 0; i < n; i++) s += v[i];
  fprintf(stderr, "%-6s n=%d mean %.2f ms  median %.2f  p95 %.2f  max %.2f  (RTF %.4f)\n", name, n,
          1e3 * s / n, 1e3 * v[n / 2], 1e3 * v[(int)(0.95 * (n - 1))], 1e3 * v[n - 1],
          s / n / audio_per_item);
}

int main(int argc, char** argv) {
  const char *mdir = ".", *codes_path = NULL, *out = NULL, *tx_acc = "cpu", *conv_acc = "gpu",
             *libdir = NULL, *npudir = NULL, *cachedir = NULL;
  int nq = 32, threads = 4, C = 6, XF = 480, maxw = 0;
  double pace = 0;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--models")) mdir = argv[++i];
    else if (!strcmp(argv[i], "--codes")) codes_path = argv[++i];
    else if (!strcmp(argv[i], "--out")) out = argv[++i];
    else if (!strcmp(argv[i], "--nq")) nq = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--tx")) tx_acc = argv[++i];
    else if (!strcmp(argv[i], "--conv")) conv_acc = argv[++i];
    else if (!strcmp(argv[i], "--threads")) threads = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--ctx")) C = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--xf")) XF = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--max-windows")) maxw = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--libdir")) libdir = argv[++i];
    else if (!strcmp(argv[i], "--npudir")) npudir = argv[++i];
    else if (!strcmp(argv[i], "--cache")) cachedir = argv[++i];
    else if (!strcmp(argv[i], "--pace")) pace = atof(argv[++i]);
    else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
  }
  if (!codes_path || nq < 1 || nq > NQMAX || C < 1 || C >= W || XF > C * SPF) {
    fprintf(stderr, "bad args (need --codes; 1<=nq<=32; 1<=ctx<25; xf<=ctx*1920)\n");
    return 2;
  }
  const int Hp = W - C;

  double t_start = now();
  LiteRtEnvOption eo[4];
  int neo = 0;
#define ENVOPT(tag_, str_)                                \
  if (str_) {                                             \
    eo[neo].tag = (tag_);                                 \
    eo[neo].value.type = kLiteRtAnyTypeString;            \
    eo[neo].value.str_value = (str_);                     \
    neo++;                                                \
  }
  ENVOPT(kLiteRtEnvOptionTagRuntimeLibraryDir, libdir);
  ENVOPT(kLiteRtEnvOptionTagDispatchLibraryDir, npudir);
  ENVOPT(kLiteRtEnvOptionTagCompilerPluginLibraryDir, npudir);
  ENVOPT(kLiteRtEnvOptionTagCompilerCacheDir, cachedir);
  LiteRtEnvironment env;
  CK(LiteRtCreateEnvironment(neo, eo, &env));

  char p1[512], p2[512], p3[512];
  snprintf(p1, sizeof p1, "%s/mimi_dec_tx_fp16.tflite", mdir);
  snprintf(p2, sizeof p2, "%s/mimi_deconly_fp16.tflite", mdir);
  snprintf(p3, sizeof p3, "%s/mimi_rvq.bin", mdir);
  Net tx, conv;
  load_net(&tx, env, p1, tx_acc, threads);
  load_net(&conv, env, p2, conv_acc, threads);
  if (tx.in_n != H512 * W || tx.out_n != conv.in_n || conv.out_n != WS) {
    fprintf(stderr, "shape mismatch: tx %zu->%zu conv %zu->%zu\n", tx.in_n, tx.out_n, conv.in_n,
            conv.out_n);
    return 1;
  }
  size_t nb;
  float* rvq = slurp(p3, &nb);
  size_t ncb;
  int16_t* codes = slurp(codes_path, &ncb);
  int F = (int)(ncb / (NQMAX * sizeof(int16_t)));
  double t_ready = now();
  fprintf(stderr, "ready in %.0f ms; %d frames (%.1f s), nq=%d, window %d hop %d ctx %d xf %d\n",
          1e3 * (t_ready - t_start), F, F / 12.5, nq, W, Hp, C, XF);

  int nwin = F <= W ? 1 : 1 + (F - W + Hp - 1) / Hp;
  if (maxw && nwin > maxw) nwin = maxw;
  size_t outn = (size_t)(W + (size_t)(nwin - 1) * Hp) * SPF;
  float* y = calloc(outn, sizeof(float));
  float emb[H512 * W], *mid = malloc(tx.out_n * sizeof(float)), *aud = malloc(WS * sizeof(float));
  int16_t wc[W * NQMAX];
  double *trvq = malloc(nwin * sizeof(double)), *ttx = malloc(nwin * sizeof(double)),
         *tconv = malloc(nwin * sizeof(double)), *tall = malloc(nwin * sizeof(double));
  double first_audio = 0;

  for (int k = 0; k < nwin; k++) {
    double t0 = now();
    int f0 = k * Hp;
    for (int t = 0; t < W; t++) {  // pad past the end by repeating the last frame
      int f = f0 + t < F ? f0 + t : F - 1;
      memcpy(wc + t * NQMAX, codes + (size_t)f * NQMAX, NQMAX * sizeof(int16_t));
    }
    rvq_window(rvq, wc, nq, emb);
    double t1 = now();
    run_net(&tx, emb, mid);
    double t2 = now();
    run_net(&conv, mid, aud);
    double t3 = now();
    size_t base = (size_t)f0 * SPF;
    if (k == 0) {
      memcpy(y, aud, WS * sizeof(float));
      first_audio = t3 - t0;
    } else {
      size_t seam = (size_t)C * SPF;  // local index of first kept sample
      for (int i = 0; i < XF; i++) {
        float a = (i + 0.5f) / XF;
        size_t g = base + seam - XF + i;
        y[g] = (1.f - a) * y[g] + a * aud[seam - XF + i];
      }
      memcpy(y + base + seam, aud + seam, (WS - seam) * sizeof(float));
    }
    trvq[k] = t1 - t0; ttx[k] = t2 - t1; tconv[k] = t3 - t2; tall[k] = t3 - t0;
    if (pace > 0) {
      double rest = pace - (now() - t0);
      if (rest > 0) usleep((useconds_t)(rest * 1e6));
    }
  }
  double t_end = now();
  size_t keep = (size_t)F * SPF < outn ? (size_t)F * SPF : outn;
  if (maxw) keep = outn;
  double audio_s = keep / 24000.0;
  fprintf(stderr, "decoded %.1f s of audio in %.2f s wall (RTF %.4f); first 2 s window %.1f ms\n",
          audio_s, t_end - t_ready, (t_end - t_ready) / audio_s, 1e3 * first_audio);
  double hop_s = Hp / 12.5;
  stats("rvq", trvq, nwin, hop_s);
  stats("tx", ttx, nwin, hop_s);
  stats("conv", tconv, nwin, hop_s);
  stats("total", tall, nwin, hop_s);
  if (out) write_wav(out, y, keep);
  return 0;
}
