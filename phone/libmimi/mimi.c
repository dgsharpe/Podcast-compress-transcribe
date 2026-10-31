// libmimi implementation. See mimi.h.
#include "mimi.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"

#define WS (MIMI_W * MIMI_SPF)
#define D 256
#define H512 512
#define CB 2048

static uint32_t rd32(const uint8_t* p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }
static uint16_t rd16(const uint8_t* p) { return p[0] | p[1] << 8; }

int mimi_parse_header(const uint8_t* b, size_t len, MimiInfo* in) {
  if (len < MIMI_HEADER_BYTES || memcmp(b, "MIMI", 4)) return -1;
  in->version = b[4]; in->nq = b[5]; in->bits = b[6];
  in->sample_rate = rd32(b + 8); in->fr_num = rd16(b + 12); in->fr_den = rd16(b + 14);
  in->n_frames = rd32(b + 16); in->frame_bytes = rd32(b + 20);
  in->data_offset = rd32(b + 24); in->meta_len = rd32(b + 28);
  memcpy(in->model, b + 32, 32); in->model[32] = 0;
  if (in->version != 1 || in->bits != 11 || in->nq < 1 || in->nq > MIMI_NQMAX ||
      in->sample_rate != 24000 || in->fr_num != 25 || in->fr_den != 2 ||
      in->frame_bytes != (uint32_t)(in->nq * in->bits + 7) / 8)
    return -1;
  return 0;
}

void mimi_unpack_frames(const uint8_t* fr, int n, const MimiInfo* in, int16_t* codes) {
  for (int f = 0; f < n; f++) {
    const uint8_t* p = fr + (size_t)f * in->frame_bytes;
    int16_t* c = codes + (size_t)f * MIMI_NQMAX;
    for (int i = 0; i < MIMI_NQMAX; i++) {
      if (i >= in->nq) { c[i] = 0; continue; }
      int bit = i * in->bits, v = 0;
      for (int j = 0; j < in->bits; j++, bit++) v |= ((p[bit >> 3] >> (bit & 7)) & 1) << j;
      c[i] = (int16_t)v;
    }
  }
}

int mimi_num_blocks(uint32_t n_frames) { return (int)((n_frames + MIMI_H - 1) / MIMI_H); }

void mimi_block_span(uint32_t n_frames, int k, int* ws, int* valid) {
  *ws = k * MIMI_H - MIMI_C;
  int v = (int)n_frames - k * MIMI_H;
  *valid = v < 0 ? 0 : v > MIMI_H ? MIMI_H : v;
}

typedef struct {
  LiteRtModel model;
  LiteRtCompiledModel cm;
  LiteRtTensorBuffer in, out;
  size_t in_n, out_n;
} Net;

struct MimiDecoder {
  LiteRtEnvironment env;
  Net tx, conv;
  float* rvq;
  float *emb, *mid, *aud, tail[MIMI_XF];
  int last_k, have_tail;
  double t_rvq, t_tx, t_conv;
};

static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}

#define TRY(x, what)                                                        \
  do {                                                                      \
    LiteRtStatus s_ = (x);                                                  \
    if (s_ != kLiteRtStatusOk) {                                            \
      if (err) snprintf(err, err_len, "%s failed (status %d)", what, s_);   \
      return -1;                                                            \
    }                                                                       \
  } while (0)

static char* dupstr(const char* s) { char* p = malloc(strlen(s) + 1); strcpy(p, s); return p; }

static int buf_for(LiteRtEnvironment env, Net* n, int is_in, char* err, size_t err_len) {
  LiteRtSignature sig;
  LiteRtTensor t;
  LiteRtRankedTensorType tt;
  LiteRtTensorBufferRequirements req;
  TRY(LiteRtGetModelSignature(n->model, 0, &sig), "signature");
  if (is_in) TRY(LiteRtGetSignatureInputTensorByIndex(sig, 0, &t), "input tensor");
  else TRY(LiteRtGetSignatureOutputTensorByIndex(sig, 0, &t), "output tensor");
  TRY(LiteRtGetRankedTensorType(t, &tt), "tensor type");
  if (tt.element_type != kLiteRtElementTypeFloat32) {
    if (err) snprintf(err, err_len, "unexpected tensor element type %d", tt.element_type);
    return -1;
  }
  size_t ne = 1;
  for (unsigned i = 0; i < tt.layout.rank; i++) ne *= tt.layout.dimensions[i];
  if (is_in) TRY(LiteRtGetCompiledModelInputBufferRequirements(n->cm, 0, 0, &req), "in req");
  else TRY(LiteRtGetCompiledModelOutputBufferRequirements(n->cm, 0, 0, &req), "out req");
  TRY(LiteRtCreateManagedTensorBufferFromRequirements(env, &tt, req, is_in ? &n->in : &n->out),
      "tensor buffer");
  if (is_in) n->in_n = ne; else n->out_n = ne;
  return 0;
}

static int load_net(LiteRtEnvironment env, Net* n, const char* path, MimiBackend be, int threads,
                    char* err, size_t err_len) {
  LiteRtOptions opt;
  LiteRtOpaqueOptions oo;
  char toml[48];
  TRY(LiteRtCreateModelFromFile(env, path, &n->model), path);
  TRY(LiteRtCreateOptions(&opt), "options");
  if (be == MIMI_CPU) {
    TRY(LiteRtSetOptionsHardwareAccelerators(opt, kLiteRtHwAcceleratorCpu), "cpu accel");
    snprintf(toml, sizeof toml, "num_threads = %d\n", threads > 0 ? threads : 2);
    TRY(LiteRtCreateOpaqueOptions("xnnpack", dupstr(toml), free, &oo), "cpu options");
    TRY(LiteRtAddOpaqueOptions(opt, oo), "add cpu options");
  } else if (be == MIMI_GPU) {
    TRY(LiteRtSetOptionsHardwareAccelerators(opt, kLiteRtHwAcceleratorGpu), "gpu accel");
    snprintf(toml, sizeof toml, "precision = %d\n", kLiteRtDelegatePrecisionFp16);
    TRY(LiteRtCreateOpaqueOptions("gpu_options", dupstr(toml), free, &oo), "gpu options");
    TRY(LiteRtAddOpaqueOptions(opt, oo), "add gpu options");
  } else {
    TRY(LiteRtSetOptionsHardwareAccelerators(opt, kLiteRtHwAcceleratorNpu | kLiteRtHwAcceleratorCpu),
        "npu accel");
  }
  TRY(LiteRtCreateCompiledModel(env, n->model, opt, &n->cm), "compile");
  if (be != MIMI_CPU) {
    bool full = false;
    LiteRtCompiledModelIsFullyAccelerated(n->cm, &full);
    if (!full) {
      if (err) snprintf(err, err_len, "%s: %s backend did not take the whole graph", path,
                        be == MIMI_GPU ? "GPU" : "NPU");
      return -1;
    }
  }
  if (buf_for(env, n, 1, err, err_len) || buf_for(env, n, 0, err, err_len)) return -1;
  return 0;
}

static int run_net(Net* n, const float* x, float* y) {
  void* p;
  if (LiteRtLockTensorBuffer(n->in, &p, kLiteRtTensorBufferLockModeWrite)) return -1;
  memcpy(p, x, n->in_n * sizeof(float));
  LiteRtUnlockTensorBuffer(n->in);
  if (LiteRtRunCompiledModel(n->cm, 0, 1, &n->in, 1, &n->out)) return -1;
  if (LiteRtLockTensorBuffer(n->out, &p, kLiteRtTensorBufferLockModeRead)) return -1;
  memcpy(y, p, n->out_n * sizeof(float));
  LiteRtUnlockTensorBuffer(n->out);
  return 0;
}

static float* slurp_floats(const char* path, size_t want) {
  FILE* f = fopen(path, "rb");
  if (!f) return NULL;
  float* p = malloc(want * sizeof(float));
  size_t got = p ? fread(p, sizeof(float), want, f) : 0;
  fclose(f);
  if (got != want) { free(p); return NULL; }
  return p;
}

MimiDecoder* mimi_decoder_create(const MimiConfig* cfg, char* err, size_t err_len) {
  MimiDecoder* d = calloc(1, sizeof *d);
  LiteRtEnvOption eo[4];
  int neo = 0;
#define ENVOPT(tag_, s_)                                                                \
  if (s_) {                                                                             \
    eo[neo].tag = (tag_); eo[neo].value.type = kLiteRtAnyTypeString;                    \
    eo[neo].value.str_value = (s_); neo++;                                              \
  }
  ENVOPT(kLiteRtEnvOptionTagRuntimeLibraryDir, cfg->runtime_dir);
  ENVOPT(kLiteRtEnvOptionTagDispatchLibraryDir, cfg->npu_dir);
  ENVOPT(kLiteRtEnvOptionTagCompilerPluginLibraryDir, cfg->npu_dir);
  ENVOPT(kLiteRtEnvOptionTagCompilerCacheDir, cfg->cache_dir);
  char p[1024];
  if (LiteRtCreateEnvironment(neo, eo, &d->env) != kLiteRtStatusOk) {
    if (err) snprintf(err, err_len, "LiteRtCreateEnvironment failed");
    goto fail;
  }
  snprintf(p, sizeof p, "%s/mimi_dec_tx_fp16.tflite", cfg->model_dir);
  if (load_net(d->env, &d->tx, p, cfg->tx, cfg->threads, err, err_len)) goto fail;
  snprintf(p, sizeof p, "%s/mimi_deconly_fp16.tflite", cfg->model_dir);
  if (load_net(d->env, &d->conv, p, cfg->conv, cfg->threads, err, err_len)) goto fail;
  if (d->tx.in_n != H512 * MIMI_W || d->tx.out_n != d->conv.in_n || d->conv.out_n != WS) {
    if (err) snprintf(err, err_len, "unexpected graph shapes");
    goto fail;
  }
  snprintf(p, sizeof p, "%s/mimi_rvq.bin", cfg->model_dir);
  // sem_Win, aco_Win [256,512]; sem_Wout, aco_Wout [512,256]; sem_CB + 31 aco_CB [2048,256]
  d->rvq = slurp_floats(p, (size_t)4 * D * H512 + (size_t)MIMI_NQMAX * CB * D);
  if (!d->rvq) {
    if (err) snprintf(err, err_len, "cannot read %s", p);
    goto fail;
  }
  d->emb = malloc(H512 * MIMI_W * sizeof(float));
  d->mid = malloc(d->tx.out_n * sizeof(float));
  d->aud = malloc(WS * sizeof(float));
  d->last_k = -2;
  return d;
fail:
  mimi_decoder_destroy(d);
  return NULL;
}

void mimi_decoder_destroy(MimiDecoder* d) {
  if (!d) return;
  Net* nets[2] = {&d->tx, &d->conv};
  for (int i = 0; i < 2; i++) {
    if (nets[i]->in) LiteRtDestroyTensorBuffer(nets[i]->in);
    if (nets[i]->out) LiteRtDestroyTensorBuffer(nets[i]->out);
    if (nets[i]->cm) LiteRtDestroyCompiledModel(nets[i]->cm);
    if (nets[i]->model) LiteRtDestroyModel(nets[i]->model);
  }
  if (d->env) LiteRtDestroyEnvironment(d->env);
  free(d->rvq); free(d->emb); free(d->mid); free(d->aud);
  free(d);
}

void mimi_decoder_reset(MimiDecoder* d) { d->have_tail = 0; d->last_k = -2; }

static void rvq_window(const float* rvq, const int16_t* codes, int nq, float* emb) {
  const float* sem_Wout = rvq + 2 * D * H512;
  const float* aco_Wout = sem_Wout + H512 * D;
  const float* sem_CB = aco_Wout + H512 * D;
  const float* aco_CB = sem_CB + CB * D;
  float s[D], a[D];
  for (int t = 0; t < MIMI_W; t++) {
    const int16_t* c = codes + t * MIMI_NQMAX;
    memcpy(s, sem_CB + (size_t)(c[0] & (CB - 1)) * D, sizeof s);
    memset(a, 0, sizeof a);
    for (int i = 1; i < nq; i++) {
      const float* v = aco_CB + ((size_t)(i - 1) * CB + (c[i] & (CB - 1))) * D;
      for (int k = 0; k < D; k++) a[k] += v[k];
    }
    for (int h = 0; h < H512; h++) {
      const float* ws = sem_Wout + (size_t)h * D;
      const float* wa = aco_Wout + (size_t)h * D;
      float acc = 0.f;
      for (int k = 0; k < D; k++) acc += ws[k] * s[k] + wa[k] * a[k];
      emb[h * MIMI_W + t] = acc;
    }
  }
}

int mimi_decode_block(MimiDecoder* d, int k, const int16_t* codes, int nq, int valid, float* pcm) {
  if (valid < 0 || valid > MIMI_H || nq < 1 || nq > MIMI_NQMAX) return -1;
  double t0 = now_ms();
  rvq_window(d->rvq, codes, nq, d->emb);
  double t1 = now_ms();
  if (run_net(&d->tx, d->emb, d->mid)) return -1;
  double t2 = now_ms();
  if (run_net(&d->conv, d->mid, d->aud)) return -1;
  double t3 = now_ms();
  d->t_rvq = t1 - t0; d->t_tx = t2 - t1; d->t_conv = t3 - t2;

  const float* blk = d->aud + MIMI_C * MIMI_SPF;  // block start inside the window
  int n = valid * MIMI_SPF;
  memcpy(pcm, blk, n * sizeof(float));
  if (d->have_tail && d->last_k == k - 1) {
    int m = n < MIMI_XF ? n : MIMI_XF;
    for (int i = 0; i < m; i++) {
      float a = (i + 0.5f) / MIMI_XF;
      pcm[i] = (1.f - a) * d->tail[i] + a * blk[i];
    }
  }
  // lookahead frame = output frame k*H + H, i.e. window frame C + H
  memcpy(d->tail, d->aud + (MIMI_C + MIMI_H) * MIMI_SPF, sizeof d->tail);
  d->have_tail = valid == MIMI_H;
  d->last_k = k;
  return n;
}

void mimi_last_timing(const MimiDecoder* d, double* r, double* t, double* c) {
  if (r) *r = d->t_rvq;
  if (t) *t = d->t_tx;
  if (c) *c = d->t_conv;
}
