// mimi_play — decode a `.mimi` file (or a time range of it) to WAV with libmimi, timing every block.
// The --start case is a seek: decoding begins cold at the block containing that time.
//
// usage: mimi_play FILE.mimi OUT.wav [--start S] [--dur S] [--tx cpu|gpu|npu] [--conv cpu|gpu|npu]
//                  [--models DIR] [--libdir DIR] [--npudir DIR] [--cache DIR] [--threads N] [--pace S]
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "mimi.h"

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

static MimiBackend be(const char* s) {
  return !strcmp(s, "npu") ? MIMI_NPU : !strcmp(s, "gpu") ? MIMI_GPU : MIMI_CPU;
}

static void write_wav(const char* path, const float* x, size_t n) {
  FILE* f = fopen(path, "wb");
  uint32_t sr = 24000, br = 48000, sz = (uint32_t)(n * 2), fmt = 16, riff = 36 + sz;
  uint16_t one = 1, bits = 16, ba = 2;
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

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: mimi_play FILE.mimi OUT.wav [opts]\n"); return 2; }
  const char *in = argv[1], *out = argv[2];
  MimiConfig cfg = {.model_dir = ".", .tx = MIMI_NPU, .conv = MIMI_GPU, .threads = 2};
  double start = 0, dur = 0, pace = 0;
  for (int i = 3; i < argc; i++) {
    if (!strcmp(argv[i], "--start")) start = atof(argv[++i]);
    else if (!strcmp(argv[i], "--dur")) dur = atof(argv[++i]);
    else if (!strcmp(argv[i], "--tx")) cfg.tx = be(argv[++i]);
    else if (!strcmp(argv[i], "--conv")) cfg.conv = be(argv[++i]);
    else if (!strcmp(argv[i], "--models")) cfg.model_dir = argv[++i];
    else if (!strcmp(argv[i], "--libdir")) cfg.runtime_dir = argv[++i];
    else if (!strcmp(argv[i], "--npudir")) cfg.npu_dir = argv[++i];
    else if (!strcmp(argv[i], "--cache")) cfg.cache_dir = argv[++i];
    else if (!strcmp(argv[i], "--threads")) cfg.threads = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--pace")) pace = atof(argv[++i]);
    else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
  }
  int fd = open(in, O_RDONLY);
  struct stat st;
  if (fd < 0 || fstat(fd, &st)) { perror(in); return 1; }
  const uint8_t* map = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  MimiInfo info;
  if (mimi_parse_header(map, st.st_size, &info)) { fprintf(stderr, "bad header\n"); return 1; }
  if (info.data_offset + (size_t)info.n_frames * info.frame_bytes > (size_t)st.st_size) {
    fprintf(stderr, "truncated file\n"); return 1;
  }
  const uint8_t* frames = map + info.data_offset;
  fprintf(stderr, "%s: %u frames (%.1f s), nq=%d, model %s, meta %.*s\n", in, info.n_frames,
          info.n_frames / 12.5, info.nq, info.model, (int)info.meta_len, map + 64);

  char err[512];
  double t0 = now();
  MimiDecoder* d = mimi_decoder_create(&cfg, err, sizeof err);
  if (!d) { fprintf(stderr, "decoder: %s\n", err); return 1; }
  fprintf(stderr, "decoder ready in %.0f ms\n", 1e3 * (now() - t0));

  int nb = mimi_num_blocks(info.n_frames);
  int k0 = (int)(start * 12.5) / MIMI_H, k1 = nb;
  if (dur > 0) {
    int kd = (int)ceil(dur * 12.5 / MIMI_H);
    if (k0 + kd < k1) k1 = k0 + kd;
  }
  if (k0 >= nb) { fprintf(stderr, "start beyond end\n"); return 1; }
  float* pcm = malloc((size_t)(k1 - k0) * MIMI_H * MIMI_SPF * sizeof(float));
  int16_t win[MIMI_W * MIMI_NQMAX];
  size_t n = 0;
  double tseek = now(), first = 0, worst = 0, sum = 0;
  for (int k = k0; k < k1; k++) {
    double tb = now();
    int ws, valid;
    mimi_block_span(info.n_frames, k, &ws, &valid);
    for (int t = 0; t < MIMI_W; t++) {  // clamp to [0, n_frames)
      int f = ws + t;
      f = f < 0 ? 0 : f >= (int)info.n_frames ? (int)info.n_frames - 1 : f;
      mimi_unpack_frames(frames + (size_t)f * info.frame_bytes, 1, &info, win + t * MIMI_NQMAX);
    }
    int got = mimi_decode_block(d, k, win, info.nq, valid, pcm + n);
    if (got < 0) { fprintf(stderr, "decode failed at block %d\n", k); return 1; }
    n += got;
    double el = now() - tb;
    if (k == k0) first = now() - tseek;
    if (el > worst) worst = el;
    sum += el;
    if (pace > 0 && pace > el) usleep((useconds_t)((pace - el) * 1e6));
  }
  int nblk = k1 - k0;
  fprintf(stderr, "decoded %.1f s from %.2f s in %d blocks: first audio %.1f ms after seek, "
          "mean %.1f ms/block, worst %.1f ms (block = %.2f s; RTF %.4f)\n",
          n / 24000.0, k0 * MIMI_H / 12.5, nblk, 1e3 * first, 1e3 * sum / nblk, 1e3 * worst,
          MIMI_H / 12.5, sum / (n / 24000.0));
  write_wav(out, pcm, n);
  mimi_decoder_destroy(d);
  return 0;
}
