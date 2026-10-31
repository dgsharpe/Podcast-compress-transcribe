// libmimi — `.mimi` file parsing and block decoding for Android (LiteRT CompiledModel).
//
// Decoding is block-based. Block k covers output frames [k*H, k*H + H) and is decoded from the 25-frame
// window [k*H - C, k*H + H + 1): C frames of left context, H new frames, 1 frame of lookahead.
// Frames outside [0, n_frames) are clamped to the first/last frame. The lookahead frame's audio is held
// back and crossfaded (XF samples) into the start of block k+1 if that block comes next; after a seek
// (any non-consecutive block) there is no crossfade. Each block needs only its own window, so playback
// can start at any block.
#ifndef MIMI_H_
#define MIMI_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MIMI_W 25          // frames per model window (fixed by the graphs)
#define MIMI_C 6           // left-context frames
#define MIMI_H 18          // new frames per block (1.44 s)
#define MIMI_L 1           // lookahead frames (crossfade source)
#define MIMI_SPF 1920      // samples per frame at 24 kHz / 12.5 Hz
#define MIMI_XF 480        // crossfade length in samples (20 ms)
#define MIMI_NQMAX 32
#define MIMI_HEADER_BYTES 64

typedef struct {
  int version, nq, bits, sample_rate, fr_num, fr_den;
  uint32_t n_frames, frame_bytes, data_offset, meta_len;
  char model[33];
} MimiInfo;

// Parse the fixed 64-byte header. Returns 0 on success, -1 on a bad or unsupported header.
int mimi_parse_header(const uint8_t* buf, size_t len, MimiInfo* info);

// Unpack `n` frames of `frame_bytes` each into codes[n][MIMI_NQMAX] (unused codebooks are zeroed).
void mimi_unpack_frames(const uint8_t* frames, int n, const MimiInfo* info, int16_t* codes);

// Number of blocks for a file, and the first frame / valid frame count of block k.
int mimi_num_blocks(uint32_t n_frames);
// For block k: the window's first frame (may be negative), and the number of valid output frames.
void mimi_block_span(uint32_t n_frames, int k, int* window_start, int* valid_frames);

typedef enum { MIMI_CPU = 0, MIMI_GPU = 1, MIMI_NPU = 2 } MimiBackend;

typedef struct {
  const char* model_dir;    // mimi_dec_tx_fp16.tflite, mimi_deconly_fp16.tflite, mimi_rvq.bin
  const char* runtime_dir;  // dir holding libLiteRtClGlAccelerator.so (NULL = default search)
  const char* npu_dir;      // dir holding libLiteRtDispatch_Qualcomm.so etc. (NULL = no NPU)
  const char* cache_dir;    // NPU compile cache (NULL = none)
  MimiBackend tx, conv;     // recommended: tx = NPU (or CPU), conv = GPU
  int threads;              // CPU threads for CPU-backed graphs (2 is best on the Fold 8)
} MimiConfig;

typedef struct MimiDecoder MimiDecoder;

// Returns NULL on failure (the message is in err, if non-NULL).
MimiDecoder* mimi_decoder_create(const MimiConfig* cfg, char* err, size_t err_len);
void mimi_decoder_destroy(MimiDecoder* d);

// Forget the crossfade tail (call on seek). Decoding a non-consecutive block also resets implicitly.
void mimi_decoder_reset(MimiDecoder* d);

// Decode block k from its 25-frame window codes[MIMI_W][MIMI_NQMAX] (frames already clamped), using
// the first nq codebooks. Writes valid_frames * MIMI_SPF float samples to pcm and returns that count,
// or -1 on error.
int mimi_decode_block(MimiDecoder* d, int k, const int16_t* codes, int nq, int valid_frames,
                      float* pcm);

// Per-stage time of the last call, in milliseconds: rvq, tx, conv.
void mimi_last_timing(const MimiDecoder* d, double* rvq_ms, double* tx_ms, double* conv_ms);

#ifdef __cplusplus
}
#endif
#endif  // MIMI_H_
