// STBLarge DeepSeek-V4.1 engine, C API.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct stbl_model stbl_model;

// Open the STBLarge GGUF; ctx is the maximum sequence length, expert_slots the GPU expert cache size.
// Returns NULL and fills err on failure.
stbl_model *stbl_open(const char *path, int ctx, int expert_slots, char *err, int errlen);
// Run one token at the next position. Writes fp32 logits when logits != NULL; returns the argmax
// token, or -1 on error (see stbl_error).
int stbl_step(stbl_model *m, int token, float *logits);
int stbl_vocab(const stbl_model *m);
const char *stbl_error(const stbl_model *m);
void stbl_stats(const stbl_model *m, uint64_t *expert_hits, uint64_t *expert_misses);
// Test hook: run Engram (if layer il has one) and block il alone at position pos, on the given residual
// stream h_in (hc*dim fp32) and incoming pre_mix (hc); writes h_out and the outgoing pre_mix. Only that
// layer's caches advance, so call it for positions 0, 1, 2, ... of one sequence. Returns 0, or -1.
int stbl_block_test(stbl_model *m, int il, int pos, int token, const float *h_in, const float *pre_in,
                    float *h_out, float *pre_out);
void stbl_close(stbl_model *m);

#ifdef __cplusplus
}
#endif
