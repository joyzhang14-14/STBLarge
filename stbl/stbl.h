// STBLarge DeepSeek-V4.1 engine, C API.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct stbl_model stbl_model;

// Open the STBLarge GGUF; ctx is the maximum sequence length, expert_slots the GPU expert cache size
// (<= 0: as many as fit in MemAvailable less 12 GiB).
// Returns NULL and fills err on failure.
stbl_model *stbl_open(const char *path, int ctx, int expert_slots, char *err, int errlen);
// Run one token at the next position. Writes fp32 logits when logits != NULL; returns the argmax
// token, or -1 on error (see stbl_error).
int stbl_step(stbl_model *m, int token, float *logits);
int stbl_vocab(const stbl_model *m);
const char *stbl_error(const stbl_model *m);
void stbl_stats(const stbl_model *m, uint64_t *expert_hits, uint64_t *expert_misses);
// STBL_LOOKAHEAD counters: uncached guesses, of them routed, misses at guessed layers, of them guessed.
void stbl_lookahead(const stbl_model *m, uint64_t *out);
// Test hook: run Engram (if layer il has one) and block il alone at position pos, on the given residual
// stream h_in (hc*dim fp32) and incoming pre_mix (hc); writes h_out and the outgoing pre_mix. Only that
// layer's caches advance, so call it for positions 0, 1, 2, ... of one sequence. Returns 0, or -1.
// stage 0: Engram + block, 1: Engram only (h_out = stream after Engram), 2: block only,
// 3: attention only, 4: MoE only (3/4: h_in = the normed sublayer input, h_out = its output, dim floats).
int stbl_block_test(stbl_model *m, int il, int pos, int token, const float *h_in, const float *pre_in,
                    float *h_out, float *pre_out, int stage);
// STBL_TIMING totals in seconds, in the order hc/norm, engram, attention, router, expert read,
// expert compute, shared expert, head; returns the number of sections.
int stbl_timing(const stbl_model *m, double *sec, int n);
// Routed expert ids of the last MoE call, top-k order; returns top-k.
int stbl_last_route(const stbl_model *m, int *ids);
void stbl_close(stbl_model *m);

#ifdef __cplusplus
}
#endif
