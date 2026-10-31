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
void stbl_close(stbl_model *m);

#ifdef __cplusplus
}
#endif
