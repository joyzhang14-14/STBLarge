// STBLarge DeepSeek-V4.1 kernels (see kernels.cu). Activations are fp32 device buffers.
#pragma once
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void stbl_act_quant_fp8(float *x, int n, cudaStream_t st);
void stbl_act_quant_fp4(float *x, int n, int block, int e4m3_scale, cudaStream_t st);
void stbl_round_bf16(float *x, int n, cudaStream_t st);
void stbl_scale_bf16(float *x, float a, int n, cudaStream_t st);

void stbl_gemv_fp8(float *y, const void *w, const void *scale, const float *x, int rows, int k, int round_out,
                   cudaStream_t st);
void stbl_gemv_mxfp4(float *y, const void *w, const float *x, int rows, int k, cudaStream_t st);
void stbl_gemv_bf16(float *y, const void *w, const float *x, int rows, int k, int round_out, int group_rows,
                    cudaStream_t st);
void stbl_gemv_f32(float *y, const float *w, const float *x, int rows, int k, int round_out, cudaStream_t st);

void stbl_rms_norm(float *out, const float *x, const float *weight, int rows, int dim, float eps, cudaStream_t st);
void stbl_rope(float *x, const float *cos_sin, int heads, int width, int rope, int inverse, cudaStream_t st);
void stbl_hc_mixes(float *pre, float *post, float *comb, const float *x, const float *fn, const float *scale,
                   const float *base, int hc, int dim, int iters, float norm_eps, float hc_eps, cudaStream_t st);
void stbl_hc_pre(float *out, const float *x, const float *pre, int hc, int dim, cudaStream_t st);
void stbl_hc_post(float *out, const float *x, const float *res, const float *post, const float *comb, int hc,
                  int dim, cudaStream_t st);
void stbl_sparse_attn(float *out, const float *q, const float *kv1, int n1, const float *kv2, const int *idx,
                      int n_idx, const float *sink, int heads, int d, float scale, cudaStream_t st);
void stbl_sqrt_softplus(float *s, int n, cudaStream_t st);
void stbl_swiglu(float *out, const float *g, const float *u, float weight, float limit, int n, cudaStream_t st);
void stbl_axpy(float *y, const float *x, float a, int n, cudaStream_t st);
void stbl_embed(float *h, const void *table, int token, int dim, int hc, cudaStream_t st);
void stbl_engram_rows(float *out, const void *rows, int n_rows, cudaStream_t st);
void stbl_engram_gate(float *h, const float *kv, const float *qw, const float *kw, int dim, int hc, float eps,
                      cudaStream_t st);
void stbl_index_score(float *score, const float *q, const float *k, const float *w, int n, int heads, int d,
                      cudaStream_t st);
void stbl_compress_pool(float *out, const float *kv, const float *score, int ratio, int width, cudaStream_t st);

#ifdef __cplusplus
}
#endif
