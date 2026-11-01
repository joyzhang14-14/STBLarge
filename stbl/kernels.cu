// STBLarge DeepSeek-V4.1 kernels. Every activation buffer is fp32; where the official model keeps a
// tensor in bf16 the producing kernel rounds to bf16 (round to nearest even), so the values match the
// official inference/model.py. Decode only (one token): matrix products are GEMVs.
#include "kernels.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// 2^ceil(log2(t)) for positive normal t, by exponent/mantissa bits (the official fast_log2_ceil).
__device__ __forceinline__ float pow2_ceil(float t) {
    const uint32_t b = __float_as_uint(t);
    const int e = (int)((b >> 23) & 0xff) - 127 + ((b & 0x7fffffu) != 0);
    return __uint_as_float((uint32_t)(e + 127) << 23);
}

__device__ __forceinline__ float e8m0(uint8_t s) { return __uint_as_float((uint32_t)s << 23); }

__device__ __forceinline__ float e4m3_value(uint8_t c) {
    __nv_fp8_e4m3 v;
    v.__x = c;
    return (float)v;
}

__device__ __forceinline__ float e4m3_round(float x) {  // RNE to e4m3, x already within +-448
    return (float)__nv_fp8_e4m3(x);
}

__constant__ float kFp4[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

__device__ __forceinline__ float fp4_round(float v) {  // v within [-6, 6], ties to even on the e2m1 grid
    const float a = fabsf(v);
    const float step = a < 2.f ? .5f : (a < 4.f ? 1.f : 2.f);
    return copysignf(rintf(a / step) * step, v);
}

// ---- quantization -------------------------------------------------------------------------------

// act_quant(x, 32, ue8m0, inplace): per 32 values, scale = 2^ceil(log2(max(amax,1e-4)/448)),
// value -> e4m3(x/scale)*scale. One warp per block of 32.
__global__ void k_act_quant_fp8(float *x, int n_blocks) {
    const int b = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (b >= n_blocks) return;
    float v = x[b * 32 + lane];
    const float amax = fmaxf(warp_max(fabsf(v)), 1e-4f);
    const float s = pow2_ceil(amax * (1.0f / 448.0f));
    x[b * 32 + lane] = bf16r(e4m3_round(fminf(fmaxf(v / s, -448.f), 448.f)) * s);
}

// fp4_act_quant(x, block, inplace): E8M0 scales (block 32, indexer) or E4M3 scales (block 16,
// compressed KV). Output rounded to bf16 like the in-place official kernel. One warp per 32 values.
__global__ void k_act_quant_fp4(float *x, int n, int block, int e4m3_scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    float v = i < n ? x[i] : 0.f;
    float a = fabsf(v);
    // max over each `block` lanes (16 or 32)
    for (int o = block / 2; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    (void)lane;
    float s;
    if (e4m3_scale) {
        a = fmaxf(a, 6.f * exp2f(-9.f));
        s = e4m3_round(a / 6.f);
    } else {
        a = fmaxf(a, 6.f * exp2f(-126.f));
        s = pow2_ceil(a * (1.0f / 6.0f));
    }
    if (i < n) x[i] = bf16r(fp4_round(fminf(fmaxf(v / s, -6.f), 6.f)) * s);
}

__global__ void k_round_bf16(float *x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = bf16r(x[i]);
}

// ---- GEMV ---------------------------------------------------------------------------------------
// y[r] = sum_k x[k] * W[r][k]. One warp per output row; 8 rows per 256-thread block.

__global__ void k_gemv_fp8(float *y, const uint8_t *w, const uint8_t *scale, const float *x,
                           int rows, int k, int round_out) {
    const int r = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (r >= rows) return;
    const uint8_t *wr = w + (size_t)r * k;
    const uint8_t *sr = scale + (size_t)(r / 32) * (k / 32);
    float acc = 0.f;
    for (int kb = 0; kb < k / 32; kb++) {
        const float p = x[kb * 32 + lane] * e4m3_value(wr[kb * 32 + lane]);
        acc += warp_sum(p) * e8m0(sr[kb]);
    }
    if (lane == 0) y[r] = round_out ? bf16r(acc) : acc;
}

// MXFP4 rows as GGUF blocks of 17 bytes: E8M0 scale, then 16 bytes holding values 0..15 in the low
// nibbles and 16..31 in the high nibbles.
__global__ void k_gemv_mxfp4(float *y, const uint8_t *w, const float *x, int rows, int k) {
    const int r = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (r >= rows) return;
    const uint8_t *wr = w + (size_t)r * (k / 32) * 17;
    float acc = 0.f;
    for (int kb = 0; kb < k / 32; kb++) {
        const uint8_t *blk = wr + kb * 17;
        const uint8_t byte = blk[1 + (lane & 15)];
        const float v = kFp4[lane < 16 ? (byte & 15) : (byte >> 4)];
        acc += warp_sum(x[kb * 32 + lane] * v) * e8m0(blk[0]);
    }
    if (lane == 0) y[r] = bf16r(acc);
}

// group_rows > 0: row r reads x + (r / group_rows) * k (the block-diagonal wo_a einsum).
__global__ void k_gemv_bf16(float *y, const __nv_bfloat16 *w, const float *x, int rows, int k, int round_out,
                            int group_rows) {
    const int r = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (r >= rows) return;
    const __nv_bfloat16 *wr = w + (size_t)r * k;
    if (group_rows > 0) x += (size_t)(r / group_rows) * k;
    float acc = 0.f;
    for (int i = lane; i < k; i += 32) acc += x[i] * __bfloat162float(wr[i]);
    acc = warp_sum(acc);
    if (lane == 0) y[r] = round_out ? bf16r(acc) : acc;
}

__global__ void k_gemv_f32(float *y, const float *w, const float *x, int rows, int k, int round_out) {
    const int r = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (r >= rows) return;
    const float *wr = w + (size_t)r * k;
    float acc = 0.f;
    for (int i = lane; i < k; i += 32) acc += x[i] * wr[i];
    acc = warp_sum(acc);
    if (lane == 0) y[r] = round_out ? bf16r(acc) : acc;
}

// ---- elementwise / small ----------------------------------------------------------------------------

// RMSNorm over `dim` for `rows` rows: out = bf16(weight * x * rsqrt(mean(x^2) + eps)). One block per row.
__global__ void k_rms_norm(float *out, const float *x, const float *weight, int dim, float eps) {
    const float *xr = x + (size_t)blockIdx.x * dim;
    float *outr = out + (size_t)blockIdx.x * dim;
    __shared__ float part[32];
    float s = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) s += xr[i] * xr[i];
    s = warp_sum(s);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x / 32] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = threadIdx.x < blockDim.x / 32 ? part[threadIdx.x] : 0.f;
        t = warp_sum(t);
        if (threadIdx.x == 0) part[0] = t;
    }
    __syncthreads();
    const float r = rsqrtf(part[0] / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) outr[i] = bf16r(weight[i] * (xr[i] * r));
}

// Rotary embedding on the last `rope` dims of each of `heads` rows of width `width`: adjacent pairs
// are complex numbers multiplied by (cos, sin) (conjugated when inverse); result rounded to bf16.
__global__ void k_rope(float *x, const float *cos_sin, int heads, int width, int rope, int inverse) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;  // pair index
    const int pairs = rope / 2;
    if (i >= heads * pairs) return;
    const int h = i / pairs, p = i % pairs;
    float *v = x + (size_t)h * width + (width - rope) + 2 * p;
    const float c = cos_sin[2 * p], s = inverse ? -cos_sin[2 * p + 1] : cos_sin[2 * p + 1];
    const float a = v[0], b = v[1];
    v[0] = bf16r(a * c - b * s);
    v[1] = bf16r(a * s + b * c);
}

// Hyper-connection mixes: mixes[m] = (hc_fn[m] . x) * rsqrt(mean(x^2) + eps) over the flattened
// hc*dim stream, then the official hc_split_sinkhorn. One block of 256 threads.
__global__ void k_hc_mixes(float *pre, float *post, float *comb, const float *x, const float *fn,
                           const float *scale, const float *base, int hc, int dim, int iters,
                           float norm_eps, float hc_eps) {
    const int n = hc * dim, mix = (2 + hc) * hc;
    __shared__ float part[8][32];
    __shared__ float mixes[64];
    __shared__ float sq;
    // sum of squares
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += x[i] * x[i];
    s = warp_sum(s);
    if ((threadIdx.x & 31) == 0) part[0][threadIdx.x / 32] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int w = 0; w < (int)blockDim.x / 32; w++) t += part[0][w];
        sq = rsqrtf(t / n + norm_eps);
    }
    __syncthreads();
    // one warp per mix row
    for (int m = threadIdx.x / 32; m < mix; m += blockDim.x / 32) {
        float a = 0.f;
        for (int i = threadIdx.x & 31; i < n; i += 32) a += fn[(size_t)m * n + i] * x[i];
        a = warp_sum(a);
        if ((threadIdx.x & 31) == 0) mixes[m] = a * sq;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    for (int j = 0; j < hc; j++) {
        pre[j] = 1.f / (1.f + expf(-__fadd_rn(__fmul_rn(mixes[j], scale[0]), base[j]))) + hc_eps;
        post[j] = 2.f / (1.f + expf(-__fadd_rn(__fmul_rn(mixes[j + hc], scale[1]), base[j + hc])));
    }
    float c[16];
    for (int j = 0; j < hc; j++) {
        float mx = -INFINITY;
        for (int k = 0; k < hc; k++) {
            c[j * hc + k] = __fadd_rn(__fmul_rn(mixes[2 * hc + j * hc + k], scale[2]), base[2 * hc + j * hc + k]);
            mx = fmaxf(mx, c[j * hc + k]);
        }
        float sum = 0.f;
        for (int k = 0; k < hc; k++) { c[j * hc + k] = expf(c[j * hc + k] - mx); sum += c[j * hc + k]; }
        for (int k = 0; k < hc; k++) c[j * hc + k] = c[j * hc + k] / sum + hc_eps;
    }
    for (int it = 0; it < iters; it++) {
        if (it > 0)
            for (int j = 0; j < hc; j++) {
                float sum = 0.f;
                for (int k = 0; k < hc; k++) sum += c[j * hc + k];
                for (int k = 0; k < hc; k++) c[j * hc + k] /= sum + hc_eps;
            }
        for (int k = 0; k < hc; k++) {
            float sum = 0.f;
            for (int j = 0; j < hc; j++) sum += c[j * hc + k];
            for (int j = 0; j < hc; j++) c[j * hc + k] /= sum + hc_eps;
        }
    }
    for (int i = 0; i < hc * hc; i++) comb[i] = c[i];
}

// hc_pre: out[d] = bf16(sum_j pre[j] * x[j][d]). Like torch: each product rounded, summed in j order
// (no FMA contraction), so the result matches the reference bit for bit.
__global__ void k_hc_pre(float *out, const float *x, const float *pre, int hc, int dim) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    float s = 0.f;
    for (int j = 0; j < hc; j++) s = __fadd_rn(s, __fmul_rn(pre[j], x[(size_t)j * dim + d]));
    out[d] = bf16r(s);
}

// hc_post: out[j][d] = bf16(post[j] * x[d] + sum_i comb[i][j] * res[i][d]), rounded step by step like
// torch (see hc_pre).
__global__ void k_hc_post(float *out, const float *x, const float *res, const float *post, const float *comb,
                          int hc, int dim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hc * dim) return;
    const int j = i / dim, d = i % dim;
    float s = 0.f;
    for (int a = 0; a < hc; a++) s = __fadd_rn(s, __fmul_rn(comb[a * hc + j], res[(size_t)a * dim + d]));
    out[i] = bf16r(__fadd_rn(__fmul_rn(post[j], x[d]), s));
}

// Sparse decode attention: q [heads][d]; idx picks rows (-1 = none) from the window cache kv1 (idx < n1)
// or the compressed cache kv2 (row idx - n1); sink per head.
// Probabilities are rounded to bf16 before the value product, output rounded to bf16. One block per head.
__global__ void k_sparse_attn(float *out, const float *q, const float *kv1, int n1, const float *kv2,
                              const int *idx, int n_idx, const float *sink, int d, float scale) {
    extern __shared__ float sm[];
    float *qs = sm, *sc = sm + d;  // sc: n_idx scores
    const int h = blockIdx.x;
    for (int i = threadIdx.x; i < d; i += blockDim.x) qs[i] = q[(size_t)h * d + i];
    __syncthreads();
    const int warps = blockDim.x / 32, lane = threadIdx.x & 31;
    for (int t = threadIdx.x / 32; t < n_idx; t += warps) {
        const int r = idx[t];
        const float *row = r < n1 ? kv1 + (size_t)r * d : kv2 + (size_t)(r - n1) * d;
        float a = 0.f;
        if (r >= 0)
            for (int i = lane; i < d; i += 32) a += qs[i] * row[i];
        a = warp_sum(a);
        if (lane == 0) sc[t] = r >= 0 ? a * scale : -INFINITY;
    }
    __syncthreads();
    __shared__ float red[2];
    if (threadIdx.x == 0) {
        float mx = -1e30f;
        for (int t = 0; t < n_idx; t++) mx = fmaxf(mx, sc[t]);
        float sum = 0.f;
        for (int t = 0; t < n_idx; t++) { sc[t] = expf(sc[t] - mx); sum += sc[t]; }
        red[0] = sum + expf(sink[h] - mx);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float a = 0.f;
        for (int t = 0; t < n_idx; t++) {
            const int r = idx[t];
            if (r >= 0) a += bf16r(sc[t]) * (r < n1 ? kv1[(size_t)r * d + i] : kv2[(size_t)(r - n1) * d + i]);
        }
        out[(size_t)h * d + i] = bf16r(a / red[0]);
    }
}

// Router scores: s = sqrt(softplus(logit)) (softplus threshold 20 as torch).
__global__ void k_sqrt_softplus(float *s, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = s[i];
    s[i] = sqrtf(x > 20.f ? x : log1pf(expf(x)));
}

// SwiGLU with the training clamps: out = bf16(silu(min(g, lim)) * clamp(u, -lim, lim) * w).
__global__ void k_swiglu(float *out, const float *g, const float *u, float weight, float limit, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gv = g[i], uv = u[i];
    if (limit > 0) { uv = fminf(fmaxf(uv, -limit), limit); gv = fminf(gv, limit); }
    out[i] = bf16r(gv / (1.f + expf(-gv)) * uv * weight);
}

__global__ void k_axpy(float *y, const float *x, float a, int n) {  // y += a*x (fp32)
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}

__global__ void k_scale_bf16(float *x, float a, int n) {  // x = bf16(x * a)
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = bf16r(x[i] * a);
}

// Engram rows as stored: 256 E4M3 bytes then 8 E8M0 scales (one per 32); out = bf16(value * scale).
__global__ void k_engram_rows(float *out, const uint8_t *rows, int n_rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows * 256) return;
    const uint8_t *r = rows + (size_t)(i / 256) * 264;
    const int c = i % 256;
    out[i] = bf16r(e4m3_value(r[c]) * e8m0(r[256 + c / 32]));
}

// Engram gate, one block per hc copy j: key = kv[j*dim..], value = kv[hc*dim..];
// rstd = rsqrt(mean(h^2)+eps) * rsqrt(mean(key^2)+eps); dot = sum(h*w*key)*rstd/sqrt(dim);
// gate = sigmoid(sign(dot)*sqrt(max(|dot|,1e-6))); h = bf16(h + gate*value).
__global__ void k_engram_gate(float *h, const float *kv, const float *qw, const float *kw, int dim, int hc,
                              float eps, float inv_sqrt_dim) {
    const int j = blockIdx.x;
    float *hj = h + (size_t)j * dim;
    const float *key = kv + (size_t)j * dim, *value = kv + (size_t)hc * dim;
    __shared__ float part[3][8];
    float hh = 0.f, kk = 0.f, dot = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float w = qw[(size_t)j * dim + i] * kw[(size_t)j * dim + i];
        hh += hj[i] * hj[i];
        kk += key[i] * key[i];
        dot += hj[i] * w * key[i];
    }
    hh = warp_sum(hh); kk = warp_sum(kk); dot = warp_sum(dot);
    if ((threadIdx.x & 31) == 0) { part[0][threadIdx.x / 32] = hh; part[1][threadIdx.x / 32] = kk; part[2][threadIdx.x / 32] = dot; }
    __syncthreads();
    __shared__ float gate;
    if (threadIdx.x == 0) {
        float a = 0, b = 0, c = 0;
        for (int w = 0; w < (int)blockDim.x / 32; w++) { a += part[0][w]; b += part[1][w]; c += part[2][w]; }
        const float rstd = rsqrtf(a / dim + eps) * rsqrtf(b / dim + eps);
        const float d = c * rstd * inv_sqrt_dim;
        gate = 1.f / (1.f + expf(-copysignf(sqrtf(fmaxf(fabsf(d), 1e-6f)), d)));
    }
    __syncthreads();
    for (int i = threadIdx.x; i < dim; i += blockDim.x) hj[i] = bf16r(hj[i] + gate * value[i]);
}

// Indexer scores: score[t] = bf16(sum_h bf16(relu(bf16(q_h . k_t)) * w_h)), one warp per position t.
__global__ void k_index_score(float *score, const float *q, const float *k, const float *w, int n, int heads, int d) {
    const int t = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (t >= n) return;
    float acc = 0.f;
    for (int h = 0; h < heads; h++) {
        float a = 0.f;
        for (int i = lane; i < d; i += 32) a += q[(size_t)h * d + i] * k[(size_t)t * d + i];
        a = bf16r(warp_sum(a));
        acc += bf16r(fmaxf(a, 0.f) * w[h]);
    }
    if (lane == 0) score[t] = bf16r(acc);
}

// Compressor pooling over a completed group: out[c] = bf16(sum_r kv[r][c] * softmax_r(score[.][c])).
__global__ void k_compress_pool(float *out, const float *kv, const float *score, int ratio, int width) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= width) return;
    float mx = -INFINITY;
    for (int r = 0; r < ratio; r++) mx = fmaxf(mx, score[(size_t)r * width + c]);
    float sum = 0.f, acc = 0.f;
    for (int r = 0; r < ratio; r++) sum += expf(score[(size_t)r * width + c] - mx);
    for (int r = 0; r < ratio; r++) acc += kv[(size_t)r * width + c] * (expf(score[(size_t)r * width + c] - mx) / sum);
    out[c] = bf16r(acc);
}

__global__ void k_embed(float *h, const __nv_bfloat16 *table, int token, int dim, int hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    const float v = __bfloat162float(table[(size_t)token * dim + i]);
    for (int j = 0; j < hc; j++) h[(size_t)j * dim + i] = v;
}

}  // namespace

// ---- host wrappers --------------------------------------------------------------------------------

#define GRID(n, b) (((n) + (b) - 1) / (b))

extern "C" {

void stbl_act_quant_fp8(float *x, int n, cudaStream_t st) {
    k_act_quant_fp8<<<GRID(n / 32, 8), 256, 0, st>>>(x, n / 32);
}
void stbl_act_quant_fp4(float *x, int n, int block, int e4m3_scale, cudaStream_t st) {
    k_act_quant_fp4<<<GRID(n, 256), 256, 0, st>>>(x, n, block, e4m3_scale);
}
void stbl_round_bf16(float *x, int n, cudaStream_t st) { k_round_bf16<<<GRID(n, 256), 256, 0, st>>>(x, n); }
void stbl_gemv_fp8(float *y, const void *w, const void *scale, const float *x, int rows, int k, int round_out,
                   cudaStream_t st) {
    k_gemv_fp8<<<GRID(rows, 8), 256, 0, st>>>(y, (const uint8_t *)w, (const uint8_t *)scale, x, rows, k, round_out);
}
void stbl_gemv_mxfp4(float *y, const void *w, const float *x, int rows, int k, cudaStream_t st) {
    k_gemv_mxfp4<<<GRID(rows, 8), 256, 0, st>>>(y, (const uint8_t *)w, x, rows, k);
}
void stbl_gemv_bf16(float *y, const void *w, const float *x, int rows, int k, int round_out, int group_rows,
                    cudaStream_t st) {
    k_gemv_bf16<<<GRID(rows, 8), 256, 0, st>>>(y, (const __nv_bfloat16 *)w, x, rows, k, round_out, group_rows);
}
void stbl_scale_bf16(float *x, float a, int n, cudaStream_t st) { k_scale_bf16<<<GRID(n, 256), 256, 0, st>>>(x, a, n); }
void stbl_engram_rows(float *out, const void *rows, int n_rows, cudaStream_t st) {
    k_engram_rows<<<GRID(n_rows * 256, 256), 256, 0, st>>>(out, (const uint8_t *)rows, n_rows);
}
void stbl_engram_gate(float *h, const float *kv, const float *qw, const float *kw, int dim, int hc, float eps,
                      cudaStream_t st) {
    k_engram_gate<<<hc, 256, 0, st>>>(h, kv, qw, kw, dim, hc, eps, (float)(1.0 / sqrt((double)dim)));
}
void stbl_index_score(float *score, const float *q, const float *k, const float *w, int n, int heads, int d,
                      cudaStream_t st) {
    k_index_score<<<GRID(n, 8), 256, 0, st>>>(score, q, k, w, n, heads, d);
}
void stbl_compress_pool(float *out, const float *kv, const float *score, int ratio, int width, cudaStream_t st) {
    k_compress_pool<<<GRID(width, 256), 256, 0, st>>>(out, kv, score, ratio, width);
}
void stbl_gemv_f32(float *y, const float *w, const float *x, int rows, int k, int round_out, cudaStream_t st) {
    k_gemv_f32<<<GRID(rows, 8), 256, 0, st>>>(y, w, x, rows, k, round_out);
}
void stbl_rms_norm(float *out, const float *x, const float *weight, int rows, int dim, float eps, cudaStream_t st) {
    k_rms_norm<<<rows, 256, 0, st>>>(out, x, weight, dim, eps);
}
void stbl_rope(float *x, const float *cos_sin, int heads, int width, int rope, int inverse, cudaStream_t st) {
    k_rope<<<GRID(heads * rope / 2, 256), 256, 0, st>>>(x, cos_sin, heads, width, rope, inverse);
}
void stbl_hc_mixes(float *pre, float *post, float *comb, const float *x, const float *fn, const float *scale,
                   const float *base, int hc, int dim, int iters, float norm_eps, float hc_eps, cudaStream_t st) {
    k_hc_mixes<<<1, 256, 0, st>>>(pre, post, comb, x, fn, scale, base, hc, dim, iters, norm_eps, hc_eps);
}
void stbl_hc_pre(float *out, const float *x, const float *pre, int hc, int dim, cudaStream_t st) {
    k_hc_pre<<<GRID(dim, 256), 256, 0, st>>>(out, x, pre, hc, dim);
}
void stbl_hc_post(float *out, const float *x, const float *res, const float *post, const float *comb, int hc,
                  int dim, cudaStream_t st) {
    k_hc_post<<<GRID(hc * dim, 256), 256, 0, st>>>(out, x, res, post, comb, hc, dim);
}
void stbl_sparse_attn(float *out, const float *q, const float *kv1, int n1, const float *kv2, const int *idx,
                      int n_idx, const float *sink, int heads, int d, float scale, cudaStream_t st) {
    k_sparse_attn<<<heads, 256, (size_t)(d + n_idx) * sizeof(float), st>>>(out, q, kv1, n1, kv2, idx, n_idx, sink, d,
                                                                          scale);
}
void stbl_sqrt_softplus(float *s, int n, cudaStream_t st) { k_sqrt_softplus<<<GRID(n, 256), 256, 0, st>>>(s, n); }
void stbl_swiglu(float *out, const float *g, const float *u, float weight, float limit, int n, cudaStream_t st) {
    k_swiglu<<<GRID(n, 256), 256, 0, st>>>(out, g, u, weight, limit, n);
}
void stbl_axpy(float *y, const float *x, float a, int n, cudaStream_t st) {
    k_axpy<<<GRID(n, 256), 256, 0, st>>>(y, x, a, n);
}
void stbl_embed(float *h, const void *table, int token, int dim, int hc, cudaStream_t st) {
    k_embed<<<GRID(dim, 256), 256, 0, st>>>(h, (const __nv_bfloat16 *)table, token, dim, hc);
}

}  // extern "C"
