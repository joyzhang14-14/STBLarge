// STBLarge DeepSeek-V4.1 decode engine: the official inference/model.py forward for one token at a
// time, on the lossless STBLarge GGUF. Routed experts stay on SSD and are read into a GPU LRU cache;
// Engram rows are read from the file on lookup; everything else is resident on the GPU.
// STBL_DUMP=FILE appends the residual stream (hc x dim fp32) after every layer of every token;
// STBL_DUMP_ROUTES=FILE appends the routed expert ids (int32, top-k order) of every layer.
#include <cuda_runtime.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <algorithm>
#include <list>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "gguf.h"
#include "kernels.h"
#include "stbl.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(e_)); } while (0)

namespace {

struct Cfg {
    int vocab, dim, inter, layers, heads, head_dim, rope, q_rank, o_rank, o_groups, experts, topk, window;
    int ih, idim, index_topk, cand_layer, cand_blocks, cand_block, hc, sinkhorn;
    float eps, hc_eps, swiglu_limit, route_scale;
    std::vector<int> ratio, kv_source, index_source;
};

struct Linear {  // FP8 weight with E8M0 32x32 block scales, BF16 or F32
    uint32_t type = 0;
    void *w = nullptr, *s = nullptr;
    int rows = 0, k = 0;
};

struct Layer {
    float *hc_fn[2], *hc_scale[2], *hc_base[2], *norm[2];  // [0] attention, [1] ffn
    float *sinks, *q_norm, *kv_norm;
    Linear wq_a, wq_b, wkv, wo_a, wo_b;
    Linear comp_wkv, comp_gate, idx_wk, idx_wq_b, idx_proj;
    float *comp_norm = nullptr, *idx_k_norm = nullptr;
    float *gate_w, *gate_b;
    Linear sh_w1, sh_w2, sh_w3;
    const uint8_t *exp_w[3];
    uint64_t exp_bytes[3];
    Linear eng_kv;
    float *eng_q = nullptr, *eng_k = nullptr;
    uint64_t eng_rows_off = 0;
    int engram = -1;
    bool kv_source = false, index_source = false;
    float *win_kv;                                          // [window][head_dim]
    float *comp_kv = nullptr, *idx_k = nullptr;             // kv sources: [ctx/ratio][head_dim], [ctx/ratio][idim]
    float *comp_state_kv = nullptr, *comp_state_sc = nullptr;  // ratio > 1: [ratio][head_dim]
};

struct ExpertCache {
    int slots = 0;
    uint64_t slot_bytes = 0;
    uint8_t *mem = nullptr, *stage = nullptr;
    std::unordered_map<uint64_t, std::pair<int, std::list<uint64_t>::iterator>> where;
    std::list<uint64_t> lru;
    uint64_t hits = 0, misses = 0;
};

}  // namespace

struct stbl_model {
    GgufFile f;
    Cfg c;
    std::vector<Layer> L;
    void *embed = nullptr, *head = nullptr;
    float *out_norm = nullptr;
    cudaStream_t st = nullptr;
    int ctx = 0, pos = 0;
    std::string err;
    // engram
    std::vector<int> eng_layers;
    std::vector<uint64_t> token_map, primes, mult;
    int eng_pad = 0, eng_heads = 0, eng_ngram = 0;
    std::vector<int64_t> history;
    uint8_t *eng_stage = nullptr;
    // rope tables [ctx][rope/2][cos, sin]
    float *rope_plain = nullptr, *rope_comp = nullptr;
    // activations
    float *h, *h2, *res, *xn, *sub, *xq, *pre_mix, *pre_attn, *pre_ffn, *post, *comb;
    float *t0, *qr, *q, *latent, *kbuf, *o, *low, *idx_q, *idx_w, *idx_score;
    float *router, *moe_xq, *moe_g, *moe_u, *moe_mid, *moe_out, *moe_y, *eng_vals, *eng_kv, *logits;
    int *attn_idx;
    ExpertCache ec;
    std::vector<int> topk_shared, last_route;
    std::vector<char> candidates;
    FILE *dump = nullptr, *dump_routes = nullptr;
};

namespace {

const GgufTensor &need(const GgufFile &f, const std::string &name) {
    const GgufTensor *t = f.tensor(name);
    if (!t) throw std::runtime_error("missing tensor " + name);
    return *t;
}

void *upload(const GgufFile &f, const GgufTensor &t) {
    void *d;
    CK(cudaMalloc(&d, t.nbytes));
    CK(cudaMemcpy(d, f.data(t), t.nbytes, cudaMemcpyHostToDevice));
    return d;
}

float *upload_f32(const GgufFile &f, const std::string &name) {
    const GgufTensor &t = need(f, name);
    if (t.type != QT_F32) throw std::runtime_error(name + " is not F32");
    return (float *)upload(f, t);
}

Linear load_linear(const GgufFile &f, const std::string &name) {
    const GgufTensor &t = need(f, name + ".weight");
    Linear l;
    l.type = t.type;
    l.k = (int)t.dims[0];
    l.rows = (int)t.dims[1];
    l.w = upload(f, t);
    if (t.type == QT_STBL_E4M3) l.s = upload(f, need(f, name + ".scale"));
    else if (t.type != QT_BF16 && t.type != QT_F32) throw std::runtime_error(name + ": unexpected type");
    return l;
}

float *dalloc(size_t n) {
    float *p;
    CK(cudaMalloc(&p, n * sizeof(float)));
    CK(cudaMemset(p, 0, n * sizeof(float)));
    return p;
}

// Official precompute_freqs_cis in fp32: base frequencies, YaRN ramp when orig > 0.
float *rope_table(int dim, int ctx, double orig, double base, double factor, double fast, double slow) {
    std::vector<float> freqs(dim / 2);
    for (int i = 0; i < dim / 2; i++) freqs[i] = 1.0f / powf((float)base, (float)(2 * i) / (float)dim);
    if (orig > 0) {
        auto corr = [&](double rot) { return dim * log(orig / (rot * 2 * M_PI)) / (2 * log(base)); };
        const int low = std::max((int)floor(corr(fast)), 0), high = std::min((int)ceil(corr(slow)), dim - 1);
        const float span = (float)std::max((double)(high - low), 1e-3);
        for (int i = 0; i < dim / 2; i++) {
            const float ramp = std::min(std::max(((float)i - (float)low) / span, 0.f), 1.f);
            const float smooth = 1 - ramp;
            freqs[i] = freqs[i] / (float)factor * (1 - smooth) + freqs[i] * smooth;
        }
    }
    std::vector<float> t((size_t)ctx * dim);
    for (int p = 0; p < ctx; p++)
        for (int i = 0; i < dim / 2; i++) {
            const float a = (float)p * freqs[i];
            t[((size_t)p * (dim / 2) + i) * 2] = cosf(a);
            t[((size_t)p * (dim / 2) + i) * 2 + 1] = sinf(a);
        }
    float *d = dalloc(t.size());
    CK(cudaMemcpy(d, t.data(), t.size() * 4, cudaMemcpyHostToDevice));
    return d;
}

// official linear(): FP8 weights quantize their input to FP8 per 32 (power-of-two scales) first
void gemv(stbl_model *m, const Linear &l, float *y, const float *x, bool round_out) {
    if (l.type == QT_STBL_E4M3) {
        CK(cudaMemcpyAsync(m->xq, x, (size_t)l.k * 4, cudaMemcpyDeviceToDevice, m->st));
        stbl_act_quant_fp8(m->xq, l.k, m->st);
        stbl_gemv_fp8(y, l.w, l.s, m->xq, l.rows, l.k, round_out, m->st);
    } else if (l.type == QT_BF16) {
        stbl_gemv_bf16(y, l.w, x, l.rows, l.k, round_out, 0, m->st);
    } else {
        stbl_gemv_f32(y, (const float *)l.w, x, l.rows, l.k, round_out, m->st);
    }
}

std::vector<float> to_host(stbl_model *m, const float *d, int n) {
    std::vector<float> v(n);
    CK(cudaMemcpyAsync(v.data(), d, (size_t)n * 4, cudaMemcpyDeviceToHost, m->st));
    CK(cudaStreamSynchronize(m->st));
    return v;
}

// ---- experts ------------------------------------------------------------------------------------

const uint8_t *expert(stbl_model *m, int layer, int e) {
    ExpertCache &ec = m->ec;
    const uint64_t key = (uint64_t)layer << 32 | (uint32_t)e;
    auto it = ec.where.find(key);
    if (it != ec.where.end()) {
        ec.lru.splice(ec.lru.begin(), ec.lru, it->second.second);
        ec.hits++;
        return ec.mem + (uint64_t)it->second.first * ec.slot_bytes;
    }
    ec.misses++;
    int slot;
    if ((int)ec.where.size() < ec.slots) {
        slot = (int)ec.where.size();
    } else {
        const uint64_t old = ec.lru.back();
        slot = ec.where[old].first;
        ec.where.erase(old);
        ec.lru.pop_back();
    }
    const Layer &l = m->L[layer];
    CK(cudaStreamSynchronize(m->st));  // the stage buffer may still feed an earlier copy
    uint64_t off = 0;
    for (int p = 0; p < 3; p++) {
        const uint64_t file_off = (uint64_t)(l.exp_w[p] - m->f.map) + (uint64_t)e * l.exp_bytes[p];
        if (pread(m->f.fd, ec.stage + off, l.exp_bytes[p], (off_t)file_off) != (ssize_t)l.exp_bytes[p])
            throw std::runtime_error("expert read failed");
        off += l.exp_bytes[p];
    }
    uint8_t *dst = ec.mem + (uint64_t)slot * ec.slot_bytes;
    CK(cudaMemcpyAsync(dst, ec.stage, ec.slot_bytes, cudaMemcpyHostToDevice, m->st));
    ec.lru.push_front(key);
    ec.where[key] = {slot, ec.lru.begin()};
    return dst;
}

// ---- MoE ------------------------------------------------------------------------------------------

void moe(stbl_model *m, int il, const float *x, float *out) {
    const Cfg &c = m->c;
    Layer &l = m->L[il];
    stbl_gemv_f32(m->router, l.gate_w, x, c.experts, c.dim, 0, m->st);
    stbl_sqrt_softplus(m->router, c.experts, m->st);
    const std::vector<float> sc = to_host(m, m->router, c.experts);
    const std::vector<float> bias = to_host(m, l.gate_b, c.experts);
    std::vector<int> order(c.experts);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + c.topk, order.end(), [&](int a, int b) {
        const float va = sc[a] + bias[a], vb = sc[b] + bias[b];
        return va > vb || (va == vb && a < b);
    });
    float sum = 0.f;
    for (int i = 0; i < c.topk; i++) sum += sc[order[i]];
    m->last_route.assign(order.begin(), order.begin() + c.topk);
    if (m->dump_routes)
        for (int i = 0; i < c.topk; i++) fwrite(&order[i], 4, 1, m->dump_routes);
    CK(cudaMemsetAsync(m->moe_y, 0, (size_t)c.dim * 4, m->st));
    CK(cudaMemcpyAsync(m->moe_xq, x, (size_t)c.dim * 4, cudaMemcpyDeviceToDevice, m->st));
    stbl_act_quant_fp8(m->moe_xq, c.dim, m->st);
    const uint64_t up = l.exp_bytes[0], down = l.exp_bytes[0] + l.exp_bytes[1];
    for (int i = 0; i < c.topk; i++) {
        const int e = order[i];
        float w = sc[e];
        w /= sum + 1e-20f;
        w *= c.route_scale;
        const uint8_t *ew = expert(m, il, e);
        stbl_gemv_mxfp4(m->moe_g, ew, m->moe_xq, c.inter, c.dim, m->st);
        stbl_gemv_mxfp4(m->moe_u, ew + up, m->moe_xq, c.inter, c.dim, m->st);
        stbl_swiglu(m->moe_mid, m->moe_g, m->moe_u, w, c.swiglu_limit, c.inter, m->st);
        stbl_act_quant_fp8(m->moe_mid, c.inter, m->st);
        stbl_gemv_mxfp4(m->moe_out, ew + down, m->moe_mid, c.dim, c.inter, m->st);
        stbl_axpy(m->moe_y, m->moe_out, 1.f, c.dim, m->st);
    }
    gemv(m, l.sh_w1, m->moe_g, x, true);
    gemv(m, l.sh_w3, m->moe_u, x, true);
    stbl_swiglu(m->moe_mid, m->moe_g, m->moe_u, 1.f, c.swiglu_limit, c.inter, m->st);
    gemv(m, l.sh_w2, m->moe_out, m->moe_mid, true);
    stbl_axpy(m->moe_y, m->moe_out, 1.f, c.dim, m->st);
    CK(cudaMemcpyAsync(out, m->moe_y, (size_t)c.dim * 4, cudaMemcpyDeviceToDevice, m->st));
    stbl_round_bf16(out, c.dim, m->st);
}

// ---- attention ------------------------------------------------------------------------------------

int owner_of(const Cfg &c, int il) {
    int o = -1;
    for (int s : c.kv_source) if (s <= il) o = s;
    return o;
}

// official select_candidate_blocks for one decode query: blocks scored by their best position, the
// newest (partial) block pinned, top `blocks` blocks kept.
std::vector<char> candidate_mask(const std::vector<float> &s, int n, int blocks, int block) {
    const int nb = (n + block - 1) / block;
    std::vector<float> bs(nb, -INFINITY);
    for (int i = 0; i < n; i++) bs[i / block] = std::max(bs[i / block], s[i]);
    bs[(n - 1) / block] = INFINITY;
    std::vector<int> o(nb);
    std::iota(o.begin(), o.end(), 0);
    const int k = std::min(blocks, nb);
    std::partial_sort(o.begin(), o.begin() + k, o.end(), [&](int a, int b) { return bs[a] > bs[b] || (bs[a] == bs[b] && a < b); });
    std::vector<char> keep(n, 0);
    for (int j = 0; j < k; j++)
        if (bs[o[j]] > -INFINITY)
            for (int i = o[j] * block; i < std::min(n, (o[j] + 1) * block); i++) keep[i] = 1;
    return keep;
}

// Indexer of an index source layer: scores the compressed positions and keeps the top index_topk.
void indexer(stbl_model *m, int il, const float *x, int comp_len, const float *rope_cs) {
    const Cfg &c = m->c;
    Layer &l = m->L[il];
    const Layer &owner = m->L[owner_of(c, il)];
    gemv(m, l.idx_wq_b, m->idx_q, m->qr, true);
    stbl_rope(m->idx_q, rope_cs, c.ih, c.idim, c.rope, 0, m->st);
    stbl_act_quant_fp4(m->idx_q, c.ih * c.idim, 32, 0, m->st);
    gemv(m, l.idx_proj, m->idx_w, x, true);
    stbl_scale_bf16(m->idx_w, (float)(1.0 / sqrt((double)c.idim) * (1.0 / sqrt((double)c.ih))), c.ih, m->st);
    stbl_index_score(m->idx_score, m->idx_q, owner.idx_k, m->idx_w, comp_len, c.ih, c.idim, m->st);
    std::vector<float> s = to_host(m, m->idx_score, comp_len);
    if (il == c.cand_layer) {
        m->candidates = candidate_mask(s, comp_len, c.cand_blocks, c.cand_block);
    } else if (c.cand_layer >= 0 && il > c.cand_layer) {
        for (int i = 0; i < comp_len; i++) if (!m->candidates[i]) s[i] = -INFINITY;
    }
    std::vector<int> o(comp_len);
    std::iota(o.begin(), o.end(), 0);
    const int k = std::min(c.index_topk, comp_len);
    std::partial_sort(o.begin(), o.begin() + k, o.end(), [&](int a, int b) { return s[a] > s[b] || (s[a] == s[b] && a < b); });
    o.resize(k);
    std::sort(o.begin(), o.end());
    m->topk_shared.clear();
    for (int p : o) m->topk_shared.push_back(c.window + p);  // decode: every pick is < comp_len, so valid
}

void attention(stbl_model *m, int il, const float *x, float *out) {
    const Cfg &c = m->c;
    Layer &l = m->L[il];
    const int pos = m->pos, ratio = c.ratio[il];
    const float *cs = (ratio ? m->rope_comp : m->rope_plain) + (size_t)pos * c.rope;
    gemv(m, l.wq_a, m->t0, x, true);
    stbl_rms_norm(m->qr, m->t0, l.q_norm, 1, c.q_rank, c.eps, m->st);
    gemv(m, l.wq_b, m->q, m->qr, true);
    stbl_rope(m->q, cs, c.heads, c.head_dim, c.rope, 0, m->st);
    gemv(m, l.wkv, m->t0, x, true);
    float *kv = l.win_kv + (size_t)(pos % c.window) * c.head_dim;
    stbl_rms_norm(kv, m->t0, l.kv_norm, 1, c.head_dim, c.eps, m->st);
    stbl_rope(kv, cs, 1, c.head_dim, c.rope, 0, m->st);
    stbl_act_quant_fp8(kv, c.head_dim, m->st);
    std::vector<int> idx;
    for (int s = 0; s < c.window && s <= pos; s++) idx.push_back(s);
    const float *comp_kv = nullptr;
    if (ratio) {
        const Layer &owner = m->L[owner_of(c, il)];
        comp_kv = owner.comp_kv;
        const int comp_len = (pos + 1) / ratio;
        bool have_latent = false;
        if (l.kv_source) {  // Compressor
            if (ratio == 1) {
                gemv(m, l.comp_wkv, m->t0, x, true);
                stbl_rms_norm(m->latent, m->t0, l.comp_norm, 1, c.head_dim, c.eps, m->st);
                have_latent = true;
            } else {
                const int slot = pos % ratio;
                gemv(m, l.comp_wkv, l.comp_state_kv + (size_t)slot * c.head_dim, x, false);
                gemv(m, l.comp_gate, l.comp_state_sc + (size_t)slot * c.head_dim, x, false);
                if ((pos + 1) % ratio == 0) {
                    stbl_compress_pool(m->t0, l.comp_state_kv, l.comp_state_sc, ratio, c.head_dim, m->st);
                    stbl_rms_norm(m->latent, m->t0, l.comp_norm, 1, c.head_dim, c.eps, m->st);
                    have_latent = true;
                }
            }
        }
        const float *lat_cs = m->rope_comp + (size_t)(pos + 1 - ratio) * c.rope;
        if (l.index_source) {
            if (have_latent) {  // index keys from the latent before its RoPE
                gemv(m, l.idx_wk, m->t0, m->latent, true);
                stbl_rms_norm(m->kbuf, m->t0, l.idx_k_norm, 1, c.idim, c.eps, m->st);
                stbl_rope(m->kbuf, lat_cs, 1, c.idim, c.rope, 0, m->st);
                stbl_act_quant_fp4(m->kbuf, c.idim, 32, 0, m->st);
                CK(cudaMemcpyAsync(l.idx_k + (size_t)(pos / ratio) * c.idim, m->kbuf, (size_t)c.idim * 4,
                                   cudaMemcpyDeviceToDevice, m->st));
            }
            if (comp_len == 0) m->topk_shared.clear();
            else indexer(m, il, x, comp_len, cs);
        }
        if (have_latent) {
            stbl_rope(m->latent, lat_cs, 1, c.head_dim, c.rope, 0, m->st);
            stbl_act_quant_fp4(m->latent, c.head_dim, 16, 1, m->st);
            CK(cudaMemcpyAsync(l.comp_kv + (size_t)(pos / ratio) * c.head_dim, m->latent, (size_t)c.head_dim * 4,
                               cudaMemcpyDeviceToDevice, m->st));
        }
        for (int p : m->topk_shared) idx.push_back(p);
    }
    CK(cudaMemcpyAsync(m->attn_idx, idx.data(), idx.size() * 4, cudaMemcpyHostToDevice, m->st));
    stbl_sparse_attn(m->o, m->q, l.win_kv, c.window, comp_kv ? comp_kv : l.win_kv, m->attn_idx, (int)idx.size(),
                     l.sinks, c.heads, c.head_dim, (float)(1.0 / sqrt((double)c.head_dim)), m->st);
    CK(cudaStreamSynchronize(m->st));  // idx is a host vector
    stbl_rope(m->o, cs, c.heads, c.head_dim, c.rope, 1, m->st);
    stbl_act_quant_fp8(m->o, c.heads * c.head_dim, m->st);
    stbl_gemv_bf16(m->low, l.wo_a.w, m->o, c.o_groups * c.o_rank, c.heads * c.head_dim / c.o_groups, 1, c.o_rank, m->st);
    gemv(m, l.wo_b, out, m->low, true);
}

// ---- Engram ----------------------------------------------------------------------------------------

void engram(stbl_model *m, int il) {
    const Cfg &c = m->c;
    Layer &l = m->L[il];
    const int li = l.engram, n = (int)m->history.size() - 1;  // n = current position
    const int per_layer = (m->eng_ngram - 1) * m->eng_heads;
    std::vector<int64_t> rows(per_layer);
    uint64_t rolling = 0;
    for (int shift = 0; shift < m->eng_ngram; shift++) {
        const int64_t tok = n < shift ? m->eng_pad : m->history[n - shift];
        const uint64_t prod = (uint64_t)tok * m->mult[(size_t)li * m->eng_ngram + shift];
        rolling = shift == 0 ? prod : rolling ^ prod;
        if (shift == 0) continue;
        for (int h = 0; h < m->eng_heads; h++) {
            const int j = (shift - 1) * m->eng_heads + h;
            const uint64_t p = m->primes[(size_t)li * per_layer + j];
            uint64_t off = 0;
            for (int q = 0; q < j; q++) off += m->primes[(size_t)li * per_layer + q];
            rows[j] = (int64_t)(rolling % p + off);
        }
    }
    for (int j = 0; j < per_layer; j++)
        if (pread(m->f.fd, m->eng_stage + (size_t)j * 264, 264, (off_t)(l.eng_rows_off + (uint64_t)rows[j] * 264)) != 264)
            throw std::runtime_error("engram row read failed");
    CK(cudaMemcpyAsync(m->sub, m->eng_stage, (size_t)per_layer * 264, cudaMemcpyHostToDevice, m->st));
    stbl_engram_rows(m->eng_vals, m->sub, per_layer, m->st);
    gemv(m, l.eng_kv, m->eng_kv, m->eng_vals, true);
    stbl_engram_gate(m->h, m->eng_kv, l.eng_q, l.eng_k, c.dim, c.hc, c.eps, m->st);
    CK(cudaStreamSynchronize(m->st));  // eng_stage is reused next time
}

// ---- block / forward ---------------------------------------------------------------------------------

void block(stbl_model *m, int il) {
    const Cfg &c = m->c;
    Layer &l = m->L[il];
    const int n = c.hc * c.dim;
    CK(cudaMemcpyAsync(m->res, m->h, (size_t)n * 4, cudaMemcpyDeviceToDevice, m->st));
    stbl_hc_mixes(m->pre_attn, m->post, m->comb, m->h, l.hc_fn[0], l.hc_scale[0], l.hc_base[0], c.hc, c.dim,
                  c.sinkhorn, c.eps, c.hc_eps, m->st);
    stbl_hc_pre(m->t0, m->h, m->pre_mix, c.hc, c.dim, m->st);
    stbl_rms_norm(m->xn, m->t0, l.norm[0], 1, c.dim, c.eps, m->st);
    attention(m, il, m->xn, m->sub);
    stbl_hc_post(m->h, m->sub, m->res, m->post, m->comb, c.hc, c.dim, m->st);
    CK(cudaMemcpyAsync(m->res, m->h, (size_t)n * 4, cudaMemcpyDeviceToDevice, m->st));
    stbl_hc_mixes(m->pre_ffn, m->post, m->comb, m->h, l.hc_fn[1], l.hc_scale[1], l.hc_base[1], c.hc, c.dim,
                  c.sinkhorn, c.eps, c.hc_eps, m->st);
    stbl_hc_pre(m->t0, m->h, m->pre_attn, c.hc, c.dim, m->st);
    stbl_rms_norm(m->xn, m->t0, l.norm[1], 1, c.dim, c.eps, m->st);
    moe(m, il, m->xn, m->sub);
    stbl_hc_post(m->h, m->sub, m->res, m->post, m->comb, c.hc, c.dim, m->st);
    CK(cudaMemcpyAsync(m->pre_mix, m->pre_ffn, (size_t)c.hc * 4, cudaMemcpyDeviceToDevice, m->st));
}

int forward(stbl_model *m, int token, float *logits_out) {
    const Cfg &c = m->c;
    if (m->pos >= m->ctx) throw std::runtime_error("context full");
    if (token < 0 || token >= c.vocab) throw std::runtime_error("token out of range");
    m->history.push_back((int64_t)m->token_map[token]);
    stbl_embed(m->h, m->embed, token, c.dim, c.hc, m->st);
    const float one_hot[4] = {1.f, 0.f, 0.f, 0.f};
    CK(cudaMemcpyAsync(m->pre_mix, one_hot, sizeof(one_hot), cudaMemcpyHostToDevice, m->st));
    for (int il = 0; il < c.layers; il++) {
        if (m->L[il].engram >= 0) engram(m, il);
        block(m, il);
        if (m->dump) {
            std::vector<float> h = to_host(m, m->h, c.hc * c.dim);
            fwrite(h.data(), 4, h.size(), m->dump);
        }
    }
    stbl_hc_pre(m->t0, m->h, m->pre_mix, c.hc, c.dim, m->st);
    stbl_rms_norm(m->xn, m->t0, m->out_norm, 1, c.dim, c.eps, m->st);
    stbl_gemv_bf16(m->logits, m->head, m->xn, c.vocab, c.dim, 0, 0, m->st);
    std::vector<float> lg = to_host(m, m->logits, c.vocab);
    m->pos++;
    if (logits_out) memcpy(logits_out, lg.data(), lg.size() * 4);
    return (int)(std::max_element(lg.begin(), lg.end()) - lg.begin());
}

void load(stbl_model *m, const char *path, int ctx, int slots) {
    std::string e;
    if (!m->f.open(path, e)) throw std::runtime_error(e);
    GgufFile &f = m->f;
    Cfg &c = m->c;
    const std::string P = "deepseek41.";
    c.vocab = f.u32(P + "vocab_size");
    c.dim = f.u32(P + "hidden_size");
    c.inter = f.u32(P + "moe_intermediate_size");
    c.layers = f.u32(P + "num_hidden_layers");
    c.heads = f.u32(P + "num_attention_heads");
    c.head_dim = f.u32(P + "head_dim");
    c.rope = f.u32(P + "qk_rope_head_dim");
    c.q_rank = f.u32(P + "q_lora_rank");
    c.o_rank = f.u32(P + "o_lora_rank");
    c.o_groups = f.u32(P + "o_groups");
    c.experts = f.u32(P + "n_routed_experts");
    c.topk = f.u32(P + "num_experts_per_tok");
    c.window = f.u32(P + "sliding_window");
    c.ih = f.u32(P + "index_n_heads");
    c.idim = f.u32(P + "index_head_dim");
    c.index_topk = f.u32(P + "index_topk");
    c.cand_layer = (int)f.u32(P + "candidate_source_layer_id");
    c.cand_blocks = f.u32(P + "candidate_topk_blocks");
    c.cand_block = f.u32(P + "candidate_block_size");
    c.hc = f.u32(P + "hc_mult");
    c.sinkhorn = f.u32(P + "hc_sinkhorn_iters");
    c.eps = f.f32(P + "rms_norm_eps");
    c.hc_eps = f.f32(P + "hc_eps");
    c.swiglu_limit = f.f32(P + "swiglu_limit");
    c.route_scale = f.f32(P + "routed_scaling_factor");
    for (uint64_t v : f.uints(P + "compress_ratios")) c.ratio.push_back((int)v);
    for (uint64_t v : f.uints(P + "kv_source_layer_ids")) c.kv_source.push_back((int)v);
    for (uint64_t v : f.uints(P + "index_source_layer_ids")) c.index_source.push_back((int)v);
    if (c.hc != 4) throw std::runtime_error("hc_mult must be 4");
    m->ctx = ctx;
    CK(cudaStreamCreateWithFlags(&m->st, cudaStreamNonBlocking));

    m->rope_plain = rope_table(c.rope, ctx, 0, f.u32(P + "rope_theta"), 1, 0, 0);
    m->rope_comp = rope_table(c.rope, ctx, f.f32(P + "rope_scaling.original_max_position_embeddings"),
                              f.u32(P + "compress_rope_theta"), f.f32(P + "rope_scaling.factor"),
                              f.f32(P + "rope_scaling.beta_fast"), f.f32(P + "rope_scaling.beta_slow"));

    for (uint64_t v : f.uints(P + "engram.layer_ids")) m->eng_layers.push_back((int)v);
    m->token_map = f.uints(P + "engram.token_map");
    m->primes = f.uints(P + "engram.primes");
    m->mult = f.uints(P + "engram.multipliers");
    m->eng_pad = f.u32(P + "engram.pad_id");
    m->eng_ngram = (int)(m->mult.size() / m->eng_layers.size());
    m->eng_heads = (int)(m->primes.size() / m->eng_layers.size() / (m->eng_ngram - 1));

    m->embed = upload(f, need(f, "token_embd.weight"));
    m->head = upload(f, need(f, "output.weight"));
    m->out_norm = upload_f32(f, "output_norm.weight");
    m->L.resize(c.layers);
    uint64_t expert_bytes = 0;
    for (int il = 0; il < c.layers; il++) {
        Layer &l = m->L[il];
        const std::string b = "blk." + std::to_string(il) + ".";
        const char *site[2] = {"attn", "ffn"};
        for (int s = 0; s < 2; s++) {
            l.hc_fn[s] = upload_f32(f, b + "hc_" + site[s] + "_fn.weight");
            l.hc_scale[s] = upload_f32(f, b + "hc_" + site[s] + "_scale.weight");
            l.hc_base[s] = upload_f32(f, b + "hc_" + site[s] + "_base.weight");
            l.norm[s] = upload_f32(f, b + site[s] + "_norm.weight");
        }
        l.sinks = upload_f32(f, b + "attn_sinks.weight");
        l.q_norm = upload_f32(f, b + "attn_q_a_norm.weight");
        l.kv_norm = upload_f32(f, b + "attn_kv_a_norm.weight");
        l.wq_a = load_linear(f, b + "attn_q_a");
        l.wq_b = load_linear(f, b + "attn_q_b");
        l.wkv = load_linear(f, b + "attn_kv");
        l.wo_a = load_linear(f, b + "attn_output_a");
        l.wo_b = load_linear(f, b + "attn_output_b");
        l.kv_source = std::find(c.kv_source.begin(), c.kv_source.end(), il) != c.kv_source.end();
        l.index_source = std::find(c.index_source.begin(), c.index_source.end(), il) != c.index_source.end();
        const int ratio = c.ratio[il];
        if (l.kv_source) {
            l.comp_wkv = load_linear(f, b + "attn_compressor_kv");
            l.comp_norm = upload_f32(f, b + "attn_compressor_norm.weight");
            l.idx_wk = load_linear(f, b + "indexer.attn_k");
            l.idx_k_norm = upload_f32(f, b + "indexer.k_norm.weight");
            if (ratio > 1) {
                l.comp_gate = load_linear(f, b + "attn_compressor_gate");
                l.comp_state_kv = dalloc((size_t)ratio * c.head_dim);
                l.comp_state_sc = dalloc((size_t)ratio * c.head_dim);
                std::vector<float> ninf((size_t)ratio * c.head_dim, -INFINITY);
                CK(cudaMemcpy(l.comp_state_sc, ninf.data(), ninf.size() * 4, cudaMemcpyHostToDevice));
            }
            l.comp_kv = dalloc((size_t)(ctx / ratio + 1) * c.head_dim);
            l.idx_k = dalloc((size_t)(ctx / ratio + 1) * c.idim);
        }
        if (l.index_source) {
            l.idx_wq_b = load_linear(f, b + "indexer.attn_q_b");
            l.idx_proj = load_linear(f, b + "indexer.proj");
        }
        l.gate_w = upload_f32(f, b + "ffn_gate_inp.weight");
        l.gate_b = upload_f32(f, b + "exp_probs_b.bias");
        l.sh_w1 = load_linear(f, b + "ffn_gate_shexp");
        l.sh_w3 = load_linear(f, b + "ffn_up_shexp");
        l.sh_w2 = load_linear(f, b + "ffn_down_shexp");
        const char *part[3] = {"gate", "up", "down"};
        uint64_t bytes = 0;
        for (int p = 0; p < 3; p++) {
            const GgufTensor &t = need(f, b + "ffn_" + part[p] + "_exps.weight");
            if (t.type != QT_MXFP4 || (int)t.dims[2] != c.experts) throw std::runtime_error("unexpected expert tensor");
            l.exp_w[p] = f.data(t);
            l.exp_bytes[p] = t.nbytes / c.experts;
            bytes += l.exp_bytes[p];
        }
        expert_bytes = std::max(expert_bytes, bytes);
        auto eit = std::find(m->eng_layers.begin(), m->eng_layers.end(), il);
        if (eit != m->eng_layers.end()) {
            l.engram = (int)(eit - m->eng_layers.begin());
            l.eng_kv = load_linear(f, b + "engram_kv");
            l.eng_q = upload_f32(f, b + "engram_q_norm.weight");
            l.eng_k = upload_f32(f, b + "engram_k_norm.weight");
            l.eng_rows_off = need(f, b + "engram_embd.weight").offset;
        }
        l.win_kv = dalloc((size_t)c.window * c.head_dim);
    }
    m->ec.slots = slots;
    m->ec.slot_bytes = expert_bytes;
    CK(cudaMalloc(&m->ec.mem, (size_t)slots * expert_bytes));
    CK(cudaMallocHost(&m->ec.stage, expert_bytes));
    CK(cudaMallocHost(&m->eng_stage, 64 * 264));

    const size_t n = (size_t)c.hc * c.dim, qd = (size_t)c.heads * c.head_dim;
    m->h = dalloc(n); m->h2 = dalloc(n); m->res = dalloc(n);
    m->xn = dalloc(c.dim); m->sub = dalloc(std::max<size_t>(c.dim, 64 * 264 / 4 + 1));
    m->xq = dalloc(std::max<size_t>(qd, 32768));
    m->pre_mix = dalloc(c.hc); m->pre_attn = dalloc(c.hc); m->pre_ffn = dalloc(c.hc);
    m->post = dalloc(c.hc); m->comb = dalloc(c.hc * c.hc);
    m->t0 = dalloc(std::max<size_t>(c.dim, qd)); m->qr = dalloc(c.q_rank); m->q = dalloc(qd);
    m->latent = dalloc(c.head_dim); m->kbuf = dalloc(c.idim); m->o = dalloc(qd);
    m->low = dalloc((size_t)c.o_groups * c.o_rank);
    m->idx_q = dalloc((size_t)c.ih * c.idim); m->idx_w = dalloc(c.ih); m->idx_score = dalloc(ctx);
    m->router = dalloc(c.experts); m->moe_xq = dalloc(c.dim); m->moe_g = dalloc(c.inter); m->moe_u = dalloc(c.inter);
    m->moe_mid = dalloc(c.inter); m->moe_out = dalloc(c.dim); m->moe_y = dalloc(c.dim);
    m->eng_vals = dalloc(64 * 256); m->eng_kv = dalloc((size_t)(c.hc + 1) * c.dim);
    m->logits = dalloc(c.vocab);
    CK(cudaMalloc(&m->attn_idx, (size_t)(c.window + c.index_topk + 8) * sizeof(int)));
    const char *dump = getenv("STBL_DUMP");
    if (dump && dump[0] && !(m->dump = fopen(dump, "wb"))) throw std::runtime_error("cannot open STBL_DUMP");
    const char *routes = getenv("STBL_DUMP_ROUTES");
    if (routes && routes[0] && !(m->dump_routes = fopen(routes, "wb"))) throw std::runtime_error("cannot open STBL_DUMP_ROUTES");
}

}  // namespace

extern "C" {

stbl_model *stbl_open(const char *path, int ctx, int expert_slots, char *err, int errlen) {
    stbl_model *m = new stbl_model();
    try {
        load(m, path, ctx, expert_slots);
        return m;
    } catch (const std::exception &e) {
        if (err && errlen > 0) snprintf(err, errlen, "%s", e.what());
        delete m;
        return nullptr;
    }
}

int stbl_step(stbl_model *m, int token, float *logits) {
    try {
        return forward(m, token, logits);
    } catch (const std::exception &e) {
        m->err = e.what();
        return -1;
    }
}

int stbl_block_test(stbl_model *m, int il, int pos, int token, const float *h_in, const float *pre_in,
                    float *h_out, float *pre_out, int stage) {
    try {
        const Cfg &c = m->c;
        const size_t n = (size_t)c.hc * c.dim;
        m->pos = pos;
        if (pos == 0) m->history.clear();
        if ((int)m->history.size() == pos) m->history.push_back((int64_t)m->token_map[token]);
        if ((int)m->history.size() != pos + 1) throw std::runtime_error("positions must be fed in order");
        // copies go on m->st: a plain cudaMemcpy runs on the legacy stream, which the non-blocking
        // m->st does not wait for, and may return before a pageable source reaches the device
        if (stage >= 3) {
            CK(cudaMemcpyAsync(m->xn, h_in, (size_t)c.dim * 4, cudaMemcpyHostToDevice, m->st));
            if (stage == 3) attention(m, il, m->xn, m->sub);
            else moe(m, il, m->xn, m->sub);
            CK(cudaStreamSynchronize(m->st));
            CK(cudaMemcpy(h_out, m->sub, (size_t)c.dim * 4, cudaMemcpyDeviceToHost));
            return 0;
        }
        CK(cudaMemcpyAsync(m->h, h_in, n * 4, cudaMemcpyHostToDevice, m->st));
        CK(cudaMemcpyAsync(m->pre_mix, pre_in, (size_t)c.hc * 4, cudaMemcpyHostToDevice, m->st));
        if (m->L[il].engram >= 0 && stage != 2) engram(m, il);
        if (stage != 1) block(m, il);
        CK(cudaStreamSynchronize(m->st));
        CK(cudaMemcpy(h_out, m->h, n * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(pre_out, m->pre_mix, (size_t)c.hc * 4, cudaMemcpyDeviceToHost));
        return 0;
    } catch (const std::exception &e) {
        m->err = e.what();
        return -1;
    }
}

int stbl_last_route(const stbl_model *m, int *ids) {
    std::copy(m->last_route.begin(), m->last_route.end(), ids);
    return (int)m->last_route.size();
}

int stbl_vocab(const stbl_model *m) { return m->c.vocab; }
const char *stbl_error(const stbl_model *m) { return m->err.c_str(); }

void stbl_stats(const stbl_model *m, uint64_t *hits, uint64_t *misses) {
    *hits = m->ec.hits;
    *misses = m->ec.misses;
}

void stbl_close(stbl_model *m) {
    if (m && m->dump) fclose(m->dump);
    if (m && m->dump_routes) fclose(m->dump_routes);
    delete m;  // device memory is released with the process
}

}  // extern "C"
