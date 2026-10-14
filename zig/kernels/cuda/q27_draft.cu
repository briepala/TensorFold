// Drafter glue: dynamic conv, q/k prep from cos/sin, and fp32 block attention.

#include <cuda_bf16.h>

#include "glue_math.cuh"
#include "q27_math.cuh"

using bf16 = __nv_bfloat16;

// Two-tap conv. Row r mixes row r and r-1 inside its seg-row block. HAS_RES adds the residual after the bf16 store.
extern "C" __global__ void tf_q27_dconv(const bf16* __restrict__ x, const bf16* __restrict__ dyn, const bf16* __restrict__ base,
                                        const bf16* __restrict__ res, bf16* __restrict__ out, int seg, int d, int g, int gs,
                                        int branch, int has_res) {
    if (threadIdx.x != 0) return;
    const int row = blockIdx.x;
    for (int c = 0; c < d; ++c) {
        const float xv = tf_glue::bf(x[(size_t)row * d + c]);
        const float prev = (row % seg > 0) ? tf_glue::bf(x[(size_t)(row - 1) * d + c]) : 0.f;
        const int grp = c / gs;
        const int dyn0 = ((row * 2 + branch) * 2) * g + grp;
        const float d0 = tf_glue::bf(dyn[dyn0]);
        const float d1 = tf_glue::bf(dyn[dyn0 + g]);
        const float b0 = tf_glue::bf(base[(size_t)(branch * 2) * d + c]);
        const float b1 = tf_glue::bf(base[(size_t)(branch * 2 + 1) * d + c]);
        const float k0 = tf_glue::rbf(__fadd_rn(b0, d0));
        const float k1 = tf_glue::rbf(__fadd_rn(b1, d1));
        float y = tf_glue::rbf(__fadd_rn(__fmul_rn(xv, k0), __fmul_rn(prev, k1)));
        if (has_res) y = tf_glue::rbf(__fadd_rn(tf_glue::bf(res[(size_t)row * d + c]), y));
        out[(size_t)row * d + c] = __float2bfloat16_rn(y);
    }
}

// One rotate-half pair. inv_a and inv_b are the rsqrt of the lanes that own those elements.
__device__ void dprep_pair(const bf16* src, const bf16* norm, const float* cos, const float* sin, bf16* dst, int i, int half, int row,
                           float inv_a, float inv_b) {
    const float a = tf_glue::rbf(__fmul_rn(__fmul_rn(tf_glue::bf(src[i]), inv_a), tf_glue::bf(norm[i])));
    const float b = tf_glue::rbf(__fmul_rn(__fmul_rn(tf_glue::bf(src[half + i]), inv_b), tf_glue::bf(norm[half + i])));
    const float cs = cos[(size_t)row * half + i];
    const float sn = sin[(size_t)row * half + i];
    dst[i] = __float2bfloat16_rn(__fmaf_rn(-b, sn, __fmul_rn(a, cs)));
    dst[half + i] = __float2bfloat16_rn(__fmaf_rn(b, cs, __fmul_rn(a, sn)));
}

// Norm and rotate q/k from cos and sin. V is a copy. half 64: lane L owns one pair in each half and its own rsqrt.
extern "C" __global__ void tf_q27_dprep(const bf16* __restrict__ qkv, const bf16* __restrict__ qn, const bf16* __restrict__ kn,
                                        const float* __restrict__ cos, const float* __restrict__ sin, bf16* __restrict__ qo,
                                        bf16* __restrict__ ko, bf16* __restrict__ vo, float eps, int h, int hkv, int half, int L,
                                        int stride) {
    const int row = blockIdx.x, head = blockIdx.y;
    const int dim = half * 2;
    const bf16* src = qkv + (size_t)row * stride + (size_t)head * dim;
    if (half == 64 && head < h + hkv && threadIdx.x < 32) {
        const int lane = threadIdx.x;
        const int i = lane * 2;
        const float a0 = tf_glue::bf(src[i]), a1 = tf_glue::bf(src[i + 1]);
        const float b0 = tf_glue::bf(src[half + i]), b1 = tf_glue::bf(src[half + i + 1]);
        const float pa = tf_q27m::bfly32(__fmaf_rn(a0, a0, __fmul_rn(a1, a1)));
        const float pb = tf_q27m::bfly32(__fmaf_rn(b0, b0, __fmul_rn(b1, b1)));
        const float inv = tf_q27m::mufu_rsq(__fmaf_rn(__fadd_rn(pa, pb), __fdiv_rn(1.f, static_cast<float>(dim)), eps));
        const bf16* norm = head < h ? qn : kn;
        bf16* dst = head < h ? qo + ((size_t)head * L + row) * dim : ko + ((size_t)(head - h) * L + row) * dim;
        dprep_pair(src, norm, cos, sin, dst, i, half, row, inv, inv);
        dprep_pair(src, norm, cos, sin, dst, i + 1, half, row, inv, inv);
        return;
    }
    if (threadIdx.x != 0 || half > 64) return;
    if (head >= h + hkv) {
        bf16* dst = vo + ((size_t)(head - h - hkv) * L + row) * dim;
        for (int i = 0; i < dim; ++i) dst[i] = src[i];
        return;
    }
    float ss = 0.f;
    for (int i = 0; i < dim; ++i) {
        const float v = tf_glue::bf(src[i]);
        ss = __fmaf_rn(v, v, ss);
    }
    const float inv = tf_q27m::mufu_rsq(__fmaf_rn(ss, __fdiv_rn(1.f, static_cast<float>(dim)), eps));
    const bf16* norm = head < h ? qn : kn;
    bf16* dst = head < h ? qo + ((size_t)head * L + row) * dim : ko + ((size_t)(head - h) * L + row) * dim;
    for (int i = 0; i < half; ++i) dprep_pair(src, norm, cos, sin, dst, i, half, row, inv, inv);
}

__device__ float dot_d(const bf16* a, const bf16* b, int d) {
    float s = 0.f;
    for (int i = 0; i < d; ++i) s = __fmaf_rn(tf_glue::bf(a[i]), tf_glue::bf(b[i]), s);
    return s;
}

// One query row of one kv head: context keys in tiles of bn, then the block's own keys, fp32 dots, p rounded bf16.
extern "C" __global__ void tf_q27_block_attn(const bf16* __restrict__ Q, const bf16* __restrict__ KB, const bf16* __restrict__ VB,
                                             const unsigned long long* __restrict__ table, const int* __restrict__ lens,
                                             bf16* __restrict__ O, float scale, int window, int R, int G, int hkv, int L, int D,
                                             int bn, int causal) {
    if (threadIdx.x != 0 || L > 64 || bn > 64 || D > 128) return;
    const int j = blockIdx.x, g = blockIdx.y, r = blockIdx.z;
    const int slen = lens[j];
    const int qrow = j * L + r;
    const bf16* kc = reinterpret_cast<const bf16*>(table[2 * j]) + (size_t)g * slen * D;
    const bf16* vc = reinterpret_cast<const bf16*>(table[2 * j + 1]) + (size_t)g * slen * D;
    for (int h = 0; h < G; ++h) {
        const int head = g * G + h;
        const bf16* q = Q + ((size_t)head * R + qrow) * D;
        float m = tf_q27m::neg_inf(), l = 0.f, acc[128];
        for (int i = 0; i < D; ++i) acc[i] = 0.f;
        for (int n0 = 0; n0 < slen; n0 += bn) {
            float scores[64];
            float tile_m = m;
            for (int t = 0; t < bn; ++t) {
                const int kidx = n0 + t;
                const bool seen = kidx < slen && (slen + r - kidx < window + 1);
                float sc = tf_q27m::neg_inf();
                if (seen) sc = __fmul_rn(dot_d(q, kc + (size_t)kidx * D, D), scale);
                scores[t] = sc;
                if (sc > tile_m) tile_m = sc;
            }
            const float m_safe = tile_m == tf_q27m::neg_inf() ? 0.f : tile_m;
            const float alpha = tf_q27m::mufu_exp(m - m_safe);
            float sum_p = 0.f;
            for (int t = 0; t < bn; ++t) {
                const float p = scores[t] == tf_q27m::neg_inf() ? 0.f : tf_q27m::mufu_exp(scores[t] - m_safe);
                scores[t] = p;
                sum_p = __fadd_rn(sum_p, p);
            }
            l = __fadd_rn(__fmul_rn(l, alpha), sum_p);
            for (int i = 0; i < D; ++i) acc[i] = __fmul_rn(acc[i], alpha);
            for (int t = 0; t < bn; ++t) {
                const int kidx = n0 + t;
                if (kidx >= slen) continue;
                const float pb = tf_glue::rbf(scores[t]);
                const bf16* vv = vc + (size_t)kidx * D;
                for (int i = 0; i < D; ++i) acc[i] = __fmaf_rn(pb, tf_glue::bf(vv[i]), acc[i]);
            }
            m = tile_m;
        }
        float scores[64];
        float tile_m = m;
        for (int col = 0; col < L; ++col) {
            const bool seen = causal ? col <= r : true;
            float sc = tf_q27m::neg_inf();
            if (seen) {
                const int krow = j * L + col;
                sc = __fmul_rn(dot_d(q, KB + ((size_t)g * R + krow) * D, D), scale);
            }
            scores[col] = sc;
            if (sc > tile_m) tile_m = sc;
        }
        const float m_safe = tile_m == tf_q27m::neg_inf() ? 0.f : tile_m;
        const float alpha = tf_q27m::mufu_exp(m - m_safe);
        float sum_p = 0.f;
        for (int col = 0; col < L; ++col) {
            const float p = scores[col] == tf_q27m::neg_inf() ? 0.f : tf_q27m::mufu_exp(scores[col] - m_safe);
            scores[col] = p;
            sum_p = __fadd_rn(sum_p, p);
        }
        l = __fadd_rn(__fmul_rn(l, alpha), sum_p);
        for (int i = 0; i < D; ++i) acc[i] = __fmul_rn(acc[i], alpha);
        for (int col = 0; col < L; ++col) {
            const int krow = j * L + col;
            const float pb = tf_glue::rbf(scores[col]);
            const bf16* vv = VB + ((size_t)g * R + krow) * D;
            for (int i = 0; i < D; ++i) acc[i] = __fmaf_rn(pb, tf_glue::bf(vv[i]), acc[i]);
        }
        const float inv = tf_q27m::mufu_rcp(l);
        bf16* dst = O + (size_t)qrow * (G * hkv * D) + (size_t)head * D;
        for (int i = 0; i < D; ++i) dst[i] = __float2bfloat16_rn(tf_glue::rbf(__fmul_rn(acc[i], inv)));
    }
}
