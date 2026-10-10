//! Mixer glue: the GDN in-projection's conv/norm/gates, attention q/k norm with rotary, and one-row attention.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "glue_math.cuh"

namespace tf_q27 {

using bf16 = __nv_bfloat16;

// One row's conv, SiLU and q/k/v prep (glue.py's _gdn_pre). 64 threads: 2 lanes a dk element pair at dk = 128.
__global__ void __launch_bounds__(64) gdn_pre(const bf16* __restrict__ qkv, const bf16* __restrict__ conv_state,
                                              const bf16* __restrict__ conv_w, const int* __restrict__ window,
                                              const bf16* __restrict__ a, const bf16* __restrict__ b,
                                              const float* __restrict__ a_log, const bf16* __restrict__ dt_bias,
                                              bf16* __restrict__ q, bf16* __restrict__ k, bf16* __restrict__ v,
                                              float* __restrict__ g, float* __restrict__ beta, int c, int kh, int vh,
                                              int dk, int nkeep) {
    const int row = blockIdx.x;
    const int head = blockIdx.y;               // 0..kh-1 q, kh..2kh-1 k, then vh v heads
    const int pair = threadIdx.x >> 1;         // 32 element pairs
    const int half = threadIdx.x & 1;
    const int ch = head * dk + pair * 2 + half;
    const int conv_dim = 2 * kh * dk + vh * dk;
    if (ch >= conv_dim) return;
    float acc = 0.f;
    const bool multi = window != nullptr;
#pragma unroll 4
    for (int j = 0; j < nkeep + 1; ++j) {
        const int src = multi ? window[row * (nkeep + 1) + j] : j - nkeep;
        const bool from_state = src < nkeep;
        const float x = from_state ? tf_glue::bf(conv_state[src * conv_dim + ch])
                                   : tf_glue::bf(qkv[(src - nkeep) * conv_dim + ch]);
        acc = __fmaf_rn(x, tf_glue::bf(conv_w[ch * (nkeep + 1) + j]), acc);
    }
    const float silu = tf_glue::rbf(acc / (1.f + tf_glue::tf_exp(-acc)));
    if (head < 2 * kh) {
        const bool is_q = head < kh;
        __shared__ float sq[64];
        sq[threadIdx.x] = (pair * 2 + half < dk) ? silu * silu : 0.f;
        __syncthreads();
        float ss = 0.f;
        for (int i = 0; i < 64; ++i) ss += sq[i];
        const float inv = rsqrtf(__fdividef(ss, (float)dk) + 1e-6f);
        const float scale = is_q ? __fdividef(1.f, (float)dk) : rsqrtf((float)dk);
        const float out = tf_glue::rbf(silu * inv * scale);
        const size_t at = ((size_t)row * kh + (is_q ? head : head - kh)) * dk + pair * 2 + half;
        if (is_q) q[at] = __float2bfloat16(out); else k[at] = __float2bfloat16(out);
    } else {
        const int hv = head - 2 * kh;
        v[((size_t)row * vh + hv) * dk + pair * 2 + half] = __float2bfloat16(silu);
        if (threadIdx.x == 0) {
            const float av = tf_glue::bf(a[(size_t)row * vh + hv]) + tf_glue::bf(dt_bias[hv]);
            const float sp = av > 20.f ? av : __logf(1.f + tf_glue::tf_exp(av));
            g[(size_t)row * vh + hv] = __expf(-__expf(a_log[hv]) * sp);
            beta[(size_t)row * vh + hv] = 1.f / (1.f + tf_glue::tf_exp(-tf_glue::bf(b[(size_t)row * vh + hv])));
        }
    }
}

// q/k RMSNorm then rotary (rotate-half pairs; mrope picks the axis's position when three coordinates are given).
__global__ void __launch_bounds__(32) attn_prep(const bf16* __restrict__ qg, const bf16* __restrict__ kv,
                                                const bf16* __restrict__ qn, const bf16* __restrict__ kn,
                                                const float* __restrict__ pos, const float* __restrict__ inv,
                                                bf16* __restrict__ qo, bf16* __restrict__ ko, float eps,
                                                int h, int hkv, int d, int half, int rows, int hsec, int wsec) {
    const int row = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
    const bool is_q = head < h;
    const int hd = is_q ? head : head - h;
    const size_t base = is_q ? ((size_t)row * h + hd) * 2 * d : (size_t)row * hkv * d + (size_t)hd * d;
    const bf16* __restrict__ src = is_q ? qg + base : kv + base;
    const bf16* __restrict__ norm = is_q ? qn : kn;
    bf16* __restrict__ dst = is_q ? qo + ((size_t)row * h + hd) * d : ko + ((size_t)row * hkv + hd) * d;
    float acc = 0.f;
    for (int i = lane; i < d; i += 32) acc = __fmaf_rn(tf_glue::bf(src[i]), tf_glue::bf(src[i]), acc);
#pragma unroll
    for (int m = 16; m; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    const float inv_norm = rsqrtf(__fdividef(acc, (float)d) + eps);
    __shared__ bf16 normed[256];
    for (int i = lane; i < d; i += 32) normed[i] = __float2bfloat16(tf_glue::rbf(tf_glue::bf(src[i]) * inv_norm * tf_glue::bf(norm[i])));
    __syncthreads();
    for (int i = lane; i < d; i += 32) {
        const int partner = i < half ? i + half : (i < 2 * half ? i - half : i);
        float out = tf_glue::bf(normed[i]);
        if (i < 2 * half) {
            const int freq_i = i % half;
            float p;
            if (rows > 0) {                     // mrope: three coordinates, the axis by interleaved section
                const int sec = freq_i % 3;
                int axis = 0;
                if (sec == 1 && freq_i < 3 * hsec) axis = 1;
                if (sec == 2 && freq_i < 3 * wsec) axis = 2;
                p = pos[axis * rows + row];
            } else {
                p = pos[row];
            }
            const float ang = p * inv[freq_i];
            const float cs = __cosf(ang), sn = __sinf(ang);
            const float pn = tf_glue::bf(normed[partner]);
            out = i < half ? out * cs - pn * sn : out * cs + pn * sn;
        }
        dst[i] = __float2bfloat16(tf_glue::rbf(out));
    }
}

// One query row's attention over n keys: softmax(Q K^T scale) V in fp32, key order fixed. A warp, a lane d/32 elements.
__global__ void __launch_bounds__(32) attn_row(const bf16* __restrict__ q, const bf16* __restrict__ keys,
                                               const bf16* __restrict__ values, bf16* __restrict__ out,
                                               int heads, int hkv, int d, long long n, float scale) {
    const long long row = blockIdx.x;
    const int head = blockIdx.y, lane = threadIdx.x;
    const int kv_head = head / (heads / hkv);
    const bf16* qr = q + ((size_t)row * heads + head) * d;
    float o[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float m = -INFINITY, l = 0.f;
    for (long long k0 = 0; k0 < n; k0 += 32) {
        const long long k = k0 + lane;
        float score = -INFINITY;
        float kv_dot = 0.f;
        if (k < n) {
            const bf16* kr = keys + (size_t)k * hkv * d + (size_t)kv_head * d;
            for (int i = 0; i < d; ++i) kv_dot = __fmaf_rn(tf_glue::bf(qr[i]), tf_glue::bf(kr[i]), kv_dot);
            score = kv_dot * scale;
        }
        float new_m = m;
        if (score > new_m) new_m = score;
#pragma unroll
        for (int mm = 16; mm; mm >>= 1) new_m = fmaxf(new_m, __shfl_xor_sync(0xffffffffu, new_m, mm));
        const float alpha = (m == -INFINITY) ? 0.f : tf_glue::tf_exp(m - new_m);
        const float p = score == -INFINITY ? 0.f : tf_glue::tf_exp(score - new_m);
        l = l * alpha;
#pragma unroll
        for (int e = 0; e < 8; ++e) o[e] *= alpha;
        if (k < n) {
            l += p;
            const bf16* vr = values + (size_t)k * hkv * d + (size_t)kv_head * d;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int i = lane + e * 32;
                if (i < d) o[e] = __fmaf_rn(p, tf_glue::bf(vr[i]), o[e]);
            }
        }
        m = new_m;
    }
    const float inv_l = 1.f / l;
    bf16* dst = out + ((size_t)row * heads + head) * d;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int i = lane + e * 32;
        if (i < d) dst[i] = __float2bfloat16(tf_glue::rbf(o[e] * inv_l));
    }
}

}  // namespace tf_q27
