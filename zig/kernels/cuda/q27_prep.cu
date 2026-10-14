// GDN prep and attention prep. dk 128 q/k is a 64-thread pair reduction. Positions are int32. A and B are fp16 or bf16.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "glue_math.cuh"
#include "q27_math.cuh"

using bf16 = __nv_bfloat16;

__device__ float ab_f32(const void* p, int i, int f16) {
    if (f16) return __half2float(reinterpret_cast<const __half*>(p)[i]);
    return tf_glue::bf(reinterpret_cast<const bf16*>(p)[i]);
}

// One channel of the depthwise conv, then SiLU rounded through bf16.
__device__ float gdn_conv(const bf16* qkv, const bf16* cs, const bf16* cw, const int* win, int row, int ch, int c, int cs_row, int nkeep) {
    float sum = 0.f;
    if (ch < c) {
        for (int j = 0; j < nkeep + 1; ++j) {
            const int src = win[row * (nkeep + 1) + j];
            const float x = src < nkeep ? tf_glue::bf(cs[((size_t)cs_row + src) * c + ch])
                                        : tf_glue::bf(qkv[(size_t)(src - nkeep) * c + ch]);
            const float w = tf_glue::bf(cw[(size_t)ch * (nkeep + 1) + j]);
            sum = __fmaf_rn(x, w, sum);
        }
    }
    return tf_glue::rbf(__fmul_rn(sum, tf_q27m::mufu_sigmoid(sum)));
}

// Conv then SiLU, RMS-scaled q and k. dk 128: tid owns 2*tid and 2*tid+1, a warp butterfly, then the two warp sums.
extern "C" __global__ void tf_q27_gdn_pre(const bf16* __restrict__ qkv, const bf16* __restrict__ cs, const bf16* __restrict__ cw,
                                          const int* __restrict__ win, const void* __restrict__ a, const void* __restrict__ b,
                                          const float* __restrict__ alog, const float* __restrict__ dtb, bf16* __restrict__ q,
                                          bf16* __restrict__ k, bf16* __restrict__ v, float* __restrict__ g,
                                          float* __restrict__ beta, const int* __restrict__ sid, int c, int kh, int vh, int dk,
                                          int nkeep, int multi, int ab_f16) {
    __shared__ float ws[2];
    const int row = blockIdx.x, head = blockIdx.y;
    const int cs_row = multi ? sid[row] * nkeep : 0;
    if (dk == 128 && head < 2 * kh) {
        const int tid = threadIdx.x;
        const int i0 = tid * 2;
        const int chb = head * dk;
        const float c0 = gdn_conv(qkv, cs, cw, win, row, chb + i0, c, cs_row, nkeep);
        const float c1 = gdn_conv(qkv, cs, cw, win, row, chb + i0 + 1, c, cs_row, nkeep);
        float part = __fmaf_rn(c1, c1, __fmul_rn(c0, c0));
        part = tf_q27m::bfly32(part);
        if ((tid & 31) == 0) ws[tid >> 5] = part;
        __syncthreads();
        const float inv = tf_q27m::mufu_inv_rms(__fadd_rn(ws[0], ws[1]), dk, 1e-6f);
        const bool is_q = head < kh;
        const float scale = is_q ? __fdiv_rn(1.f, static_cast<float>(dk)) : __fdiv_rn(1.f, __fsqrt_rn(static_cast<float>(dk)));
        const int hk = is_q ? head : head - kh;
        bf16* dst = (is_q ? q : k) + ((size_t)row * kh + hk) * dk;
        dst[i0] = __float2bfloat16_rn(tf_glue::rbf(__fmul_rn(__fmul_rn(c0, inv), scale)));
        dst[i0 + 1] = __float2bfloat16_rn(tf_glue::rbf(__fmul_rn(__fmul_rn(c1, inv), scale)));
        return;
    }
    if (threadIdx.x != 0 || dk > 128) return;
    float cv[128];
    const int ch0 = head * dk;
    if (head < 2 * kh) {
        float ss = 0.f;
        for (int i = 0; i < dk; ++i) {
            cv[i] = gdn_conv(qkv, cs, cw, win, row, ch0 + i, c, cs_row, nkeep);
            ss = __fmaf_rn(cv[i], cv[i], ss);
        }
        const float inv = tf_q27m::mufu_inv_rms(ss, dk, 1e-6f);
        const bool is_q = head < kh;
        const float scale = is_q ? __fdiv_rn(1.f, static_cast<float>(dk)) : __fdiv_rn(1.f, __fsqrt_rn(static_cast<float>(dk)));
        const int hk = is_q ? head : head - kh;
        bf16* dst = (is_q ? q : k) + ((size_t)row * kh + hk) * dk;
        for (int i = 0; i < dk; ++i) dst[i] = __float2bfloat16_rn(tf_glue::rbf(__fmul_rn(__fmul_rn(cv[i], inv), scale)));
    } else {
        const int hv = head - 2 * kh;
        bf16* dst = v + ((size_t)row * vh + hv) * dk;
        for (int i = 0; i < dk; ++i) dst[i] = __float2bfloat16_rn(gdn_conv(qkv, cs, cw, win, row, ch0 + i, c, cs_row, nkeep));
        const float av = __fadd_rn(ab_f32(a, row * vh + hv, ab_f16), dtb[hv]);
        const float sp = tf_q27m::mufu_softplus(av);
        const float e = tf_q27m::mufu_exp(alog[hv]);
        g[(size_t)row * vh + hv] = tf_q27m::mufu_exp(__fmul_rn(-e, sp));
        beta[(size_t)row * vh + hv] = tf_q27m::mufu_sigmoid(ab_f32(b, row * vh + hv, ab_f16));
    }
}

// RMS then rotate-half. pos is int32. mrope_rows > 0 reads three coordinates at axis * mrope_rows + row.
extern "C" __global__ void tf_q27_attn_prep(const bf16* __restrict__ qg, const bf16* __restrict__ kv, const bf16* __restrict__ qn,
                                            const bf16* __restrict__ kn, const int* __restrict__ pos, const float* __restrict__ invf,
                                            bf16* __restrict__ qo, bf16* __restrict__ ko, float eps, int h, int hkv, int d, int half,
                                            int mrope_rows, int hsec, int wsec) {
    if (threadIdx.x != 0 || d > 256) return;
    const int row = blockIdx.x, head = blockIdx.y;
    const bool is_q = head < h;
    const int hd = is_q ? head : head - h;
    const bf16* src = is_q ? qg + ((size_t)row * h + hd) * 2 * d : kv + ((size_t)row * hkv + hd) * d;
    const bf16* norm = is_q ? qn : kn;
    bf16* dst = is_q ? qo + ((size_t)row * h + hd) * d : ko + ((size_t)row * hkv + hd) * d;
    float ss = 0.f;
    for (int i = 0; i < d; ++i) {
        const float xv = tf_glue::bf(src[i]);
        ss = __fmaf_rn(xv, xv, ss);
    }
    const float inv = tf_q27m::mufu_inv_rms(ss, d, eps);
    float xn[256];
    for (int i = 0; i < d; ++i) xn[i] = tf_glue::rbf(__fmul_rn(__fmul_rn(tf_glue::bf(src[i]), inv), tf_glue::bf(norm[i])));
    for (int i = 0; i < d; ++i) {
        float out = xn[i];
        if (i < 2 * half) {
            const int freq = i < half ? i : i - half;
            int axis = 0;
            if (mrope_rows > 0) {
                const int sec = freq % 3;
                if (sec == 1 && freq < 3 * hsec) axis = 1;
                if (sec == 2 && freq < 3 * wsec) axis = 2;
            }
            const int pr = mrope_rows > 0 ? pos[axis * mrope_rows + row] : pos[row];
            const float ang = __fmul_rn(static_cast<float>(pr), invf[freq]);
            float sn, cs;
            tf_q27m::sincos(ang, sn, cs);
            const float pn = xn[i < half ? i + half : i - half];
            out = i < half ? __fsub_rn(__fmul_rn(xn[i], cs), __fmul_rn(pn, sn)) : __fadd_rn(__fmul_rn(xn[i], cs), __fmul_rn(pn, sn));
        }
        dst[i] = __float2bfloat16_rn(tf_glue::rbf(out));
    }
}
