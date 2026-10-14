// Shared sine and cosine for q27 attention prep: one Cody-Waite reduction and one Taylor, on host and device.
#pragma once

#include "glue_math.cuh"

namespace tf_q27m {

__device__ __forceinline__ void sincos(float x, float& sn, float& cs) {
    const float j = __fsub_rn(__fmaf_rn(x, 0x1.45f306p-1f, 0x1.8p23f), 0x1.8p23f);
    float f = __fmaf_rn(j, -0x1.921fb6p0f, x);
    f = __fmaf_rn(j, -0x1.777a5cp-25f, f);
    const float f2 = __fmul_rn(f, f);
    float sp = -0x1.a01a02p-13f;
    sp = __fmaf_rn(sp, f2, 0x1.111112p-7f);
    sp = __fmaf_rn(sp, f2, -0x1.555556p-3f);
    sp = __fmaf_rn(sp, f2, 1.0f);
    const float s = __fmul_rn(f, sp);
    float cp = 0x1.a01a02p-16f;
    cp = __fmaf_rn(cp, f2, -0x1.6c16c2p-10f);
    cp = __fmaf_rn(cp, f2, 0x1.555556p-5f);
    cp = __fmaf_rn(cp, f2, -0.5f);
    const float c = __fmaf_rn(cp, f2, 1.0f);
    const int q = static_cast<int>(j) & 3;
    if (q == 0) {
        sn = s;
        cs = c;
    } else if (q == 1) {
        sn = c;
        cs = -s;
    } else if (q == 2) {
        sn = -s;
        cs = -c;
    } else {
        sn = -c;
        cs = s;
    }
}

__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }

__device__ __forceinline__ float mufu_ex2(float a) {
    float r;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}

__device__ __forceinline__ float mufu_rcp(float a) {
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}

__device__ __forceinline__ float mufu_sqrt(float a) {
    float r;
    asm("sqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}

__device__ __forceinline__ float mufu_rsq(float a) {
    float r;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}

// exp2f as the captured cubins emit it: raw EX2, and one half-then-square when the argument is below -126.
__device__ __forceinline__ float mufu_exp2(float a) {
    float t = a;
    if (a < -126.f) t = __fmul_rn(t, 0.5f);
    float r = mufu_ex2(t);
    if (a < -126.f) r = __fmul_rn(r, r);
    return r;
}

// tl.exp: exp2 of x times the f32 nearest log2(e).
__device__ __forceinline__ float mufu_exp(float x) {
    return mufu_exp2(__fmul_rn(x, __uint_as_float(0x3fb8aa3bu)));
}

__device__ __forceinline__ float fma_ftz(float a, float b, float c) {
    float r;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
    return r;
}

// tl.log, the captured integer reduction and the FTZ polynomial on the mantissa.
__device__ __forceinline__ float mufu_log(float x) {
    const unsigned xi = __float_as_uint(x);
    const int exp_bits = ((int)xi - 0x3f2aaaab) & (int)0xff800000;
    const float f = __fadd_rn(__uint_as_float(xi - (unsigned)exp_bits), -1.f);
    float p = __uint_as_float(0x3e055027u);
    p = fma_ftz(f, -p, __uint_as_float(0x3e1039f6u));
    p = fma_ftz(f, p, __uint_as_float(0xbdf8cdccu));
    p = fma_ftz(f, p, __uint_as_float(0x3e0f2955u));
    p = fma_ftz(f, p, __uint_as_float(0xbe2ad8b9u));
    p = fma_ftz(f, p, __uint_as_float(0x3e4ced0bu));
    p = fma_ftz(f, p, __uint_as_float(0xbe7fff22u));
    p = fma_ftz(f, p, __uint_as_float(0x3eaaaa78u));
    p = fma_ftz(f, p, -0.5f);
    float ln = fma_ftz(f, __fmul_rn(f, p), f);
    float e = fma_ftz((float)exp_bits, __uint_as_float(0x34000000u), 0.f);
    ln = fma_ftz(e, __uint_as_float(0x3f317218u), ln);
    if (xi >= 0x7f800000u) ln = fma_ftz(x, __uint_as_float(0x7f800000u), __uint_as_float(0x7f800000u));
    return ln;
}

// tl.sigmoid: rcp(1 + exp(-x)), with the huge-denominator scale the cubin applies.
__device__ __forceinline__ float mufu_sigmoid(float x) {
    float d = __fadd_rn(mufu_exp(-x), 1.f);
    float s = 1.f;
    if (d > __uint_as_float(0x7e800000u)) {
        d = __fmul_rn(d, 0.25f);
        s = 0.25f;
    }
    return __fmul_rn(mufu_rcp(d), s);
}

// tl.log(1 + tl.exp(a)), or a itself past 20.
__device__ __forceinline__ float mufu_softplus(float a) {
    if (a > 20.f) return a;
    return mufu_log(__fadd_rn(1.f, mufu_exp(a)));
}

// rcp(sqrt(ss * (1/n) + eps)). 1/n is exact for the power-of-two widths in the 27B shapes.
__device__ __forceinline__ float mufu_inv_rms(float ss, int n, float eps) {
    const float z = __fmaf_rn(ss, __fdiv_rn(1.f, static_cast<float>(n)), eps);
    return mufu_rcp(mufu_sqrt(z));
}

__device__ __forceinline__ float inv_rms(float ss, int n, float eps) {
    return __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, static_cast<float>(n)), eps)));
}

// Warp sum. Every lane keeps its own parenthesization: self plus the xor partner, from 16 down to 1.
__device__ __forceinline__ float bfly32(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffff, v, off));
    return v;
}

}  // namespace tf_q27m
