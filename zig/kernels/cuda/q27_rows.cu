// Row glue for the 27B. Width 5120 add-RMSNorm is the captured 256-thread sum; every other width stays one thread.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "glue_math.cuh"
#include "q27_math.cuh"

using bf16 = __nv_bfloat16;



__device__ __forceinline__ float pair_sq(float a, float b) { return __fmaf_rn(b, b, __fmul_rn(a, a)); }

__device__ __forceinline__ float chunk_hl(const float* e) {
    return __fadd_rn(__fadd_rn(pair_sq(e[4], e[5]), pair_sq(e[6], e[7])), __fadd_rn(pair_sq(e[2], e[3]), pair_sq(e[0], e[1])));
}

__device__ __forceinline__ float chunk_lh(const float* e) {
    return __fadd_rn(__fadd_rn(pair_sq(e[0], e[1]), pair_sq(e[2], e[3])), __fadd_rn(pair_sq(e[4], e[5]), pair_sq(e[6], e[7])));
}

__device__ __forceinline__ float add8(const float* e) {
    return __fadd_rn(__fadd_rn(__fadd_rn(e[0], e[1]), __fadd_rn(e[2], e[3])), __fadd_rn(__fadd_rn(e[4], e[5]), __fadd_rn(e[6], e[7])));
}

// Captured _group_sums: eight lanes sum eight bf16, then the lane-0 butterfly is the one the shared store keeps.
__device__ void group_sums(const bf16* x, float* xs, int groups) {
    for (int g = 0; g < groups; ++g) {
        float part[8];
        for (int lane = 0; lane < 8; ++lane) {
            float e[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) e[k] = tf_glue::bf(x[g * 64 + lane * 8 + k]);
            part[lane] = add8(e);
        }
        for (int off = 4; off > 0; off >>= 1) {
            float next[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) next[i] = __fadd_rn(part[i], part[i ^ off]);
#pragma unroll
            for (int i = 0; i < 8; ++i) part[i] = next[i];
        }
        xs[g] = part[0];
    }
}

__device__ __forceinline__ float bfly_from(float v, int off) {
    for (; off > 0; off >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffff, v, off));
    return v;
}

__device__ void load8(const bf16* x, const bf16* r, bf16* h, int base, int has_r, float* e) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float xv = tf_glue::bf(x[base + i]);
        if (has_r) {
            const float hv = tf_glue::rbf(__fadd_rn(xv, tf_glue::bf(r[base + i])));
            h[base + i] = __float2bfloat16_rn(hv);
            xv = hv;
        }
        e[i] = xv;
    }
}

__device__ void store8(bf16* y, const bf16* w, int base, const float* v, float inv) {
#pragma unroll
    for (int i = 0; i < 8; ++i) y[base + i] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(v[i], inv), tf_glue::bf(w[base + i])));
}

// h = bf16(x + r) when has_r, else x is left untouched. y = bf16((x * inv) * w). xs sums the stored y.
__device__ void add_rms_5120(const bf16* x, const bf16* r, const bf16* w, bf16* h, bf16* y, float* xs, float eps, int has_r,
                             float* ws) {
    const int tid = threadIdx.x;
    const size_t at = (size_t)blockIdx.x * 5120;
    const bf16* xr = x + at;
    const bf16* rr = r + at;
    bf16* hr = h + at;
    bf16* yr = y + at;
    float a[8], b[8], c[8];
    load8(xr, rr, hr, tid * 8, has_r, a);
    load8(xr, rr, hr, 2048 + tid * 8, has_r, b);
    const int third = tid < 128;
    if (third) load8(xr, rr, hr, 4096 + tid * 8, has_r, c);
    const float c2 = third ? chunk_lh(c) : 0.f;
    const float partial = __fadd_rn(__fadd_rn(chunk_hl(a), chunk_lh(b)), c2);
    const float wv = bfly_from(partial, 16);
    if ((tid & 31) == 0) ws[tid >> 5] = wv;
    __syncthreads();
    float inv_s, inv;
    const float z = __fmaf_rn(bfly_from(ws[tid & 7], 4), __uint_as_float(0x394ccccdu), eps);
    asm("sqrt.approx.f32 %0, %1;" : "=f"(inv_s) : "f"(z));
    asm("rcp.approx.f32 %0, %1;" : "=f"(inv) : "f"(inv_s));
    store8(yr, w, tid * 8, a, inv);
    store8(yr, w, 2048 + tid * 8, b, inv);
    if (third) store8(yr, w, 4096 + tid * 8, c, inv);
    float e0[8], e1[8], e2[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        e0[i] = tf_glue::bf(yr[tid * 8 + i]);
        e1[i] = tf_glue::bf(yr[2048 + tid * 8 + i]);
        if (third) e2[i] = tf_glue::bf(yr[4096 + tid * 8 + i]);
    }
    const float g0 = bfly_from(add8(e0), 4);
    const float g1 = bfly_from(add8(e1), 4);
    const float g2 = third ? bfly_from(add8(e2), 4) : 0.f;
    if ((tid & 7) == 0) {
        float* xsr = xs + (size_t)blockIdx.x * 80;
        xsr[tid >> 3] = g0;
        xsr[32 + (tid >> 3)] = g1;
        if (third) xsr[64 + (tid >> 3)] = g2;
    }
}

extern "C" __global__ void tf_q27_add_rmsnorm(const bf16* __restrict__ x, const bf16* __restrict__ r,
                                              const bf16* __restrict__ w, bf16* __restrict__ h, bf16* __restrict__ y,
                                              float* __restrict__ xs, float eps, int d, int has_r) {
    __shared__ float ws[8];
    if (d == 5120) {
        add_rms_5120(x, r, w, h, y, xs, eps, has_r, ws);
        return;
    }
    if (threadIdx.x != 0) return;
    const int row = blockIdx.x;
    const size_t at = (size_t)row * d;
    float ss = 0.f;
    for (int i = 0; i < d; ++i) {
        float xv = tf_glue::bf(x[at + i]);
        if (has_r) {
            const float hv = tf_glue::rbf(__fadd_rn(xv, tf_glue::bf(r[at + i])));
            h[at + i] = __float2bfloat16_rn(hv);
            xv = hv;
        }
        ss = __fmaf_rn(xv, xv, ss);
    }
    const float inv = tf_q27m::inv_rms(ss, d, eps);
    for (int i = 0; i < d; ++i) {
        const float xv = has_r ? tf_glue::bf(h[at + i]) : tf_glue::bf(x[at + i]);
        const float yv = tf_glue::rbf(__fmul_rn(__fmul_rn(xv, inv), tf_glue::bf(w[i])));
        y[at + i] = __float2bfloat16_rn(yv);
    }
    group_sums(y + at, xs + (size_t)row * (d / 64), d / 64);
}

__device__ void gated_store(const bf16* y, const bf16* z, const bf16* w, bf16* out, int i, float inv) {
    const float yn = __fmul_rn(__fmul_rn(tf_glue::bf(y[i]), inv), tf_glue::bf(w[i]));
    const float zv = tf_glue::bf(z[i]);
    const float silu = __fmul_rn(zv, tf_q27m::mufu_sigmoid(zv));
    out[i] = __float2bfloat16_rn(tf_glue::rbf(__fmul_rn(silu, yn)));
}

// silu(z) * rmsnorm(y) * w per value head, then group sums. dv 128: lane L sums four squares and keeps its butterfly.
extern "C" __global__ void tf_q27_gated_norm(const bf16* __restrict__ y, const bf16* __restrict__ z,
                                             const bf16* __restrict__ w, bf16* __restrict__ out, float* __restrict__ xs,
                                             float eps, int vh, int dv) {
    const int row = blockIdx.x, head = blockIdx.y;
    const size_t at = ((size_t)row * vh + head) * dv;
    if (dv == 128) {
        const int lane = threadIdx.x;
        const int b = lane * 4;
        const float e0 = tf_glue::bf(y[at + b]), e1 = tf_glue::bf(y[at + b + 1]);
        const float e2 = tf_glue::bf(y[at + b + 2]), e3 = tf_glue::bf(y[at + b + 3]);
        const float pairs = __fadd_rn(__fmaf_rn(e1, e1, __fmul_rn(e0, e0)), __fmaf_rn(e3, e3, __fmul_rn(e2, e2)));
        const float inv = tf_q27m::mufu_inv_rms(tf_q27m::bfly32(pairs), dv, eps);
        for (int k = 0; k < 4; ++k) gated_store(y + at, z + at, w, out + at, b + k, inv);
        __syncthreads();
        if (lane == 0) group_sums(out + at, xs + (size_t)row * (vh * dv / 64) + head * (dv / 64), dv / 64);
        return;
    }
    if (threadIdx.x != 0) return;
    float ss = 0.f;
    for (int i = 0; i < dv; ++i) {
        const float yv = tf_glue::bf(y[at + i]);
        ss = __fmaf_rn(yv, yv, ss);
    }
    const float inv = tf_q27m::mufu_inv_rms(ss, dv, eps);
    for (int i = 0; i < dv; ++i) gated_store(y + at, z + at, w, out + at, i, inv);
    group_sums(out + at, xs + (size_t)row * (vh * dv / 64) + head * (dv / 64), dv / 64);
}

// out = bf16((g * sigmoid(g)) * u), and the row's group sums.
extern "C" __global__ void tf_q27_swiglu(const bf16* __restrict__ gate, const bf16* __restrict__ up, bf16* __restrict__ out,
                                         float* __restrict__ xs, int n) {
    if (threadIdx.x != 0) return;
    const int row = blockIdx.x;
    const size_t at = (size_t)row * n;
    for (int i = 0; i < n; ++i) {
        const float gv = tf_glue::bf(gate[at + i]);
        const float prod = __fmul_rn(__fmul_rn(gv, tf_q27m::mufu_sigmoid(gv)), tf_glue::bf(up[at + i]));
        out[at + i] = __float2bfloat16_rn(tf_glue::rbf(prod));
    }
    group_sums(out + at, xs + (size_t)row * (n / 64), n / 64);
}

// out = bf16(o * sigmoid(gate)). Gate sits D elements into the [q | gate] pair.
extern "C" __global__ void tf_q27_gate_mul(const bf16* __restrict__ o, const bf16* __restrict__ qg, bf16* __restrict__ out,
                                           float* __restrict__ xs, int heads, int d) {
    if (threadIdx.x != 0) return;
    const int row = blockIdx.x, head = blockIdx.y;
    const size_t src = ((size_t)row * heads + head) * d;
    const size_t gate = ((size_t)row * heads + head) * 2 * d + d;
    for (int i = 0; i < d; ++i) {
        const float gv = tf_glue::bf(qg[gate + i]);
        const float prod = __fmul_rn(tf_glue::bf(o[src + i]), tf_glue::tf_sigmoid(gv));
        out[src + i] = __float2bfloat16_rn(tf_glue::rbf(prod));
    }
    group_sums(out + src, xs + (size_t)row * (heads * d / 64) + head * (d / 64), d / 64);
}

// 64-wide group sums of a bf16 row that may be strided by ld. One thread a row, index order.
extern "C" __global__ void tf_q27_group_sums(const bf16* __restrict__ x, float* __restrict__ xs, int ld, int k) {
    if (threadIdx.x != 0) return;
    const int row = blockIdx.x;
    const int groups = k / 64;
    group_sums(x + (size_t)row * ld, xs + (size_t)row * groups, groups);
}

// MLX 4-bit row: eight u32 words a 64-group, nibble then bf16(q * s + b) as a mul and an add.
extern "C" __global__ void tf_q27_embed(const int* __restrict__ ids, const unsigned* __restrict__ words,
                                        const bf16* __restrict__ scales, const bf16* __restrict__ bias, bf16* __restrict__ out,
                                        int d) {
    if (threadIdx.x != 0) return;
    const int row = blockIdx.x;
    const int tok = ids[row];
    const int groups = d / 64;
    for (int g = 0; g < groups; ++g) {
        const float s = tf_glue::bf(scales[(size_t)tok * groups + g]);
        const float b = tf_glue::bf(bias[(size_t)tok * groups + g]);
        for (int w = 0; w < 8; ++w) {
            const unsigned word = words[(size_t)tok * (d / 8) + g * 8 + w];
            for (int n = 0; n < 8; ++n) {
                const float q = static_cast<float>((word >> (4 * n)) & 0xFu);
                const int i = g * 64 + w * 8 + n;
                out[(size_t)row * d + i] = __float2bfloat16_rn(tf_glue::rbf(__fadd_rn(__fmul_rn(q, s), b)));
            }
        }
    }
}

// bf16 table row copied as stored. Any block size: threads stride the row.
extern "C" __global__ void tf_q27_b16_embed(const int* __restrict__ ids, const bf16* __restrict__ table, bf16* __restrict__ out,
                                            int d) {
    const int row = blockIdx.x;
    const bf16* src = table + (size_t)ids[row] * d;
    bf16* dst = out + (size_t)row * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) dst[i] = src[i];
}

// bf16 to fp16 through fp32, round to nearest even, the cast an fp16 weight reads.
extern "C" __global__ void tf_q27_bf16_to_f16(const bf16* __restrict__ x, __half* __restrict__ y, long long n) {
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        y[i] = __float2half_rn(__bfloat162float(x[i]));
}
