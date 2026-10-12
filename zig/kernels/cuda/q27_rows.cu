//! Row glue of the Qwen3.8-27B decode path: add+RMSNorm, gated norm, SwiGLU and embedding rows.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "glue_math.cuh"

namespace tf_q27 {

using bf16 = __nv_bfloat16;

// One row's 64-wide group sums of a bf16 vector, written fp32 (fixed lane order, row-independent).
__device__ __forceinline__ void group_sums(const bf16* x, float* xs, int groups) {
    const int lane = threadIdx.x;
    for (int g = lane; g < groups; g += 32) {
        float sum = 0.f;
#pragma unroll 8
        for (int j = 0; j < 64; ++j) sum += tf_glue::bf(x[g * 64 + j]);
        xs[g] = sum;
    }
}

// h = x + r rounded bf16; y = rmsnorm(h) * w rounded bf16; xs = y's group sums. One warp a row.
__global__ void __launch_bounds__(32) add_rmsnorm(const bf16* __restrict__ x, const bf16* __restrict__ r,
                                                  const bf16* __restrict__ w, bf16* __restrict__ h,
                                                  bf16* __restrict__ y, float* __restrict__ xs, float eps, int d) {
    const int row = blockIdx.x, lane = threadIdx.x;
    const size_t at = (size_t)row * d;
    float acc = 0.f;
    for (int i = lane; i < d; i += 32) {
        const float hv = tf_glue::rbf(tf_glue::bf(x[at + i]) + tf_glue::bf(r[at + i]));
        h[at + i] = __float2bfloat16(hv);
        acc = __fmaf_rn(hv, hv, acc);
    }
#pragma unroll
    for (int m = 16; m; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    const float inv = rsqrtf(__fdividef(acc, (float)d) + eps);
    for (int i = lane; i < d; i += 32)
        y[at + i] = __float2bfloat16(tf_glue::rbf(tf_glue::bf(h[at + i]) * inv * tf_glue::bf(w[i])));
    __syncwarp();
    group_sums(y + at, xs + (size_t)row * (d / 64), d / 64);
}

// out = silu(z) * rmsnorm(y) * w rounded bf16, and its group sums. One warp a (row, head).
__global__ void __launch_bounds__(32) gated_norm(const bf16* __restrict__ y, const bf16* __restrict__ z,
                                                 const bf16* __restrict__ w, bf16* __restrict__ out,
                                                 float* __restrict__ xs, float eps, int vh, int dv) {
    const int row = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
    const size_t src = ((size_t)row * vh + head) * dv;
    const size_t dst = (size_t)row * vh * dv + head * dv;
    float acc = 0.f;
    for (int i = lane; i < dv; i += 32) acc = __fmaf_rn(tf_glue::bf(y[src + i]), tf_glue::bf(y[src + i]), acc);
#pragma unroll
    for (int m = 16; m; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    const float inv = rsqrtf(__fdividef(acc, (float)dv) + eps);
    for (int i = lane; i < dv; i += 32) {
        const float zv = tf_glue::bf(z[src + i]);
        const float silu = tf_glue::rbf(zv / (1.f + tf_glue::tf_exp(-zv)));
        out[dst + i] = __float2bfloat16(tf_glue::rbf(silu * (tf_glue::bf(y[src + i]) * inv) * tf_glue::bf(w[i])));
    }
    __syncwarp();
    group_sums(out + dst, xs + (size_t)row * (vh * dv / 64) + head * (dv / 64), dv / 64);
}

// out = silu(gate) * up rounded bf16, and its group sums. One warp a (row, 64-group).
__global__ void __launch_bounds__(32) swiglu(const bf16* __restrict__ gate, const bf16* __restrict__ up,
                                             bf16* __restrict__ out, float* __restrict__ xs, int n) {
    const int row = blockIdx.x, g = blockIdx.y, lane = threadIdx.x;
    float sum = 0.f;
#pragma unroll
    for (int j = lane * 2; j < 64; j += 64) {
        const size_t at = (size_t)row * n + g * 64 + j;
        const float gv = tf_glue::bf(gate[at]);
        const float v = tf_glue::rbf(tf_glue::rbf(gv / (1.f + tf_glue::tf_exp(-gv))) * tf_glue::bf(up[at]));
        out[at] = __float2bfloat16(v);
        sum += v;
    }
    sum += __shfl_xor_sync(0xffffffffu, sum, 1);   // pairs of lanes hold 2 elements each
    sum += __shfl_xor_sync(0xffffffffu, sum, 16);  // 16 lane-pairs make the 64-span
    if (lane == 0) xs[(size_t)row * (n / 64) + g] = sum;
}

// out = o * sigmoid(gate) rounded bf16, and its group sums. One warp a (row, head).
__global__ void __launch_bounds__(32) gate_mul(const bf16* __restrict__ o, const bf16* __restrict__ qg,
                                               bf16* __restrict__ out, float* __restrict__ xs, int heads, int d) {
    const int row = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
    const size_t src = ((size_t)row * heads + head) * d;
    const size_t gate_at = src + d;
    const size_t dst = (size_t)row * heads * d + head * d;
    for (int i = lane; i < d; i += 32) {
        const float g = tf_glue::bf(qg[gate_at + i]);
        out[dst + i] = __float2bfloat16(tf_glue::rbf(tf_glue::bf(o[src + i]) / (1.f + tf_glue::tf_exp(-g))));
    }
    __syncwarp();
    group_sums(out + dst, xs + (size_t)row * (heads * d / 64) + head * (d / 64), d / 64);
}

// A bf16 embedding row gather: out = table[id] as stored (the EXL3 pack's plain table); ids are int32 rows.
__global__ void __launch_bounds__(32) b16_embed(const int* __restrict__ ids, const bf16* __restrict__ table,
                                                bf16* __restrict__ out, int d) {
    const int row = blockIdx.x, lane = threadIdx.x;
    const bf16* src = table + (size_t)ids[row] * d;
    bf16* dst = out + (size_t)row * d;
    for (int i = lane * 8; i < d; i += 256) *reinterpret_cast<uint4*>(dst + i) = *reinterpret_cast<const uint4*>(src + i);
}

// bf16 values to fp16 as torch's .to(float16) rounds them (through fp32, to nearest even): an fp16 weight's inputs.
__global__ void __launch_bounds__(256) bf16_to_f16(const bf16* __restrict__ x, __half* __restrict__ y, long long n) {
    for (long long i = blockIdx.x * 256LL + threadIdx.x; i < n; i += (long long)gridDim.x * 256)
        y[i] = __float2half_rn(__bfloat162float(x[i]));
}

}  // namespace tf_q27
