// Device code of src/tensorfold/families/qwen3_5/cuda/b16.cu (python-0.6, lines 1-216): a row's bits are its own.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace tf_b16 {

template <typename T>
__device__ __forceinline__ float f2f(T v);
template <>
__device__ __forceinline__ float f2f<half>(half v) { return __half2float(v); }
template <>
__device__ __forceinline__ float f2f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ __forceinline__ T f_from(float v);
template <>
__device__ __forceinline__ half f_from<half>(float v) { return __float2half_rn(v); }
template <>
__device__ __forceinline__ __nv_bfloat16 f_from<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

// Eight elements (16 bytes) widened to fp32, either dtype.
template <typename T>
__device__ __forceinline__ void ld8(const T* __restrict__ p, float (&v)[8]) {
    const uint4 u = __ldg(reinterpret_cast<const uint4*>(p));
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if constexpr (std::is_same_v<T, half>) {
            const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&w[i]));
            v[2 * i] = f.x; v[2 * i + 1] = f.y;
        } else {
            const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w[i]));
            v[2 * i] = f.x; v[2 * i + 1] = f.y;
        }
    }
}

// Grid (ceil(N/WARPS), ceil(M/R)): a warp one column for R rows, a shared weight load; K % 8 == 0, 16-byte rows.
template <typename T, int R, int WARPS>
__global__ void __launch_bounds__(WARPS * 32) b16_kernel(const T* __restrict__ x, const T* __restrict__ w0,
                                                         const T* __restrict__ bias, T* __restrict__ y0, int M, int K,
                                                         int N0, const T* __restrict__ w1, T* __restrict__ y1, int N1) {
    const T* w = blockIdx.z ? w1 : w0;                     // grid z 2: a second weight of the same rows, its own out
    T* y = blockIdx.z ? y1 : y0;
    const int N = blockIdx.z ? N1 : N0;
    const int lane = threadIdx.x & 31;
    const int row0 = blockIdx.y * R;
    const int col = blockIdx.x * WARPS + (threadIdx.x >> 5);
    if (col >= N) return;
    const T* wr = w + (size_t)col * K;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;
    for (int k = 8 * lane; k + 8 <= K; k += 256) {
        float b[8];
        ld8(wr + k, b);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (row0 + r >= M) break;
            float a[8];
            ld8(x + (size_t)(row0 + r) * K + k, a);
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[r] = fmaf(a[i], b[i], acc[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (row0 + r >= M) break;
        float v = acc[r];
#pragma unroll
        for (int m = 16; m >= 1; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
        if (lane == 0) {
            if (bias != nullptr) v += f2f(__ldg(bias + col));
            y[(size_t)(row0 + r) * N + col] = f_from<T>(v);
        }
    }
}

// Prompt rows on the bf16 mma: BM x 64 tiles, 64 inputs a stage, one fp32 chain over K a row (bits chunk-invariant).
constexpr int PBN = 64, PST = 3, PROW = 128;

template <int BM>
constexpr int pstage() { return (BM + PBN) * PROW; }

__device__ __forceinline__ uint32_t sm(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(sm(p)));
}

__device__ __forceinline__ void cpz(void* dst, const void* src, bool ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sm(dst)), "l"(src), "r"(ok ? 16 : 0));
}

__device__ __forceinline__ int pswz(int r, int c) { return r * PROW + ((c ^ (r & 7)) << 4); }

template <int BM>
__global__ void __launch_bounds__(128) b16_prompt_kernel(const __nv_bfloat16* __restrict__ x,
                                                         const __nv_bfloat16* __restrict__ w0,
                                                         __nv_bfloat16* __restrict__ y0, int M, int K, int N0,
                                                         const __nv_bfloat16* __restrict__ w1,
                                                         __nv_bfloat16* __restrict__ y1, int N1) {
    const __nv_bfloat16* w = blockIdx.z ? w1 : w0;         // grid z 2: a second weight of the same rows
    __nv_bfloat16* y = blockIdx.z ? y1 : y0;
    const int N = blockIdx.z ? N1 : N0;
    constexpr int MI = BM / 32, STAGE = pstage<BM>();                     // m16 tiles a warp
    extern __shared__ __align__(128) unsigned char pbuf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp >> 1, wn = warp & 1;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * PBN, KT = K / 64;
    auto load = [&](int s, int kt) {
        unsigned char* p = pbuf + s * STAGE;
        for (int c = tid; c < (BM + PBN) * 8; c += 128) {
            const int r = c >> 3, ch = c & 7;
            const bool wrow = r >= BM;
            const int src_row = wrow ? n0 + r - BM : m0 + r, lim = wrow ? N : M;
            const __nv_bfloat16* base = wrow ? w : x;
            cpz(p + pswz(r, ch), base + (size_t)(src_row < lim ? src_row : 0) * K + kt * 64 + ch * 8, src_row < lim);
        }
    };
    float acc[MI][4][4] = {};
    for (int s = 0; s < PST - 1; ++s) {
        if (s < KT) load(s, s);
        asm volatile("cp.async.commit_group;\n" ::);
    }
    for (int kt = 0; kt < KT; ++kt) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(PST - 2));
        __syncthreads();
        if (kt + PST - 1 < KT) load((kt + PST - 1) % PST, kt + PST - 1);
        asm volatile("cp.async.commit_group;\n" ::);
        const unsigned char* p = pbuf + (kt % PST) * STAGE;
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            uint32_t a[MI][4], b[2][4];
#pragma unroll
            for (int i = 0; i < MI; ++i)
                ldsm4(a[i], p + pswz(wm * (BM / 2) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8,
                                     ks * 2 + (lane >> 4)));
#pragma unroll
            for (int j = 0; j < 2; ++j)
                ldsm4(b[j], p + pswz(BM + wn * 32 + j * 16 + (lane & 7) + ((lane >> 4) << 3),
                                     ks * 2 + ((lane >> 3) & 1)));
#pragma unroll
            for (int i = 0; i < MI; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, "
                                 "{%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                 : "+f"(acc[i][j][0]), "+f"(acc[i][j][1]), "+f"(acc[i][j][2]), "+f"(acc[i][j][3])
                                 : "r"(a[i][0]), "r"(a[i][1]), "r"(a[i][2]), "r"(a[i][3]), "r"(b[j >> 1][(j & 1) * 2]),
                                   "r"(b[j >> 1][(j & 1) * 2 + 1]));
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * (BM / 2) + i * 16 + (lane >> 2) + h * 8;
                const int col = n0 + wn * 32 + j * 8 + (lane & 3) * 2;
                if (row >= M) continue;
                if (col < N) y[(size_t)row * N + col] = __float2bfloat16_rn(acc[i][j][2 * h]);
                if (col + 1 < N) y[(size_t)row * N + col + 1] = __float2bfloat16_rn(acc[i][j][2 * h + 1]);
            }
}

}  // namespace tf_b16

// The instantiations the native runtime launches: both dtypes at every row band, the prompt tiles, both dtypes.
#define TF_B16_ROWS(T, R, W)                                                                                       \
    template __global__ void tf_b16::b16_kernel<T, R, W>(const T*, const T*, const T*, T*, int, int, int,          \
                                                         const T*, T*, int);
#define TF_B16_ALL(T) TF_B16_ROWS(T, 1, 4) TF_B16_ROWS(T, 4, 8) TF_B16_ROWS(T, 16, 16)
TF_B16_ALL(__nv_bfloat16)
TF_B16_ALL(half)

#define TF_B16_PROMPT(BM)                                                                                          \
    template __global__ void tf_b16::b16_prompt_kernel<BM>(const __nv_bfloat16*, const __nv_bfloat16*,             \
                                                           __nv_bfloat16*, int, int, int, const __nv_bfloat16*,     \
                                                           __nv_bfloat16*, int);
TF_B16_PROMPT(32)
TF_B16_PROMPT(64)
TF_B16_PROMPT(128)
