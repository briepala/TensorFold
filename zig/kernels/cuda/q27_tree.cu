//! Tree attention over committed keys and drafted paths: fixed folds, a row's bits never depend on the window.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "glue_math.cuh"

namespace tf_q27 {

using bf16 = __nv_bfloat16;

constexpr int TILE = 64;    // keys a sub-tile
constexpr int CHUNK = 512;  // keys a chunk (a slot's fold unit)
constexpr int CHUNK_TILES = CHUNK / TILE;
constexpr int GR = 4;       // chunks folded into one whole group
constexpr int MAXD = 128;   // the deepest drafted path (also the head dim this family runs)

// One row's streaming softmax state: the running max, denominator and D-wide output (a lane owns 4 elements).
struct Acc {
    float m, l;
    float o[4];
};

// Fold one 64-key tile held in shared memory into the running state (_tile's math, tf_glue exps, fixed order).
__device__ __forceinline__ void tile_rows(const bf16* q, const bf16 (*kbuf)[MAXD], const bf16 (*vbuf)[MAXD], Acc& a,
                                          const int* valid, float scale, int d) {
    const int lane = threadIdx.x & 31;
    float s1 = -INFINITY, s2 = -INFINITY;
    const bool v1 = valid[lane] != 0, v2 = valid[32 + lane] != 0;
    if (v1 || v2) {
        float dot1 = 0.f, dot2 = 0.f;
        const bf16* kr1 = kbuf[lane];
        const bf16* kr2 = kbuf[32 + lane];
        for (int i = 0; i < d; ++i) {
            const float qv = tf_glue::bf(q[i]);
            if (v1) dot1 = __fmaf_rn(qv, tf_glue::bf(kr1[i]), dot1);
            if (v2) dot2 = __fmaf_rn(qv, tf_glue::bf(kr2[i]), dot2);
        }
        if (v1) s1 = dot1 * scale;
        if (v2) s2 = dot2 * scale;
    }
    float tile_m = fmaxf(s1, s2);
#pragma unroll
    for (int mm = 16; mm; mm >>= 1) tile_m = fmaxf(tile_m, __shfl_xor_sync(0xffffffffu, tile_m, mm));
    const bool active = tile_m != -INFINITY;
    float next_m = a.m;
    if (active && tile_m > next_m) next_m = tile_m;
    const float alpha = !active ? 1.f : (a.m == -INFINITY ? 0.f : tf_glue::tf_exp(a.m - next_m));
    const float p1 = v1 && active ? tf_glue::tf_exp(s1 - next_m) : 0.f;
    const float p2 = v2 && active ? tf_glue::tf_exp(s2 - next_m) : 0.f;
    float psum = p1 + p2;
#pragma unroll
    for (int mm = 16; mm; mm >>= 1) psum += __shfl_xor_sync(0xffffffffu, psum, mm);
    a.l = a.l * alpha + psum;
#pragma unroll
    for (int e = 0; e < 4; ++e) a.o[e] *= alpha;
    for (int j = 0; j < 32; ++j) {
        const float pj = __shfl_sync(0xffffffffu, p1, j);
        if (pj != 0.f) {
            const float pb = tf_glue::bf(__float2bfloat16_rn(pj));  // p rounded bf16 as the mma tile carries it
            const bf16* vr = vbuf[j];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int i = lane + e * 32;
                if (i < d) a.o[e] = __fmaf_rn(pb, tf_glue::bf(vr[i]), a.o[e]);
            }
        }
    }
    for (int j = 0; j < 32; ++j) {
        const float pj = __shfl_sync(0xffffffffu, p2, j);
        if (pj != 0.f) {
            const float pb = tf_glue::bf(__float2bfloat16_rn(pj));
            const bf16* vr = vbuf[32 + j];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int i = lane + e * 32;
                if (i < d) a.o[e] = __fmaf_rn(pb, tf_glue::bf(vr[i]), a.o[e]);
            }
        }
    }
    a.m = active ? next_m : a.m;
}

// The group fold (_fold): a partial's (cm, cl, co) folded in; empty partials are identity.
__device__ __forceinline__ void fold(float& m, float& l, float o[4], float cm, float cl, const float co[4]) {
    const bool active = cl > 0.f;
    float next_m = m;
    if (active && cm > next_m) next_m = cm;
    const float a = !active ? 1.f : (m == -INFINITY ? 0.f : tf_glue::tf_exp(m - next_m));
    const float b = active ? tf_glue::tf_exp(cm - next_m) : 0.f;
    l = l * a + cl * b;
#pragma unroll
    for (int e = 0; e < 4; ++e) o[e] = o[e] * a + co[e] * b;
    m = next_m;
}

// Each node's root-to-row path (parents are window rows, -1 at a root) and its depth. One thread a node.
__global__ void __launch_bounds__(128) paths_kernel(const int* __restrict__ parents, int* __restrict__ paths,
                                                    int* __restrict__ depths, int w) {
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= w) return;
    int cur = node, depth = 0;
    while (cur >= 0 && depth < MAXD) {
        ++depth;
        cur = parents[cur];
    }
    depths[node] = depth;
    cur = node;
    for (int slot = depth - 1; slot >= 0; --slot) {
        paths[node * MAXD + slot] = cur;
        cur = parents[cur];
    }
}

// The committed-key items: whole groups (4 chunks folded) and fully-committed tail chunks. A warp a (row, head).
__global__ void __launch_bounds__(512) shared_kernel(const bf16* __restrict__ Q, const unsigned long long* __restrict__ OFF,
                                                     const int* __restrict__ STREAM, const int* __restrict__ ITEMS,
                                                     float* __restrict__ PO, float* __restrict__ PM, float* __restrict__ PL,
                                                     int W, int H, int HK, int G, int D, float scale) {
    const int item = blockIdx.x / HK, hk = blockIdx.x % HK;
    const int s = ITEMS[item * 3], first = ITEMS[item * 3 + 1], code = ITEMS[item * 3 + 2];
    const int start = STREAM[s * 4], rows = STREAM[s * 4 + 1], p = STREAM[s * 4 + 2], nslots = STREAM[s * 4 + 3];
    const int nch = (p + rows + CHUNK - 1) / CHUNK;
    const int groups = (nch - nslots) / (GR - 1);
    const bool whole = code >= 0;
    const int chunk = whole ? code * GR : groups * GR - 1 - code;
    const int n = whole ? GR : 1;
    const int slot = whole ? code : chunk - groups * (GR - 1);
    if ((chunk + n) * CHUNK > p) return;  // a whole group must lie inside the committed keys
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int rr = first + warp;  // this warp's (row, head) pair
    const bool ok = rr < rows * G;
    const int node = start + rr / G, head = hk * G + rr % G;
    __shared__ bf16 qs[16][MAXD];
    __shared__ bf16 kbuf[TILE][MAXD], vbuf[TILE][MAXD];
    __shared__ int valid[TILE];
    if (ok) {
        for (int i = lane; i < D; i += 32) qs[warp][i] = Q[((size_t)node * H + head) * D + i];
    }
    __syncthreads();
    const bf16* KC = reinterpret_cast<const bf16*>(OFF[s * 2]);
    const bf16* VC = reinterpret_cast<const bf16*>(OFF[s * 2 + 1]);
    Acc ga = {};
    ga.m = -INFINITY;
    for (int c = 0; c < n && ok; ++c) {
        Acc ca = {};
        ca.m = -INFINITY;
        for (int t = 0; t < CHUNK_TILES; ++t) {
            const int key0 = (chunk + c) * CHUNK + t * TILE;
            for (int j = 0; j < TILE; ++j) {
                const bool on = key0 + j < p;
                if (lane == j % 32 && j / 32 == 0) valid[j] = on ? 1 : 0;
                const bf16* kr = KC + ((size_t)(key0 + j) * HK + hk) * D;
                const bf16* vr = VC + ((size_t)(key0 + j) * HK + hk) * D;
                for (int i = lane; i < D; i += 32) {
                    kbuf[j][i] = on ? kr[i] : __float2bfloat16(0.f);
                    vbuf[j][i] = on ? vr[i] : __float2bfloat16(0.f);
                }
                if (lane == 0) valid[j] = on ? 1 : 0;
            }
            __syncthreads();
            tile_rows(qs[warp], kbuf, vbuf, ca, valid, scale, D);
            __syncthreads();
        }
        if (whole) {
            fold(ga.m, ga.l, ga.o, ca.m, ca.l, ca.o);
        } else {  // a tail item's partial goes out unfolded
            ga = ca;
        }
    }
    if (!ok) return;
    const size_t base = ((size_t)slot * W + node) * H + head;
    if (lane == 0) {
        PM[base] = ga.m;
        PL[base] = ga.l;
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int i = lane + e * 32;
        if (i < D) PO[base * D + i] = ga.o[e];
    }
}

// The tail chunks: the last committed keys mixed with each row's own path keys. A warp a head of the node.
__global__ void __launch_bounds__(512) tail_kernel(const bf16* __restrict__ Q, const bf16* __restrict__ KN,
                                                   const bf16* __restrict__ VN, const unsigned long long* __restrict__ OFF,
                                                   const int* __restrict__ STREAM, const int* __restrict__ ROWS,
                                                   const int* __restrict__ PATHS, const int* __restrict__ DEPTHS,
                                                   float* __restrict__ PO, float* __restrict__ PM, float* __restrict__ PL,
                                                   int W, int H, int HK, int G, int D, float scale) {
    const int node = blockIdx.x, hk = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp >= G) return;
    const int head = hk * G + warp;
    const int s = ROWS[node];
    const int p = STREAM[s * 4 + 2], rows = STREAM[s * 4 + 1];
    const int nslots = STREAM[s * 4 + 3];
    const int nch = (p + rows + CHUNK - 1) / CHUNK;
    const int groups = (nch - nslots) / (GR - 1);
    const int chunk = p / CHUNK + blockIdx.z;
    if (chunk >= nch) return;
    const bf16* KC = reinterpret_cast<const bf16*>(OFF[s * 2]);
    const bf16* VC = reinterpret_cast<const bf16*>(OFF[s * 2 + 1]);
    __shared__ bf16 qs[16][MAXD];
    __shared__ bf16 kbuf[TILE][MAXD], vbuf[TILE][MAXD];
    __shared__ int valid[TILE];
    const int depth = DEPTHS[node];
    for (int i = lane; i < D; i += 32) qs[warp][i] = Q[((size_t)node * H + head) * D + i];
    __syncthreads();
    Acc a = {};
    a.m = -INFINITY;
    for (int t = 0; t < CHUNK_TILES; ++t) {
        for (int j = 0; j < TILE; ++j) {
            const int logical = chunk * CHUNK + t * TILE + j;
            const bool committed = logical < p;
            const int path_slot = logical - p;
            const bool on_path = path_slot >= 0 && path_slot < depth;
            const int kn = on_path ? PATHS[node * MAXD + path_slot] : 0;
            const bf16* ksrc = committed ? KC + ((size_t)logical * HK + hk) * D
                                         : (on_path ? KN + ((size_t)kn * HK + hk) * D : kbuf[j]);
            const bf16* vsrc = committed ? VC + ((size_t)logical * HK + hk) * D
                                         : (on_path ? VN + ((size_t)kn * HK + hk) * D : vbuf[j]);
            for (int i = lane; i < D; i += 32) {
                kbuf[j][i] = committed || on_path ? ksrc[i] : __float2bfloat16(0.f);
                vbuf[j][i] = committed || on_path ? vsrc[i] : __float2bfloat16(0.f);
            }
            if (lane == 0) valid[j] = (committed || on_path) ? 1 : 0;
        }
        __syncthreads();
        tile_rows(qs[warp], kbuf, vbuf, a, valid, scale, D);
        __syncthreads();
    }
    const size_t base = ((size_t)(chunk - groups * (GR - 1)) * W + node) * H + head;
    if (lane == 0) {
        PM[base] = a.m;
        PL[base] = a.l;
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int i = lane + e * 32;
        if (i < D) PO[base * D + i] = a.o[e];
    }
}

// Fold every slot in order (whole groups, then tail chunks in GR batches) and write o / l. A warp a head.
__global__ void __launch_bounds__(512) merge_kernel(const float* __restrict__ PO, const float* __restrict__ PM,
                                                    const float* __restrict__ PL, bf16* __restrict__ OUT,
                                                    const int* __restrict__ STREAM, const int* __restrict__ ROWS,
                                                    int W, int H, int HK, int G, int D) {
    const int node = blockIdx.x, hk = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp >= G) return;
    const int head = hk * G + warp;
    const int s = ROWS[node];
    const int rows = STREAM[s * 4 + 1], p = STREAM[s * 4 + 2], nslots = STREAM[s * 4 + 3];
    const int nch = (p + rows + CHUNK - 1) / CHUNK;
    const int groups = (nch - nslots) / (GR - 1);
    float m = -INFINITY, l = 0.f;
    float o[4] = {0.f, 0.f, 0.f, 0.f};
    float cm = 0.f, cl = 0.f;
    float co[4] = {0.f, 0.f, 0.f, 0.f};
    for (int slot = 0; slot < groups; ++slot) {
        const size_t base = ((size_t)slot * W + node) * H + head;
        cm = PM[base];
        cl = PL[base];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int i = lane + e * 32;
            if (i < D) co[e] = PO[base * D + i];
        }
        fold(m, l, o, cm, cl, co);
    }
    float gm = -INFINITY, gl = 0.f;
    float go[4] = {0.f, 0.f, 0.f, 0.f};
    for (int slot = groups; slot < nslots; ++slot) {
        const size_t base = ((size_t)slot * W + node) * H + head;
        cm = PM[base];
        cl = PL[base];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int i = lane + e * 32;
            if (i < D) co[e] = PO[base * D + i];
        }
        fold(gm, gl, go, cm, cl, co);
        if ((slot - groups + 1) % GR == 0 || slot == nslots - 1) {
            fold(m, l, o, gm, gl, go);
            gm = -INFINITY;
            gl = 0.f;
            go[0] = go[1] = go[2] = go[3] = 0.f;
        }
    }
    const float inv = 1.f / l;
    bf16* dst = OUT + ((size_t)node * H + head) * D;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int i = lane + e * 32;
        if (i < D) dst[i] = __float2bfloat16(o[e] * inv);
    }
}

}  // namespace tf_q27
