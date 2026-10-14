// Tree attention. One warp a query head. Keys fold in 64-key tiles. OFF is int64 element offsets from base.

#include <cuda_bf16.h>

#include "glue_math.cuh"
#include "q27_math.cuh"

using bf16 = __nv_bfloat16;

constexpr int TILE = 64;
constexpr int CHUNK = 512;
constexpr int CHUNK_TILES = CHUNK / TILE;
constexpr int GR = 4;
constexpr int MAXD = 128;

extern __shared__ __align__(16) char q27_tree_smem[];

struct Acc {
    float m, l, o[8];
};

__device__ Acc acc_new() {
    Acc a;
    a.m = tf_q27m::neg_inf();
    a.l = 0.f;
    for (int e = 0; e < 8; ++e) a.o[e] = 0.f;
    return a;
}

// Sequential f32 dots; Triton's tl.dot is HMMA.16816.F32.BF16 (docs/q27-sm86-hmma.md).
__device__ void tile_rows(const bf16* q, const bf16* kbuf, const bf16* vbuf, Acc& a, const int* valid, float scale, int d) {
    const int lane = threadIdx.x & 31;
    const bool v1 = valid[lane] != 0, v2 = valid[32 + lane] != 0;
    float s1 = tf_q27m::neg_inf(), s2 = tf_q27m::neg_inf();
    if (v1 || v2) {
        float d1 = 0.f, d2 = 0.f;
        for (int i = 0; i < d; ++i) {
            const float qv = tf_glue::bf(q[i]);
            if (v1) d1 = __fmaf_rn(qv, tf_glue::bf(kbuf[lane * d + i]), d1);
            if (v2) d2 = __fmaf_rn(qv, tf_glue::bf(kbuf[(32 + lane) * d + i]), d2);
        }
        if (v1) s1 = __fmul_rn(d1, scale);
        if (v2) s2 = __fmul_rn(d2, scale);
    }
    float tile_m = fmaxf(s1, s2);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) tile_m = fmaxf(tile_m, __shfl_xor_sync(0xffffffffu, tile_m, off));
    const bool active = tile_m != tf_q27m::neg_inf();
    float next_m = a.m;
    if (active && tile_m > next_m) next_m = tile_m;
    const float alpha = !active ? 1.f : (a.m == tf_q27m::neg_inf() ? 0.f : tf_q27m::mufu_exp(a.m - next_m));
    const float p1 = (v1 && active) ? tf_q27m::mufu_exp(s1 - next_m) : 0.f;
    const float p2 = (v2 && active) ? tf_q27m::mufu_exp(s2 - next_m) : 0.f;
    const float psum = tf_glue::warp_sum(__fadd_rn(p1, p2));
    a.l = __fadd_rn(__fmul_rn(a.l, alpha), psum);
#pragma unroll
    for (int e = 0; e < 8; ++e) a.o[e] = __fmul_rn(a.o[e], alpha);
    for (int j = 0; j < 32; ++j) {
        const float pj = __shfl_sync(0xffffffffu, p1, j);
        if (pj != 0.f) {
            const float pb = tf_glue::rbf(pj);
            const bf16* vr = vbuf + j * d;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int i = lane + e * 32;
                if (i < d) a.o[e] = __fmaf_rn(pb, tf_glue::bf(vr[i]), a.o[e]);
            }
        }
    }
    for (int j = 0; j < 32; ++j) {
        const float pj = __shfl_sync(0xffffffffu, p2, j);
        if (pj != 0.f) {
            const float pb = tf_glue::rbf(pj);
            const bf16* vr = vbuf + (32 + j) * d;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int i = lane + e * 32;
                if (i < d) a.o[e] = __fmaf_rn(pb, tf_glue::bf(vr[i]), a.o[e]);
            }
        }
    }
    if (active) a.m = next_m;
}

__device__ void fold(Acc& a, float cm, float cl, const float co[8]) {
    const bool on = cl > 0.f;
    float next_m = a.m;
    if (on && cm > next_m) next_m = cm;
    const float fa = !on ? 1.f : (a.m == tf_q27m::neg_inf() ? 0.f : tf_q27m::mufu_exp(a.m - next_m));
    const float fb = on ? tf_q27m::mufu_exp(cm - next_m) : 0.f;
    a.l = __fadd_rn(__fmul_rn(a.l, fa), __fmul_rn(cl, fb));
#pragma unroll
    for (int e = 0; e < 8; ++e) a.o[e] = __fadd_rn(__fmul_rn(a.o[e], fa), __fmul_rn(co[e], fb));
    a.m = next_m;
}

__device__ void load_slot(const float* po, const float* pm, const float* pl, int slot, int W, int node, int H, int head,
                           int d, float& cm, float& cl, float co[8]) {
    const int lane = threadIdx.x & 31;
    const size_t base = ((size_t)slot * W + node) * H + head;
    cm = pm[base];
    cl = pl[base];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int i = lane + e * 32;
        co[e] = i < d ? po[base * d + i] : 0.f;
    }
}

__device__ void store_partial(Acc a, float* po, float* pm, float* pl, size_t base, int d) {
    const int lane = threadIdx.x & 31;
    if (lane == 0) {
        pm[base] = a.m;
        pl[base] = a.l;
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int i = lane + e * 32;
        if (i < d) po[base * d + i] = a.o[e];
    }
}

extern "C" __global__ void tf_q27_paths(const int* __restrict__ parents, int* __restrict__ paths, int* __restrict__ depths, int w) {
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= w) return;
    int cur = node, depth = 0;
    while (cur >= 0 && depth < MAXD) {
        depth += 1;
        cur = parents[cur];
    }
    depths[node] = depth;
    cur = node;
    for (int slot = depth - 1; slot >= 0; --slot) {
        paths[node * MAXD + slot] = cur;
        cur = parents[cur];
    }
}

extern "C" __global__ void tf_q27_shared(const bf16* __restrict__ Q, const bf16* __restrict__ base, const long long* __restrict__ off,
                                         const int* __restrict__ stream, const int* __restrict__ items, float* __restrict__ po,
                                         float* __restrict__ pm, float* __restrict__ pl, int W, int H, int HK, int G, int D,
                                         float scale) {
    bf16* kbuf = reinterpret_cast<bf16*>(q27_tree_smem);
    bf16* vbuf = kbuf + TILE * D;
    int* valid = reinterpret_cast<int*>(vbuf + TILE * D);
    const int item = blockIdx.x / HK, hk = blockIdx.x - item * HK;
    const int s = items[item * 3], first = items[item * 3 + 1], code = items[item * 3 + 2];
    const int start = stream[s * 4], rows = stream[s * 4 + 1], p = stream[s * 4 + 2], nslots = stream[s * 4 + 3];
    const int nch = (p + rows + CHUNK - 1) / CHUNK;
    const int groups = (nch - nslots) / (GR - 1);
    const bool whole = code >= 0;
    const int chunk = whole ? code * GR : groups * GR - 1 - code;
    const int n = whole ? GR : 1;
    const int slot = whole ? code : chunk - groups * (GR - 1);
    if ((chunk + n) * CHUNK > p || D > 256) return;
    const int warp = threadIdx.x >> 5;
    const int rr = first + warp;
    const bool live = warp < 16 && rr < rows * G;
    const int node = start + (live ? rr / G : 0);
    const int head = hk * G + (live ? rr % G : 0);
    const bf16* q = Q + ((size_t)node * H + head) * D;
    const bf16* KC = base + off[s * 2];
    const bf16* VC = base + off[s * 2 + 1];
    Acc ga = acc_new();
    for (int c = 0; c < n; ++c) {
        Acc ca = acc_new();
        for (int t = 0; t < CHUNK_TILES; ++t) {
            const int key0 = (chunk + c) * CHUNK + t * TILE;
            for (int idx = threadIdx.x; idx < TILE * D; idx += blockDim.x) {
                const int j = idx / D, i = idx - j * D;
                const bool on = key0 + j < p;
                const size_t at = ((size_t)(key0 + j) * HK + hk) * D + i;
                kbuf[idx] = on ? KC[at] : __float2bfloat16_rn(0.f);
                vbuf[idx] = on ? VC[at] : __float2bfloat16_rn(0.f);
            }
            if (threadIdx.x < TILE) valid[threadIdx.x] = key0 + threadIdx.x < p;
            __syncthreads();
            if (live) tile_rows(q, kbuf, vbuf, ca, valid, scale, D);
            __syncthreads();
        }
        if (whole) fold(ga, ca.m, ca.l, ca.o);
        else ga = ca;
    }
    if (!live) return;
    store_partial(ga, po, pm, pl, ((size_t)slot * W + node) * H + head, D);
}

extern "C" __global__ void tf_q27_tail(const bf16* __restrict__ Q, const bf16* __restrict__ KN, const bf16* __restrict__ VN,
                                       const bf16* __restrict__ base, const long long* __restrict__ off, const int* __restrict__ stream,
                                       const int* __restrict__ rows_of, const int* __restrict__ paths, const int* __restrict__ depths,
                                       float* __restrict__ po, float* __restrict__ pm, float* __restrict__ pl, int W, int H, int HK,
                                       int G, int D, float scale) {
    bf16* kbuf = reinterpret_cast<bf16*>(q27_tree_smem);
    bf16* vbuf = kbuf + TILE * D;
    int* valid = reinterpret_cast<int*>(vbuf + TILE * D);
    const int node = blockIdx.x, hk = blockIdx.y;
    const int s = rows_of[node];
    const int p = stream[s * 4 + 2], rows = stream[s * 4 + 1], nslots = stream[s * 4 + 3];
    const int nch = (p + rows + CHUNK - 1) / CHUNK;
    const int groups = (nch - nslots) / (GR - 1);
    const int chunk = p / CHUNK + blockIdx.z;
    if (chunk >= nch || D > 256) return;
    const int warp = threadIdx.x >> 5;
    const bool live = warp < G;
    const int head = hk * G + (live ? warp : 0);
    const bf16* q = Q + ((size_t)node * H + head) * D;
    const bf16* KC = base + off[s * 2];
    const bf16* VC = base + off[s * 2 + 1];
    const int depth = depths[node];
    Acc a = acc_new();
    for (int t = 0; t < CHUNK_TILES; ++t) {
        for (int idx = threadIdx.x; idx < TILE * D; idx += blockDim.x) {
            const int j = idx / D, i = idx - j * D;
            const int logical = chunk * CHUNK + t * TILE + j;
            const bool committed = logical < p;
            const int path_slot = logical - p;
            const bool on_path = path_slot >= 0 && path_slot < depth;
            const int kn = on_path ? paths[node * MAXD + path_slot] : 0;
            bf16 kv = __float2bfloat16_rn(0.f), vv = kv;
            if (committed) {
                kv = KC[((size_t)logical * HK + hk) * D + i];
                vv = VC[((size_t)logical * HK + hk) * D + i];
            } else if (on_path) {
                kv = KN[((size_t)kn * HK + hk) * D + i];
                vv = VN[((size_t)kn * HK + hk) * D + i];
            }
            kbuf[idx] = kv;
            vbuf[idx] = vv;
        }
        if (threadIdx.x < TILE) {
            const int logical = chunk * CHUNK + t * TILE + threadIdx.x;
            const int path_slot = logical - p;
            valid[threadIdx.x] = logical < p || (path_slot >= 0 && path_slot < depth);
        }
        __syncthreads();
        if (live) tile_rows(q, kbuf, vbuf, a, valid, scale, D);
        __syncthreads();
    }
    if (!live) return;
    const int slot = chunk - groups * (GR - 1);
    store_partial(a, po, pm, pl, ((size_t)slot * W + node) * H + head, D);
}

extern "C" __global__ void tf_q27_merge(const float* __restrict__ po, const float* __restrict__ pm, const float* __restrict__ pl,
                                        bf16* __restrict__ out, const int* __restrict__ stream, const int* __restrict__ rows_of, int W,
                                        int H, int HK, int G, int D) {
    const int node = blockIdx.x, hk = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp >= G) return;
    const int head = hk * G + warp;
    const int s = rows_of[node];
    const int rows = stream[s * 4 + 1], p = stream[s * 4 + 2], nslots = stream[s * 4 + 3];
    const int nch = (p + rows + CHUNK - 1) / CHUNK;
    const int groups = (nch - nslots) / (GR - 1);
    Acc a = acc_new();
    float cm, cl, co[8];
    for (int slot = 0; slot < groups; ++slot) {
        load_slot(po, pm, pl, slot, W, node, H, head, D, cm, cl, co);
        fold(a, cm, cl, co);
    }
    Acc gacc = acc_new();
    for (int slot = groups; slot < nslots; ++slot) {
        load_slot(po, pm, pl, slot, W, node, H, head, D, cm, cl, co);
        fold(gacc, cm, cl, co);
        if ((slot - groups + 1) % GR == 0 || slot == nslots - 1) {
            fold(a, gacc.m, gacc.l, gacc.o);
            gacc = acc_new();
        }
    }
    const float inv = tf_q27m::mufu_rcp(a.l);
    bf16* dst = out + ((size_t)node * H + head) * D;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int i = lane + e * 32;
        if (i < D) dst[i] = __float2bfloat16_rn(__fmul_rn(a.o[e], inv));
    }
}
