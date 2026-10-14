// Host IEEE vs FTZ copies of libdevice __nv_sinf/__nv_cosf vs the device builtins. No device lambdas.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>

extern "C" __device__ float __nv_sinf(float);
extern "C" __device__ float __nv_cosf(float);

__global__ void k_nv(const float* a, float* s, float* c, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        s[i] = __nv_sinf(a[i]);
        c[i] = __nv_cosf(a[i]);
    }
}

static float fbits(uint32_t u) {
    float x;
    memcpy(&x, &u, 4);
    return x;
}

static uint32_t ubits(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    return u;
}

static float ftz(float x) {
    const uint32_t u = ubits(x);
    if ((u & 0x7f800000u) == 0 && (u & 0x7fffffu) != 0) return copysignf(0.f, x);
    return x;
}

static float fma_ieee(float a, float b, float c) { return fmaf(a, b, c); }
static float fma_ftz(float a, float b, float c) { return ftz(fmaf(ftz(a), ftz(b), ftz(c))); }

static int cvt_rni(float x) {
    const float mag = 0x1.0p23f;
    if (x >= mag || x <= -mag) return (int)x;
    const float magic = x < 0 ? -mag : mag;
    return (int)((x + magic) - magic);
}

static float nv_poly(float a, int for_cos, int use_ftz) {
    float (*fm)(float, float, float) = use_ftz ? fma_ftz : fma_ieee;
    if (fabsf(a) >= 1.056150e5f) return for_cos ? cosf(a) : sinf(a);
    const int j = cvt_rni(a * fbits(0x3f22f983u));
    const float jf = (float)j;
    float f = fm(jf, fbits(0xbfc90fdau), a);
    f = fm(jf, fbits(0xb3a22168u), f);
    f = fm(jf, fbits(0xa7c234c5u), f);
    const int q = for_cos ? j + 1 : j;
    const float s = use_ftz ? ftz(f * f) : f * f;
    const int odd = q & 1;
    const float b = odd ? 1.f : f;
    const float sb = fm(s, b, 0.f);
    float p = odd ? fm(fbits(0x37cbac00u), s, fbits(0xbab607edu)) : fbits(0xb94d4153u);
    p = fm(p, s, odd ? fbits(0x3d2aaabbu) : fbits(0x3c0885e4u));
    p = fm(p, s, odd ? fbits(0xbeffffffu) : fbits(0xbe2aaaa8u));
    float r = fm(p, sb, b);
    if (q & 2) r = fm(r, -1.f, 0.f);
    return r;
}

int main() {
    std::vector<float> ang;
    for (int row = 0; row < 64; ++row) {
        const int pr = (row * 17 + 3) % 500;
        for (int i = 0; i < 32; ++i) {
            const float inv = powf(1e7f, -((float)i) / 32.f);
            ang.push_back((float)pr * inv);
        }
    }
    ang.push_back(0.f);
    ang.push_back(-0.f);
    ang.push_back(1.f);
    const int n = (int)ang.size();
    float *d_a = nullptr, *d_s = nullptr, *d_c = nullptr;
    cudaMalloc(&d_a, n * 4);
    cudaMalloc(&d_s, n * 4);
    cudaMalloc(&d_c, n * 4);
    cudaMemcpy(d_a, ang.data(), n * 4, cudaMemcpyHostToDevice);
    k_nv<<<(n + 127) / 128, 128>>>(d_a, d_s, d_c, n);
    std::vector<float> ds(n), dc(n);
    cudaMemcpy(ds.data(), d_s, n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(dc.data(), d_c, n * 4, cudaMemcpyDeviceToHost);
    int mis_ieee_s = 0, mis_ieee_c = 0, mis_ftz_s = 0, mis_ftz_c = 0, shown = 0;
    for (int i = 0; i < n; ++i) {
        const float hs_i = nv_poly(ang[i], 0, 0), hc_i = nv_poly(ang[i], 1, 0);
        const float hs_f = nv_poly(ang[i], 0, 1), hc_f = nv_poly(ang[i], 1, 1);
        if (ubits(hs_i) != ubits(ds[i])) {
            mis_ieee_s++;
            if (shown++ < 8)
                printf("DIFF sin ieee i=%d ang=%08x host=%08x dev=%08x\n", i, ubits(ang[i]), ubits(hs_i), ubits(ds[i]));
        }
        if (ubits(hc_i) != ubits(dc[i])) mis_ieee_c++;
        if (ubits(hs_f) != ubits(ds[i])) mis_ftz_s++;
        if (ubits(hc_f) != ubits(dc[i])) mis_ftz_c++;
    }
    printf("SINPOLY n=%d ieee_sin=%d ieee_cos=%d ftz_sin=%d ftz_cos=%d\n", n, mis_ieee_s, mis_ieee_c, mis_ftz_s, mis_ftz_c);
    printf("SINPOLY_EXIT:%d\n", (mis_ieee_s || mis_ieee_c) && (mis_ftz_s || mis_ftz_c) ? 1 : 0);
    return 0;
}
