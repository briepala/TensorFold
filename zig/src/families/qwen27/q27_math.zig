//! Host copy of q27_math.cuh plus libdevice __nv_sinf/__nv_cosf (CUDA Math API / libdevice.10.bc, PTX cvt.rni.s32.f32).

const std = @import("std");
const gm = @import("nemotron").glue_math;

pub fn negInf() f32 {
    return -std.math.inf(f32);
}

pub fn invRms(ss: f32, n: usize, eps: f32) f32 {
    @setFloatMode(.strict);
    const nf: f32 = @floatFromInt(n);
    return 1.0 / @sqrt((ss / nf) + eps);
}

/// 1/sqrt(dim) in fp64, then the nearest fp32. Tree attention and the drafter's block share this value.
pub fn attnScale(dim: usize) f32 {
    return @floatCast(std.math.pow(f64, @as(f64, @floatFromInt(dim)), -0.5));
}

pub fn abF32(bits: u16, half: bool) f32 {
    if (half) return @floatCast(@as(f16, @bitCast(bits)));
    return gm.bf16ToF32(bits);
}

pub fn sincos(x: f32, sn: *f32, cs: *f32) void {
    @setFloatMode(.strict);
    const j = gm.fma(x, 0x1.45f306p-1, 0x1.8p23) - 0x1.8p23;
    var f = gm.fma(j, -0x1.921fb6p0, x);
    f = gm.fma(j, -0x1.777a5cp-25, f);
    const f2 = f * f;
    var sp: f32 = -0x1.a01a02p-13;
    sp = gm.fma(sp, f2, 0x1.111112p-7);
    sp = gm.fma(sp, f2, -0x1.555556p-3);
    sp = gm.fma(sp, f2, 1.0);
    const s = f * sp;
    var cp: f32 = 0x1.a01a02p-16;
    cp = gm.fma(cp, f2, -0x1.6c16c2p-10);
    cp = gm.fma(cp, f2, 0x1.555556p-5);
    cp = gm.fma(cp, f2, -0.5);
    const c = gm.fma(cp, f2, 1.0);
    switch (@as(i32, @intFromFloat(j)) & 3) {
        0 => {
            sn.* = s;
            cs.* = c;
        },
        1 => {
            sn.* = c;
            cs.* = -s;
        },
        2 => {
            sn.* = -s;
            cs.* = -c;
        },
        else => {
            sn.* = -c;
            cs.* = s;
        },
    }
}

fn fbits(u: u32) f32 {
    return @bitCast(u);
}

/// cvt.rni.s32.f32: round to nearest, ties to even. Zig @round is ties away from zero.
fn cvtRniS32(x: f32) i32 {
    @setFloatMode(.strict);
    const mag: f32 = 0x1.0p23;
    if (x >= mag or x <= -mag) return @intFromFloat(x);
    const magic: f32 = if (x < 0) -mag else mag;
    return @intFromFloat((x + magic) - magic);
}

fn nvPoly(a: f32, for_cos: bool) f32 {
    @setFloatMode(.strict);
    if (@abs(a) >= 1.056150e5) {
        var sn: f32 = 0;
        var cs: f32 = 0;
        sincos(a, &sn, &cs);
        return if (for_cos) cs else sn;
    }
    const j = cvtRniS32(a * fbits(0x3f22f983));
    const jf: f32 = @floatFromInt(j);
    var f = gm.fma(jf, fbits(0xbfc90fda), a);
    f = gm.fma(jf, fbits(0xb3a22168), f);
    f = gm.fma(jf, fbits(0xa7c234c5), f);
    const q = if (for_cos) j + 1 else j;
    const s = f * f;
    const odd = q & 1 != 0;
    const b: f32 = if (odd) 1.0 else f;
    const sb = gm.fma(s, b, 0);
    var p: f32 = if (odd) gm.fma(fbits(0x37cbac00), s, fbits(0xbab607ed)) else fbits(0xb94d4153);
    p = gm.fma(p, s, if (odd) fbits(0x3d2aaabb) else fbits(0x3c0885e4));
    p = gm.fma(p, s, if (odd) fbits(0xbeffffff) else fbits(0xbe2aaaa8));
    var r = gm.fma(p, sb, b);
    if (q & 2 != 0) r = gm.fma(r, -1.0, 0);
    return r;
}

pub fn nvSinf(a: f32) f32 {
    return nvPoly(a, false);
}

pub fn nvCosf(a: f32) f32 {
    return nvPoly(a, true);
}

test "libdevice fast-path sin and cos at 0" {
    try std.testing.expectEqual(@as(f32, 0), nvSinf(0));
    try std.testing.expectEqual(@as(f32, 1), nvCosf(0));
}
