//! Host copy of q27_math.cuh: one Cody-Waite sine and cosine, the RMS inverse, the attention scale.

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
