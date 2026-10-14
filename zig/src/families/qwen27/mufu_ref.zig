//! Host copy of sm_86 sqrt.approx.f32 and rcp.approx.f32: IEEE on the reduced mantissa, plus a one-ulp table.

const std = @import("std");

const nmant: usize = 1 << 23;
const rcp_gz = @embedFile("mufu_rcp.gz");
const sqrt_even_gz = @embedFile("mufu_sqrt_even.gz");
const sqrt_odd_gz = @embedFile("mufu_sqrt_odd.gz");
const ex2_corr_gz = @embedFile("mufu_ex2_corr.gz");
const ex2_neg_gz = @embedFile("mufu_ex2_neg.gz");
const rsq_even_gz = @embedFile("mufu_rsq_even.gz");
const rsq_odd_gz = @embedFile("mufu_rsq_odd.gz");

var rcp_fix: [nmant]i8 = undefined;
var sqrt_even: [nmant]i8 = undefined;
var sqrt_odd: [nmant]i8 = undefined;
var ex2_corr: [nmant]i8 = undefined;
var ex2_neg: [nmant]i8 = undefined;
var rsq_even: [nmant]i8 = undefined;
var rsq_odd: [nmant]i8 = undefined;
var loaded: bool = false;

fn inflate(comptime src: []const u8, dst: []u8) void {
    var in: std.Io.Reader = .fixed(src);
    var dec: std.compress.flate.Decompress = .init(&in, .gzip, &.{});
    var out: std.Io.Writer = .fixed(dst);
    const got = dec.reader.streamRemaining(&out) catch unreachable;
    if (got != dst.len) unreachable;
}

fn ensure() void {
    if (loaded) return;
    inflate(rcp_gz, std.mem.asBytes(&rcp_fix));
    inflate(sqrt_even_gz, std.mem.asBytes(&sqrt_even));
    inflate(sqrt_odd_gz, std.mem.asBytes(&sqrt_odd));
    inflate(ex2_corr_gz, std.mem.asBytes(&ex2_corr));
    inflate(ex2_neg_gz, std.mem.asBytes(&ex2_neg));
    inflate(rsq_even_gz, std.mem.asBytes(&rsq_even));
    inflate(rsq_odd_gz, std.mem.asBytes(&rsq_odd));
    loaded = true;
}

fn correct(ieee: f32, corr: i8) u32 {
    const bits: u32 = @bitCast(ieee);
    return @intCast(@as(i32, @bitCast(bits)) + @as(i32, corr));
}

fn withExp(bits: u32, biased: i32) f32 {
    const sign = bits & 0x80000000;
    if (biased <= 0) return @bitCast(sign);
    if (biased >= 255) return @bitCast(sign | @as(u32, 0x7f800000));
    const out: u32 = (bits & 0x807fffff) | (@as(u32, @intCast(biased)) << 23);
    return @bitCast(out);
}

pub fn rcpApprox(x: f32) f32 {
    @setFloatMode(.strict);
    ensure();
    const u: u32 = @bitCast(x);
    const exp: i32 = @as(i32, @intCast((u >> 23) & 0xff)) - 127;
    const mant: u32 = u & 0x7fffff;
    const reduced: f32 = @bitCast(@as(u32, 0x3f800000) | mant);
    const bits = correct(1.0 / reduced, rcp_fix[mant]);
    const re: i32 = @as(i32, @intCast((bits >> 23) & 0xff)) - exp;
    return withExp(bits, re);
}

pub fn sqrtApprox(x: f32) f32 {
    @setFloatMode(.strict);
    ensure();
    const u: u32 = @bitCast(x);
    const exp: i32 = @as(i32, @intCast((u >> 23) & 0xff)) - 127;
    const mant: u32 = u & 0x7fffff;
    const even = (exp & 1) == 0;
    const reduced: f32 = @bitCast((if (even) @as(u32, 0x3f800000) else @as(u32, 0x40000000)) | mant);
    const out_exp: i32 = if (even) @divTrunc(exp, 2) else @divTrunc(exp - 1, 2);
    const corr: i8 = if (even) sqrt_even[mant] else sqrt_odd[mant];
    const bits = correct(@sqrt(reduced), corr);
    const se: i32 = @intCast((bits >> 23) & 0xff);
    return withExp(bits, se + out_exp);
}

pub fn rsqrtApprox(z: f32) f32 {
    return rcpApprox(sqrtApprox(z));
}

/// 1/sqrt(ss * (1/5120) + eps), with the captured MUFU reciprocal square root. ss is the 5120-wide sum of squares.
pub fn invRms5120(ss: f32, eps: f32) f32 {
    @setFloatMode(.strict);
    const scale: f32 = @bitCast(@as(u32, 0x394ccccd));
    return rsqrtApprox(@mulAdd(f32, ss, scale, eps));
}

fn scaleBits(bits: u32, n: i32) u32 {
    const e: i32 = @as(i32, @intCast((bits >> 23) & 0xff)) + n;
    if (e <= 0) return 0;
    if (e >= 255) return 0x7f800000;
    return (bits & 0x807fffff) | (@as(u32, @intCast(e)) << 23);
}

fn ex2Reduced(x: f32) f32 {
    @setFloatMode(.strict);
    ensure();
    if (x >= 128) return @bitCast(@as(u32, 0x7f800000));
    const n: f32 = @floor(x);
    const f: f32 = x - n;
    const prod: f32 = f * 8388608.0;
    var i: i32 = @intFromFloat(@trunc(prod));
    if (i < 0) i = 0;
    if (i > 0x7fffff) i = 0x7fffff;
    const idx: usize = @intCast(i);
    const fq: f32 = @as(f32, @floatFromInt(i)) / 8388608.0;
    const base_f: f32 = @floatCast(@exp2(@as(f64, fq)));
    const bits_i: i32 = @as(i32, @bitCast(@as(u32, @bitCast(base_f)))) + @as(i32, ex2_corr[idx]);
    var scaled: u32 = scaleBits(@bitCast(bits_i), @intFromFloat(n));
    // neg[] is one ulp low on a half-bin tie and on an exact fraction in (i, i+0.5).
    const half = prod == @as(f32, @floatFromInt(i)) + 0.5;
    const over = (@as(f64, x) - @as(f64, n)) * 8388608.0 - @as(f64, @floatFromInt(i));
    if (n < 0 and !(half or (over > 0.0 and over < 0.5))) {
        scaled = @bitCast(@as(i32, @bitCast(scaled)) + @as(i32, ex2_neg[idx]));
    }
    return @bitCast(scaled);
}

pub fn ex2Approx(x: f32) f32 {
    @setFloatMode(.strict);
    if (std.math.isNan(x)) return x;
    // sm_86 ex2.approx flushes every magnitude below 2^-23 to 1.
    const tiny: f32 = @bitCast(@as(u32, 0x34000000));
    if (x > -tiny and x < tiny) return 1;
    if (x < -126) {
        const t = x * 0.5;
        if (!(t >= -126)) return 0;
        const r = ex2Reduced(t);
        return r * r;
    }
    return ex2Reduced(x);
}

pub fn expApprox(x: f32) f32 {
    @setFloatMode(.strict);
    const log2e: f32 = @bitCast(@as(u32, 0x3fb8aa3b));
    return ex2Approx(x * log2e);
}

pub fn logApprox(x: f32) f32 {
    @setFloatMode(.strict);
    const xi: u32 = @bitCast(x);
    const exp_bits: i32 = (@as(i32, @bitCast(xi)) - 0x3f2aaaab) & @as(i32, @bitCast(@as(u32, 0xff800000)));
    const f: f32 = @as(f32, @bitCast(xi -% @as(u32, @bitCast(exp_bits)))) - 1;
    var p: f32 = @bitCast(@as(u32, 0x3e055027));
    p = @mulAdd(f32, f, -p, @bitCast(@as(u32, 0x3e1039f6)));
    p = @mulAdd(f32, f, p, @bitCast(@as(u32, 0xbdf8cdcc)));
    p = @mulAdd(f32, f, p, @bitCast(@as(u32, 0x3e0f2955)));
    p = @mulAdd(f32, f, p, @bitCast(@as(u32, 0xbe2ad8b9)));
    p = @mulAdd(f32, f, p, @bitCast(@as(u32, 0x3e4ced0b)));
    p = @mulAdd(f32, f, p, @bitCast(@as(u32, 0xbe7fff22)));
    p = @mulAdd(f32, f, p, @bitCast(@as(u32, 0x3eaaaa78)));
    p = @mulAdd(f32, f, p, -0.5);
    var ln = @mulAdd(f32, f, f * p, f);
    const e = @mulAdd(f32, @floatFromInt(exp_bits), @bitCast(@as(u32, 0x34000000)), 0);
    ln = @mulAdd(f32, e, @bitCast(@as(u32, 0x3f317218)), ln);
    if (xi >= 0x7f800000) ln = @mulAdd(f32, x, @bitCast(@as(u32, 0x7f800000)), @bitCast(@as(u32, 0x7f800000)));
    return ln;
}

pub fn sigmoidApprox(x: f32) f32 {
    @setFloatMode(.strict);
    var d = expApprox(-x) + 1;
    var s: f32 = 1;
    if (d > @as(f32, @bitCast(@as(u32, 0x7e800000)))) {
        d = d * 0.25;
        s = 0.25;
    }
    return rcpApprox(d) * s;
}

pub fn softplusApprox(a: f32) f32 {
    @setFloatMode(.strict);
    if (a > 20) return a;
    return logApprox(1 + expApprox(a));
}

pub fn invRmsApprox(ss: f32, n: usize, eps: f32) f32 {
    @setFloatMode(.strict);
    const inv_n = 1.0 / @as(f32, @floatFromInt(n));
    return rsqrtApprox(@mulAdd(f32, ss, inv_n, eps));
}

pub fn rsqApprox(x: f32) f32 {
    @setFloatMode(.strict);
    ensure();
    if (std.math.isNan(x) or x < 0) return std.math.nan(f32);
    if (x == 0) return std.math.inf(f32);
    const u: u32 = @bitCast(x);
    const exp: i32 = @as(i32, @intCast((u >> 23) & 0xff)) - 127;
    const mant: u32 = u & 0x7fffff;
    const even = (exp & 1) == 0;
    const reduced: f32 = @bitCast((if (even) @as(u32, 0x3f800000) else @as(u32, 0x40000000)) | mant);
    const out_exp: i32 = if (even) @divTrunc(exp, 2) else @divTrunc(exp - 1, 2);
    const corr: i8 = if (even) rsq_even[mant] else rsq_odd[mant];
    const base: f32 = @floatCast(1.0 / @sqrt(@as(f64, reduced)));
    const bits = correct(base, corr);
    const biased: i32 = @as(i32, @intCast((bits >> 23) & 0xff)) - out_exp;
    if (biased <= 0) return 0;
    if (biased >= 255) return @bitCast(@as(u32, 0x7f800000));
    return withExp(bits, biased);
}

fn fbits(x: f32) u32 {
    return @bitCast(x);
}

test "mufu tables inflate and a reciprocal of 1.5 stays near two thirds" {
    ensure();
    var nonzero: usize = 0;
    for (rcp_fix) |c| {
        try std.testing.expect(c >= -1 and c <= 1);
        if (c != 0) nonzero += 1;
    }
    try std.testing.expect(nonzero > 1000);
    const y = rcpApprox(1.5);
    try std.testing.expect(y > 0.6 and y < 0.7);
    try std.testing.expectEqual(@as(u32, 0x3fb504f3), fbits(@floatCast(@exp2(@as(f64, 0.5)))));
    try std.testing.expectEqual(@as(u32, 0x3fb504f3), fbits(ex2Approx(0.5)));
    try std.testing.expectEqual(@as(u32, 0x3f3504f2), fbits(ex2Approx(-0.5)));
    try std.testing.expectEqual(@as(u32, 0x00080000), fbits(ex2Approx(-130)));
    try std.testing.expectEqual(@as(u32, 0x3f317218), fbits(logApprox(2)));
    try std.testing.expectEqual(@as(u32, 0xbf317218), fbits(logApprox(0.5)));
    try std.testing.expectEqual(@as(u32, 0x3ecf9920), fbits(logApprox(1.5)));
    try std.testing.expectEqual(@as(u32, 0), fbits(logApprox(1)));
    try std.testing.expectEqual(@as(u32, 0x3f5105eb), fbits(rsqApprox(1.5)));
}
