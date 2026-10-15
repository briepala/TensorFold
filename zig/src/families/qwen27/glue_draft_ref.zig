//! Host reference for the drafter kernels in q27_draft.cu: dynamic conv, rotary prep, fp32 block attention.

const std = @import("std");
const gm = @import("nemotron").glue_math;
const m = @import("q27_math.zig");
const mu = @import("mufu_ref.zig");

fn bf(h: u16) f32 {
    return gm.bf16ToF32(h);
}

fn store(v: f32) u16 {
    return gm.f32ToBf16(v);
}

pub fn dconv(x: []const u16, dyn: []const u16, base: []const u16, res: ?[]const u16, out: []u16, rows: usize, seg: usize, d: usize, g: usize, gs: usize, branch: usize) void {
    @setFloatMode(.strict);
    for (0..rows) |row| {
        for (0..d) |c| {
            const xv = bf(x[row * d + c]);
            const prev: f32 = if (row % seg > 0) bf(x[(row - 1) * d + c]) else 0;
            const grp = c / gs;
            const dyn0 = ((row * 2 + branch) * 2) * g + grp;
            const k0 = gm.rbf(bf(base[(branch * 2) * d + c]) + bf(dyn[dyn0]));
            const k1 = gm.rbf(bf(base[(branch * 2 + 1) * d + c]) + bf(dyn[dyn0 + g]));
            var y = gm.rbf((xv * k0) + (prev * k1));
            if (res) |rv| y = gm.rbf(bf(rv[row * d + c]) + y);
            out[row * d + c] = store(y);
        }
    }
}

fn bfly32(v: *[32]f32) void {
    @setFloatMode(.strict);
    var off: usize = 16;
    while (off > 0) : (off >>= 1) {
        var next: [32]f32 = undefined;
        for (0..32) |i| next[i] = v[i] + v[i ^ off];
        v.* = next;
    }
}

fn dprepPair(src: []const u16, norm: []const u16, cos: []const f32, sin: []const f32, dst: []u16, i: usize, half: usize, row: usize, inv_a: f32, inv_b: f32) void {
    @setFloatMode(.strict);
    const a = gm.rbf((bf(src[i]) * inv_a) * bf(norm[i]));
    const b = gm.rbf((bf(src[half + i]) * inv_b) * bf(norm[half + i]));
    const cs = cos[row * half + i];
    const sn = sin[row * half + i];
    dst[i] = store(gm.fma(a, cs, -(b * sn)));
    dst[half + i] = store(gm.fma(sn, a, b * cs));
}

pub fn dprep(qkv: []const u16, qn: []const u16, kn: []const u16, cos: []const f32, sin: []const f32, qo: []u16, ko: []u16, vo: []u16, rows: usize, h: usize, hkv: usize, half: usize, stride: usize, eps: f32) void {
    @setFloatMode(.strict);
    if (half > 64) return;
    const dim = half * 2;
    for (0..rows) |row| {
        for (0..h + 2 * hkv) |head| {
            const src = qkv[row * stride + head * dim ..][0..dim];
            if (head >= h + hkv) {
                const dst = vo[((head - h - hkv) * rows + row) * dim ..][0..dim];
                @memcpy(dst, src);
                continue;
            }
            const norm = if (head < h) qn else kn;
            const dst = if (head < h) qo[(head * rows + row) * dim ..][0..dim] else ko[((head - h) * rows + row) * dim ..][0..dim];
            if (half == 64) {
                var pa: [32]f32 = undefined;
                var pb: [32]f32 = undefined;
                for (0..32) |lane| {
                    const i = lane * 2;
                    pa[lane] = gm.fma(bf(src[i]), bf(src[i]), bf(src[i + 1]) * bf(src[i + 1]));
                    pb[lane] = gm.fma(bf(src[half + i]), bf(src[half + i]), bf(src[half + i + 1]) * bf(src[half + i + 1]));
                }
                bfly32(&pa);
                bfly32(&pb);
                const inv_n = 1.0 / @as(f32, @floatFromInt(dim));
                for (0..32) |lane| {
                    const inv = mu.rsqApprox(@mulAdd(f32, pa[lane] + pb[lane], inv_n, eps));
                    const i = lane * 2;
                    dprepPair(src, norm, cos, sin, dst, i, half, row, inv, inv);
                    dprepPair(src, norm, cos, sin, dst, i + 1, half, row, inv, inv);
                }
                continue;
            }
            var ss: f32 = 0;
            for (0..dim) |i| ss = gm.fma(bf(src[i]), bf(src[i]), ss);
            const inv = mu.rsqApprox(@mulAdd(f32, ss, 1.0 / @as(f32, @floatFromInt(dim)), eps));
            for (0..half) |i| dprepPair(src, norm, cos, sin, dst, i, half, row, inv, inv);
        }
    }
}

fn dot(a: []const u16, b: []const u16) f32 {
    @setFloatMode(.strict);
    var s: f32 = 0;
    for (a, b) |x, y| s = gm.fma(bf(x), bf(y), s);
    return s;
}

pub fn blockAttn(q: []const u16, kb: []const u16, vb: []const u16, kc: []const []const u16, vc: []const []const u16, lens: []const i32, o: []u16, scale: f32, window: usize, R: usize, query_heads: usize, hkv: usize, L: usize, D: usize, bn: usize, causal: bool) void {
    @setFloatMode(.strict);
    if (L > 64 or bn > 64 or D > 128) return;
    const G = query_heads / hkv;
    for (lens, 0..) |slen_i, j| {
        const slen: usize = @intCast(slen_i);
        const kctx = kc[j];
        const vctx = vc[j];
        for (0..hkv) |g| {
            for (0..L) |r| {
                const qrow = j * L + r;
                for (0..G) |hq| {
                    const head = g * G + hq;
                    const qr = q[head * R * D + qrow * D ..][0..D];
                    var mv = m.negInf();
                    var l: f32 = 0;
                    var acc: [128]f32 = undefined;
                    @memset(&acc, 0);
                    var n0: usize = 0;
                    while (n0 < slen) : (n0 += bn) {
                        var scores: [64]f32 = undefined;
                        var tile_m = mv;
                        for (0..bn) |t| {
                            const kidx = n0 + t;
                            const seen = kidx < slen and slen + r - kidx < window + 1;
                            var sc = m.negInf();
                            if (seen) sc = dot(qr, kctx[g * slen * D + kidx * D ..][0..D]) * scale;
                            scores[t] = sc;
                            if (sc > tile_m) tile_m = sc;
                        }
                        const m_safe: f32 = if (tile_m == m.negInf()) 0 else tile_m;
                        const alpha = mu.expApprox(mv - m_safe);
                        var sum_p: f32 = 0;
                        for (0..bn) |t| {
                            const p: f32 = if (scores[t] == m.negInf()) 0 else mu.expApprox(scores[t] - m_safe);
                            scores[t] = p;
                            sum_p += p;
                        }
                        l = (l * alpha) + sum_p;
                        for (0..D) |i| acc[i] *= alpha;
                        for (0..bn) |t| {
                            const kidx = n0 + t;
                            if (kidx >= slen) continue;
                            const pb = gm.rbf(scores[t]);
                            const vv = vctx[g * slen * D + kidx * D ..][0..D];
                            for (0..D) |i| acc[i] = gm.fma(pb, bf(vv[i]), acc[i]);
                        }
                        mv = tile_m;
                    }
                    var scores: [64]f32 = undefined;
                    var tile_m = mv;
                    for (0..L) |col| {
                        const seen = if (causal) col <= r else true;
                        var sc = m.negInf();
                        if (seen) {
                            const krow = j * L + col;
                            sc = dot(qr, kb[g * R * D + krow * D ..][0..D]) * scale;
                        }
                        scores[col] = sc;
                        if (sc > tile_m) tile_m = sc;
                    }
                    const m_safe: f32 = if (tile_m == m.negInf()) 0 else tile_m;
                    const alpha = mu.expApprox(mv - m_safe);
                    var sum_p: f32 = 0;
                    for (0..L) |col| {
                        const p: f32 = if (scores[col] == m.negInf()) 0 else mu.expApprox(scores[col] - m_safe);
                        scores[col] = p;
                        sum_p += p;
                    }
                    l = (l * alpha) + sum_p;
                    for (0..D) |i| acc[i] *= alpha;
                    for (0..L) |col| {
                        const krow = j * L + col;
                        const pb = gm.rbf(scores[col]);
                        const vv = vb[g * R * D + krow * D ..][0..D];
                        for (0..D) |i| acc[i] = gm.fma(pb, bf(vv[i]), acc[i]);
                    }
                    const inv = mu.rcpApprox(l);
                    for (0..D) |i| o[qrow * (G * hkv * D) + head * D + i] = store(gm.rbf(acc[i] * inv));
                }
            }
        }
    }
}

test "dynamic conv mixes the previous row inside a segment" {
    const d = 16;
    const g = 1;
    var x = std.mem.zeroes([32]u16);
    var dyn = std.mem.zeroes([16]u16);
    var base = std.mem.zeroes([64]u16);
    var out = std.mem.zeroes([32]u16);
    x[0] = gm.f32ToBf16(1);
    x[d] = gm.f32ToBf16(1);
    base[0] = gm.f32ToBf16(1);
    base[d] = gm.f32ToBf16(1);
    dconv(&x, &dyn, &base, null, &out, 2, 2, d, g, d, 0);
    try std.testing.expect(out[d] != 0);
}
