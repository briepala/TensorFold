//! Host reference for the 27B row, GDN prep and attention prep kernels. One thread's loop, IEEE ops, bf16 stores.

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

pub fn groupSums(x: []const u16, xs: []f32) void {
    @setFloatMode(.strict);
    for (0..xs.len) |g| {
        var part: [8]f32 = undefined;
        for (0..8) |lane| {
            var e: [8]f32 = undefined;
            for (0..8) |k| e[k] = bf(x[g * 64 + lane * 8 + k]);
            part[lane] = add8(&e);
        }
        bfly(8, &part, 4);
        xs[g] = part[0];
    }
}

fn pairSq(a: f32, b: f32) f32 {
    @setFloatMode(.strict);
    return gm.fma(b, b, a * a);
}

fn chunkHl(e: *const [8]f32) f32 {
    @setFloatMode(.strict);
    return (pairSq(e[4], e[5]) + pairSq(e[6], e[7])) + (pairSq(e[2], e[3]) + pairSq(e[0], e[1]));
}

fn chunkLh(e: *const [8]f32) f32 {
    @setFloatMode(.strict);
    return (pairSq(e[0], e[1]) + pairSq(e[2], e[3])) + (pairSq(e[4], e[5]) + pairSq(e[6], e[7]));
}

fn add8(e: *const [8]f32) f32 {
    @setFloatMode(.strict);
    return ((e[0] + e[1]) + (e[2] + e[3])) + ((e[4] + e[5]) + (e[6] + e[7]));
}

fn bfly(comptime n: usize, vals: *[n]f32, start: usize) void {
    @setFloatMode(.strict);
    var off = start;
    while (off > 0) : (off >>= 1) {
        var next: [n]f32 = undefined;
        for (0..n) |i| next[i] = vals[i] + vals[i ^ off];
        vals.* = next;
    }
}

fn octetSums(y: []const u16, base: usize, threads: usize, xs: []f32, at: usize) void {
    @setFloatMode(.strict);
    var part: [256]f32 = undefined;
    for (0..threads) |tid| {
        var e: [8]f32 = undefined;
        for (0..8) |k| e[k] = bf(y[base + tid * 8 + k]);
        part[tid] = add8(&e);
    }
    var g: usize = 0;
    while (g < threads) : (g += 8) {
        var p: [8]f32 = undefined;
        for (0..8) |k| p[k] = part[g + k];
        bfly(8, &p, 4);
        xs[at + (g >> 3)] = p[0];
    }
}

// Width 5120 follows the captured reduction: chunk trees, warp and block butterflies, MUFU inverse.
fn addRms5120(x: []const u16, r: ?[]const u16, w: []const u16, h: []u16, y: []u16, xs: []f32, rows: usize, eps: f32) void {
    @setFloatMode(.strict);
    const D = 5120;
    for (0..rows) |row| {
        var elem: [D]f32 = undefined;
        const xr = x[row * D ..][0..D];
        for (0..D) |i| {
            var xv = bf(xr[i]);
            if (r) |rv| {
                const hv = gm.rbf(xv + bf(rv[row * D + i]));
                h[row * D + i] = store(hv);
                xv = hv;
            }
            elem[i] = xv;
        }
        var partial: [256]f32 = undefined;
        for (0..256) |tid| {
            var a: [8]f32 = undefined;
            var b: [8]f32 = undefined;
            var c: [8]f32 = undefined;
            for (0..8) |k| {
                a[k] = elem[tid * 8 + k];
                b[k] = elem[2048 + tid * 8 + k];
                c[k] = if (tid < 128) elem[4096 + tid * 8 + k] else 0.0;
            }
            const c2: f32 = if (tid < 128) chunkLh(&c) else 0.0;
            partial[tid] = (chunkHl(&a) + chunkLh(&b)) + c2;
        }
        var warps: [8]f32 = undefined;
        for (0..8) |warp| {
            var lanes: [32]f32 = undefined;
            @memcpy(lanes[0..], partial[warp * 32 ..][0..32]);
            bfly(32, &lanes, 16);
            warps[warp] = lanes[0];
        }
        bfly(8, &warps, 4);
        const inv = mu.invRms5120(warps[0], eps);
        var yb: [D]u16 = undefined;
        for (0..D) |i| yb[i] = store((elem[i] * inv) * bf(w[i]));
        @memcpy(y[row * D ..][0..D], yb[0..]);
        const xsr = xs[row * (D / 64) ..][0 .. D / 64];
        octetSums(&yb, 0, 256, xsr, 0);
        octetSums(&yb, 2048, 256, xsr, 32);
        octetSums(&yb, 4096, 128, xsr, 64);
    }
}

pub fn addRmsnorm(x: []const u16, r: ?[]const u16, w: []const u16, h: []u16, y: []u16, xs: []f32, rows: usize, d: usize, eps: f32) void {
    @setFloatMode(.strict);
    if (d == 5120) return addRms5120(x, r, w, h, y, xs, rows, eps);
    for (0..rows) |row| {
        const at = row * d;
        var ss: f32 = 0;
        for (0..d) |i| {
            var xv = bf(x[at + i]);
            if (r) |rv| {
                const hv = gm.rbf(xv + bf(rv[at + i]));
                h[at + i] = store(hv);
                xv = hv;
            }
            ss = gm.fma(xv, xv, ss);
        }
        const inv = m.invRms(ss, d, eps);
        for (0..d) |i| {
            const xv = if (r != null) bf(h[at + i]) else bf(x[at + i]);
            y[at + i] = store(gm.rbf((xv * inv) * bf(w[i])));
        }
        groupSums(y[at..][0..d], xs[row * (d / 64) ..][0 .. d / 64]);
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

fn gatedStore(y: []const u16, z: []const u16, w: []const u16, out: []u16, i: usize, inv: f32) void {
    @setFloatMode(.strict);
    const yn = (bf(y[i]) * inv) * bf(w[i]);
    const zv = bf(z[i]);
    out[i] = store(gm.rbf((zv * mu.sigmoidApprox(zv)) * yn));
}

pub fn gatedNorm(y: []const u16, z: []const u16, w: []const u16, out: []u16, xs: []f32, rows: usize, vh: usize, dv: usize, eps: f32) void {
    @setFloatMode(.strict);
    for (0..rows) |row| {
        for (0..vh) |head| {
            const at = (row * vh + head) * dv;
            if (dv == 128) {
                var part: [32]f32 = undefined;
                for (0..32) |lane| {
                    const b = lane * 4;
                    const s0 = gm.fma(bf(y[at + b + 1]), bf(y[at + b + 1]), bf(y[at + b]) * bf(y[at + b]));
                    const s1 = gm.fma(bf(y[at + b + 3]), bf(y[at + b + 3]), bf(y[at + b + 2]) * bf(y[at + b + 2]));
                    part[lane] = s0 + s1;
                }
                bfly32(&part);
                for (0..32) |lane| {
                    const inv = mu.invRmsApprox(part[lane], dv, eps);
                    const b = lane * 4;
                    for (0..4) |k| gatedStore(y[at..], z[at..], w, out[at..], b + k, inv);
                }
            } else {
                var ss: f32 = 0;
                for (0..dv) |i| ss = gm.fma(bf(y[at + i]), bf(y[at + i]), ss);
                const inv = mu.invRmsApprox(ss, dv, eps);
                for (0..dv) |i| gatedStore(y[at..], z[at..], w, out[at..], i, inv);
            }
            const gs = dv / 64;
            groupSums(out[at..][0..dv], xs[row * (vh * dv / 64) + head * gs ..][0..gs]);
        }
    }
}

pub fn swiglu(gate: []const u16, up: []const u16, out: []u16, xs: []f32, rows: usize, n: usize) void {
    @setFloatMode(.strict);
    for (0..rows) |row| {
        const at = row * n;
        for (0..n) |i| {
            const gv = bf(gate[at + i]);
            out[at + i] = store(gm.rbf((gv * mu.sigmoidApprox(gv)) * bf(up[at + i])));
        }
        groupSums(out[at..][0..n], xs[row * (n / 64) ..][0 .. n / 64]);
    }
}

pub fn gateMul(o: []const u16, qg: []const u16, out: []u16, xs: []f32, rows: usize, heads: usize, d: usize) void {
    @setFloatMode(.strict);
    for (0..rows) |row| {
        for (0..heads) |head| {
            const src = (row * heads + head) * d;
            const gate = (row * heads + head) * 2 * d + d;
            for (0..d) |i| {
                const gv = bf(qg[gate + i]);
                out[src + i] = store(gm.rbf(bf(o[src + i]) * gm.sigmoid(gv)));
            }
            const gs = d / 64;
            groupSums(out[src..][0..d], xs[row * (heads * d / 64) + head * gs ..][0..gs]);
        }
    }
}

pub fn groupSumsRows(x: []const u16, xs: []f32, ld: usize, rows: usize, k: usize) void {
    const groups = k / 64;
    for (0..rows) |row| groupSums(x[row * ld ..][0..k], xs[row * groups ..][0..groups]);
}

pub fn embed(ids: []const i32, words: []const u32, scales: []const u16, bias: []const u16, out: []u16, d: usize) void {
    @setFloatMode(.strict);
    const groups = d / 64;
    for (ids, 0..) |tok_i, row| {
        const tok: usize = @intCast(tok_i);
        for (0..groups) |g| {
            const s = bf(scales[tok * groups + g]);
            const b = bf(bias[tok * groups + g]);
            for (0..8) |w| {
                const word = words[tok * (d / 8) + g * 8 + w];
                for (0..8) |n| {
                    const q: f32 = @floatFromInt((word >> @intCast(4 * n)) & 0xF);
                    const i = g * 64 + w * 8 + n;
                    out[row * d + i] = store(gm.rbf((q * s) + b));
                }
            }
        }
    }
}

pub fn b16Embed(ids: []const i32, table: []const u16, out: []u16, d: usize) void {
    for (ids, 0..) |tok_i, row| {
        const tok: usize = @intCast(tok_i);
        @memcpy(out[row * d ..][0..d], table[tok * d ..][0..d]);
    }
}

pub fn bf16ToF16(x: []const u16, y: []f16) void {
    for (x, y) |b, *o| o.* = @floatCast(bf(b));
}

fn convCh(qkv: []const u16, cs: []const u16, cw: []const u16, win: []const i32, row: usize, ch: usize, c: usize, cs_row: usize, nkeep: usize) f32 {
    @setFloatMode(.strict);
    var sum: f32 = 0;
    if (ch < c) {
        for (0..nkeep + 1) |j| {
            const src = win[row * (nkeep + 1) + j];
            const x = if (src < @as(i32, @intCast(nkeep)))
                bf(cs[(cs_row + @as(usize, @intCast(src))) * c + ch])
            else
                bf(qkv[@as(usize, @intCast(src - @as(i32, @intCast(nkeep)))) * c + ch]);
            sum = gm.fma(x, bf(cw[ch * (nkeep + 1) + j]), sum);
        }
    }
    return gm.rbf(sum * mu.sigmoidApprox(sum));
}

pub fn gdnPre(qkv: []const u16, cs: []const u16, cw: []const u16, win: []const i32, a: []const u16, b: []const u16, alog: []const f32, dtb: []const f32, q: []u16, k: []u16, v: []u16, g: []f32, beta: []f32, sid: ?[]const i32, rows: usize, c: usize, kh: usize, vh: usize, dk: usize, nkeep: usize, ab_f16: bool) void {
    @setFloatMode(.strict);
    if (dk > 128) return;
    const multi = sid != null;
    for (0..rows) |row| {
        const cs_row: usize = if (multi) @as(usize, @intCast(sid.?[row])) * nkeep else 0;
        for (0..2 * kh + vh) |head| {
            const ch0 = head * dk;
            if (head < 2 * kh and dk == 128) {
                var c0s: [64]f32 = undefined;
                var c1s: [64]f32 = undefined;
                var part: [64]f32 = undefined;
                for (0..64) |tid| {
                    c0s[tid] = convCh(qkv, cs, cw, win, row, ch0 + tid * 2, c, cs_row, nkeep);
                    c1s[tid] = convCh(qkv, cs, cw, win, row, ch0 + tid * 2 + 1, c, cs_row, nkeep);
                    part[tid] = gm.fma(c1s[tid], c1s[tid], c0s[tid] * c0s[tid]);
                }
                var w0: [32]f32 = undefined;
                var w1: [32]f32 = undefined;
                for (0..32) |i| {
                    w0[i] = part[i];
                    w1[i] = part[i + 32];
                }
                bfly32(&w0);
                bfly32(&w1);
                const inv = mu.invRmsApprox(w0[0] + w1[0], dk, 1e-6);
                const is_q = head < kh;
                const scale: f32 = if (is_q) 1.0 / @as(f32, @floatFromInt(dk)) else 1.0 / @sqrt(@as(f32, @floatFromInt(dk)));
                const hk: usize = if (is_q) head else head - kh;
                const dst = (if (is_q) q else k)[(row * kh + hk) * dk ..][0..dk];
                for (0..64) |tid| {
                    dst[tid * 2] = store(gm.rbf((c0s[tid] * inv) * scale));
                    dst[tid * 2 + 1] = store(gm.rbf((c1s[tid] * inv) * scale));
                }
                continue;
            }
            var cv: [128]f32 = undefined;
            if (head < 2 * kh) {
                var ss: f32 = 0;
                for (0..dk) |i| {
                    cv[i] = convCh(qkv, cs, cw, win, row, ch0 + i, c, cs_row, nkeep);
                    ss = gm.fma(cv[i], cv[i], ss);
                }
                const inv = mu.invRmsApprox(ss, dk, 1e-6);
                const is_q = head < kh;
                const scale: f32 = if (is_q) 1.0 / @as(f32, @floatFromInt(dk)) else 1.0 / @sqrt(@as(f32, @floatFromInt(dk)));
                const hk: usize = if (is_q) head else head - kh;
                const dst = (if (is_q) q else k)[(row * kh + hk) * dk ..][0..dk];
                for (0..dk) |i| dst[i] = store(gm.rbf((cv[i] * inv) * scale));
            } else {
                const hv = head - 2 * kh;
                const dst = v[(row * vh + hv) * dk ..][0..dk];
                for (0..dk) |i| dst[i] = store(convCh(qkv, cs, cw, win, row, ch0 + i, c, cs_row, nkeep));
                const av = m.abF32(a[row * vh + hv], ab_f16) + dtb[hv];
                const sp = mu.softplusApprox(av);
                g[row * vh + hv] = mu.expApprox((-mu.expApprox(alog[hv])) * sp);
                beta[row * vh + hv] = mu.sigmoidApprox(m.abF32(b[row * vh + hv], ab_f16));
            }
        }
    }
}

pub fn attnPrep(qg: []const u16, kv: []const u16, qn: []const u16, kn: []const u16, pos: []const i32, invf: []const f32, qo: []u16, ko: []u16, rows: usize, h: usize, hkv: usize, d: usize, half: usize, eps: f32, mrope_rows: usize, hsec: usize, wsec: usize) void {
    @setFloatMode(.strict);
    if (d > 256) return;
    for (0..rows) |row| {
        for (0..h + hkv) |head| {
            const is_q = head < h;
            const hd = if (is_q) head else head - h;
            const src = if (is_q) qg[(row * h + hd) * 2 * d ..][0..d] else kv[(row * hkv + hd) * d ..][0..d];
            const norm = if (is_q) qn else kn;
            const dst = if (is_q) qo[(row * h + hd) * d ..][0..d] else ko[(row * hkv + hd) * d ..][0..d];
            var ss: f32 = 0;
            for (0..d) |i| ss = gm.fma(bf(src[i]), bf(src[i]), ss);
            const inv = mu.invRmsApprox(ss, d, eps);
            var xn: [256]f32 = undefined;
            for (0..d) |i| xn[i] = gm.rbf((bf(src[i]) * inv) * bf(norm[i]));
            for (0..d) |i| {
                var out = xn[i];
                if (i < 2 * half) {
                    const freq = if (i < half) i else i - half;
                    var axis: usize = 0;
                    if (mrope_rows > 0) {
                        const sec = freq % 3;
                        if (sec == 1 and freq < 3 * hsec) axis = 1;
                        if (sec == 2 and freq < 3 * wsec) axis = 2;
                    }
                    const pr = if (mrope_rows > 0) pos[axis * mrope_rows + row] else pos[row];
                    const ang = @as(f32, @floatFromInt(pr)) * invf[freq];
                    var sn: f32 = 0;
                    var cs: f32 = 0;
                    m.sincos(ang, &sn, &cs);
                    const pn = xn[if (i < half) i + half else i - half];
                    out = if (i < half) (xn[i] * cs) - (pn * sn) else (xn[i] * cs) + (pn * sn);
                }
                dst[i] = store(gm.rbf(out));
            }
        }
    }
}

test "add rmsnorm leaves h untouched without a residual and writes a normed row" {
    const d = 128;
    var x: [d]u16 = undefined;
    var w: [d]u16 = undefined;
    var h: [d]u16 = undefined;
    var y: [d]u16 = undefined;
    var xs: [d / 64]f32 = undefined;
    for (0..d) |i| {
        x[i] = gm.f32ToBf16(0.01 * @as(f32, @floatFromInt(i + 1)));
        w[i] = gm.f32ToBf16(1);
        h[i] = 0x1234;
    }
    addRmsnorm(&x, null, &w, &h, &y, &xs, 1, d, 1e-6);
    try std.testing.expectEqual(@as(u16, 0x1234), h[0]);
    try std.testing.expect(y[0] != 0);
    try std.testing.expect(xs[0] != 0);
}
