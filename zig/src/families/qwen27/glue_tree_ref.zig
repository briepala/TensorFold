//! Host reference for q27_tree.cu. One warp is 32 lanes. OFF values are bf16 element offsets from base.

const std = @import("std");
const gm = @import("nemotron").glue_math;
const m = @import("q27_math.zig");
const mu = @import("mufu_ref.zig");

const tile = 64;
const chunk = 512;
const tiles_per = chunk / tile;
const gr = 4;
const maxd = 128;

const Lane = struct { m: f32, l: f32, o: [8]f32 };

fn bf(h: u16) f32 {
    return gm.bf16ToF32(h);
}

fn fresh() [32]Lane {
    var a: [32]Lane = undefined;
    for (&a) |*ln| {
        ln.m = m.negInf();
        ln.l = 0;
        @memset(&ln.o, 0);
    }
    return a;
}

fn tileRows(q: []const u16, kbuf: []const u16, vbuf: []const u16, a: *[32]Lane, valid: *const [tile]i32, scale: f32, d: usize) void {
    @setFloatMode(.strict);
    var s1: [32]f32 = undefined;
    var s2: [32]f32 = undefined;
    for (0..32) |lane| {
        s1[lane] = m.negInf();
        s2[lane] = m.negInf();
        const v1 = valid[lane] != 0;
        const v2 = valid[32 + lane] != 0;
        if (v1 or v2) {
            var d1: f32 = 0;
            var d2: f32 = 0;
            for (0..d) |i| {
                const qv = bf(q[i]);
                if (v1) d1 = gm.fma(qv, bf(kbuf[lane * d + i]), d1);
                if (v2) d2 = gm.fma(qv, bf(kbuf[(32 + lane) * d + i]), d2);
            }
            if (v1) s1[lane] = d1 * scale;
            if (v2) s2[lane] = d2 * scale;
        }
    }
    var tile_m: [32]f32 = undefined;
    for (0..32) |lane| tile_m[lane] = @max(s1[lane], s2[lane]);
    var off: usize = 16;
    while (off > 0) : (off >>= 1) {
        var nxt: [32]f32 = undefined;
        for (0..32) |lane| nxt[lane] = @max(tile_m[lane], tile_m[lane ^ off]);
        tile_m = nxt;
    }
    var active: [32]bool = undefined;
    var next_m: [32]f32 = undefined;
    var alpha: [32]f32 = undefined;
    var p1: [32]f32 = undefined;
    var p2: [32]f32 = undefined;
    var psum: [32]f32 = undefined;
    for (0..32) |lane| {
        active[lane] = tile_m[lane] != m.negInf();
        next_m[lane] = a[lane].m;
        if (active[lane] and tile_m[lane] > next_m[lane]) next_m[lane] = tile_m[lane];
        alpha[lane] = if (!active[lane]) 1 else if (a[lane].m == m.negInf()) 0 else mu.expApprox(a[lane].m - next_m[lane]);
        const v1 = valid[lane] != 0;
        const v2 = valid[32 + lane] != 0;
        p1[lane] = if (v1 and active[lane]) mu.expApprox(s1[lane] - next_m[lane]) else 0;
        p2[lane] = if (v2 and active[lane]) mu.expApprox(s2[lane] - next_m[lane]) else 0;
        psum[lane] = p1[lane] + p2[lane];
    }
    off = 16;
    while (off > 0) : (off >>= 1) {
        var nxt: [32]f32 = undefined;
        for (0..32) |lane| nxt[lane] = psum[lane] + psum[lane ^ off];
        psum = nxt;
    }
    for (0..32) |lane| {
        a[lane].l = (a[lane].l * alpha[lane]) + psum[lane];
        for (&a[lane].o) |*o| o.* *= alpha[lane];
    }
    for (0..32) |j| {
        const pj = p1[j];
        if (pj != 0) {
            const pb = gm.rbf(pj);
            for (0..32) |lane| for (0..8) |e| {
                const i = lane + e * 32;
                if (i < d) a[lane].o[e] = gm.fma(pb, bf(vbuf[j * d + i]), a[lane].o[e]);
            };
        }
    }
    for (0..32) |j| {
        const pj = p2[j];
        if (pj != 0) {
            const pb = gm.rbf(pj);
            for (0..32) |lane| for (0..8) |e| {
                const i = lane + e * 32;
                if (i < d) a[lane].o[e] = gm.fma(pb, bf(vbuf[(32 + j) * d + i]), a[lane].o[e]);
            };
        }
    }
    for (0..32) |lane| {
        if (active[lane]) a[lane].m = next_m[lane];
    }
}

fn fold(a: *[32]Lane, cm: f32, cl: f32, co: *const [32][8]f32) void {
    @setFloatMode(.strict);
    for (a, 0..) |*ln, lane| {
        const on = cl > 0;
        var next_m = ln.m;
        if (on and cm > next_m) next_m = cm;
        const fa: f32 = if (!on) 1 else if (ln.m == m.negInf()) 0 else mu.expApprox(ln.m - next_m);
        const fb: f32 = if (on) mu.expApprox(cm - next_m) else 0;
        ln.l = (ln.l * fa) + (cl * fb);
        for (0..8) |e| ln.o[e] = (ln.o[e] * fa) + (co[lane][e] * fb);
        ln.m = next_m;
    }
}

fn foldAcc(a: *[32]Lane, b: *const [32]Lane) void {
    var co: [32][8]f32 = undefined;
    for (0..32) |lane| co[lane] = b[lane].o;
    fold(a, b[0].m, b[0].l, &co);
}

fn storePartial(a: *const [32]Lane, po: []f32, pm: []f32, pl: []f32, base: usize, d: usize) void {
    pm[base] = a[0].m;
    pl[base] = a[0].l;
    for (0..32) |lane| for (0..8) |e| {
        const i = lane + e * 32;
        if (i < d) po[base * d + i] = a[lane].o[e];
    };
}

fn taken(base: []const u16, delta: i64, index: usize) u16 {
    return base[@as(usize, @intCast(delta + @as(i64, @intCast(index))))];
}

pub fn paths(parents: []const i32, out: []i32, depths: []i32, w: usize) void {
    for (0..w) |node| {
        var cur: i32 = @intCast(node);
        var depth: i32 = 0;
        while (cur >= 0 and depth < maxd) {
            depth += 1;
            cur = parents[@intCast(cur)];
        }
        depths[node] = depth;
        cur = @intCast(node);
        var slot: i32 = depth - 1;
        while (slot >= 0) : (slot -= 1) {
            out[node * maxd + @as(usize, @intCast(slot))] = cur;
            cur = parents[@intCast(cur)];
        }
    }
}

fn nchOf(p: i32, rows: i32) i32 {
    return @divTrunc(p + rows + chunk - 1, chunk);
}

pub fn shared(q: []const u16, base: []const u16, off: []const i64, stream: []const i32, items: []const i32, po: []f32, pm: []f32, pl: []f32, w: usize, h: usize, hk: usize, d: usize, scale: f32) void {
    if (d > 256) return;
    const g = h / hk;
    const n_items = items.len / 3;
    var kbuf: [tile * 256]u16 = undefined;
    var vbuf: [tile * 256]u16 = undefined;
    for (0..n_items) |item| {
        for (0..hk) |hk_i| {
            const s: usize = @intCast(items[item * 3]);
            const first: usize = @intCast(items[item * 3 + 1]);
            const code = items[item * 3 + 2];
            const start: usize = @intCast(stream[s * 4]);
            const rows = stream[s * 4 + 1];
            const p: usize = @intCast(stream[s * 4 + 2]);
            const nslots = stream[s * 4 + 3];
            const nch = nchOf(@intCast(p), rows);
            const groups = @divTrunc(nch - nslots, gr - 1);
            const whole = code >= 0;
            const chunk_i: i32 = if (whole) code * gr else groups * gr - 1 - code;
            const n: i32 = if (whole) gr else 1;
            const slot_i: i32 = if (whole) code else chunk_i - groups * (gr - 1);
            if ((chunk_i + n) * chunk > @as(i32, @intCast(p))) continue;
            const koff = off[s * 2];
            const voff = off[s * 2 + 1];
            for (0..16) |warp| {
                const rr = first + warp;
                if (rr >= @as(usize, @intCast(rows)) * g) continue;
                const node = start + rr / g;
                const head = hk_i * g + rr % g;
                const qrow = q[(node * h + head) * d ..][0..d];
                var ga = fresh();
                var c: i32 = 0;
                while (c < n) : (c += 1) {
                    var ca = fresh();
                    for (0..tiles_per) |t| {
                        const key0: usize = @intCast((chunk_i + c) * chunk + @as(i32, @intCast(t * tile)));
                        var valid: [tile]i32 = undefined;
                        for (0..tile) |j| {
                            const on = key0 + j < p;
                            valid[j] = @intFromBool(on);
                            for (0..d) |i| {
                                const at = ((key0 + j) * hk + hk_i) * d + i;
                                kbuf[j * d + i] = if (on) taken(base, koff, at) else 0;
                                vbuf[j * d + i] = if (on) taken(base, voff, at) else 0;
                            }
                        }
                        tileRows(qrow, kbuf[0 .. tile * d], vbuf[0 .. tile * d], &ca, &valid, scale, d);
                    }
                    if (whole) foldAcc(&ga, &ca) else ga = ca;
                }
                const slot: usize = @intCast(slot_i);
                storePartial(&ga, po, pm, pl, (slot * w + node) * h + head, d);
            }
        }
    }
}

pub fn tail(q: []const u16, kn: []const u16, vn: []const u16, base: []const u16, off: []const i64, stream: []const i32, rows_of: []const i32, path: []const i32, depths: []const i32, po: []f32, pm: []f32, pl: []f32, w: usize, h: usize, hk: usize, d: usize, scale: f32) void {
    if (d > 256) return;
    const g = h / hk;
    const tails = 1 + (maxd + chunk - 1) / chunk;
    var kbuf: [tile * 256]u16 = undefined;
    var vbuf: [tile * 256]u16 = undefined;
    for (0..w) |node| {
        for (0..hk) |hk_i| {
            const s: usize = @intCast(rows_of[node]);
            const p: usize = @intCast(stream[s * 4 + 2]);
            const rows = stream[s * 4 + 1];
            const nslots = stream[s * 4 + 3];
            const nch = nchOf(@intCast(p), rows);
            const groups = @divTrunc(nch - nslots, gr - 1);
            for (0..tails) |z| {
                const chunk_i: i32 = @as(i32, @intCast(p / chunk)) + @as(i32, @intCast(z));
                if (chunk_i >= nch) continue;
                for (0..g) |warp| {
                    const head = hk_i * g + warp;
                    const qrow = q[(node * h + head) * d ..][0..d];
                    const depth = depths[node];
                    var acc = fresh();
                    const koff = off[s * 2];
                    const voff = off[s * 2 + 1];
                    for (0..tiles_per) |t| {
                        var valid: [tile]i32 = undefined;
                        for (0..tile) |j| {
                            const logical: i32 = chunk_i * chunk + @as(i32, @intCast(t * tile + j));
                            const path_slot = logical - @as(i32, @intCast(p));
                            const committed = logical < @as(i32, @intCast(p));
                            const on_path = path_slot >= 0 and path_slot < depth;
                            valid[j] = @intFromBool(committed or on_path);
                            for (0..d) |i| {
                                var kv: u16 = 0;
                                var vv: u16 = 0;
                                if (committed) {
                                    const at = (@as(usize, @intCast(logical)) * hk + hk_i) * d + i;
                                    kv = taken(base, koff, at);
                                    vv = taken(base, voff, at);
                                } else if (on_path) {
                                    const knode: usize = @intCast(path[node * maxd + @as(usize, @intCast(path_slot))]);
                                    const at = (knode * hk + hk_i) * d + i;
                                    kv = kn[at];
                                    vv = vn[at];
                                }
                                kbuf[j * d + i] = kv;
                                vbuf[j * d + i] = vv;
                            }
                        }
                        tileRows(qrow, kbuf[0 .. tile * d], vbuf[0 .. tile * d], &acc, &valid, scale, d);
                    }
                    const slot: usize = @intCast(chunk_i - groups * (gr - 1));
                    storePartial(&acc, po, pm, pl, (slot * w + node) * h + head, d);
                }
            }
        }
    }
}

fn loadSlot(po: []const f32, pm: []const f32, pl: []const f32, slot: usize, w: usize, node: usize, h: usize, head: usize, d: usize, cm: *f32, cl: *f32, co: *[32][8]f32) void {
    const base = (slot * w + node) * h + head;
    cm.* = pm[base];
    cl.* = pl[base];
    for (0..32) |lane| for (0..8) |e| {
        const i = lane + e * 32;
        co[lane][e] = if (i < d) po[base * d + i] else 0;
    };
}

pub fn merge(po: []const f32, pm: []const f32, pl: []const f32, out: []u16, stream: []const i32, rows_of: []const i32, w: usize, h: usize, hk: usize, d: usize) void {
    @setFloatMode(.strict);
    if (d > 256) return;
    const g = h / hk;
    for (0..w) |node| {
        const s: usize = @intCast(rows_of[node]);
        const rows = stream[s * 4 + 1];
        const p: usize = @intCast(stream[s * 4 + 2]);
        const nslots: usize = @intCast(stream[s * 4 + 3]);
        const nch = nchOf(@intCast(p), rows);
        const groups: usize = @intCast(@divTrunc(nch - @as(i32, @intCast(nslots)), gr - 1));
        for (0..hk) |hk_i| {
            for (0..g) |warp| {
                const head = hk_i * g + warp;
                var acc = fresh();
                for (0..groups) |slot| {
                    var cm: f32 = 0;
                    var cl: f32 = 0;
                    var co: [32][8]f32 = undefined;
                    loadSlot(po, pm, pl, slot, w, node, h, head, d, &cm, &cl, &co);
                    fold(&acc, cm, cl, &co);
                }
                var gacc = fresh();
                var slot = groups;
                while (slot < nslots) : (slot += 1) {
                    var cm: f32 = 0;
                    var cl: f32 = 0;
                    var co: [32][8]f32 = undefined;
                    loadSlot(po, pm, pl, slot, w, node, h, head, d, &cm, &cl, &co);
                    fold(&gacc, cm, cl, &co);
                    if ((slot - groups + 1) % gr == 0 or slot == nslots - 1) {
                        foldAcc(&acc, &gacc);
                        gacc = fresh();
                    }
                }
                const inv = mu.rcpApprox(acc[0].l);
                for (0..32) |lane| for (0..8) |e| {
                    const i = lane + e * 32;
                    if (i < d) out[(node * h + head) * d + i] = gm.f32ToBf16(acc[lane].o[e] * inv);
                };
            }
        }
    }
}

test "a chain's path is root to node" {
    const parents = [_]i32{ -1, 0, 1 };
    var out: [3 * maxd]i32 = undefined;
    var depths: [3]i32 = undefined;
    paths(&parents, &out, &depths, 3);
    try std.testing.expectEqual(@as(i32, 3), depths[2]);
    try std.testing.expectEqual(@as(i32, 0), out[2 * maxd]);
    try std.testing.expectEqual(@as(i32, 1), out[2 * maxd + 1]);
    try std.testing.expectEqual(@as(i32, 2), out[2 * maxd + 2]);
}
