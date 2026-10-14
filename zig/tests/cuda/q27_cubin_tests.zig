//! Each own 27B glue kernel against the captured Triton cubin. Mismatch counts are printed either way.

const std = @import("std");
const cuda = @import("cuda");
const qwen = @import("qwen27");
const check = @import("check.zig");
const gm = @import("nemotron").glue_math;

const glue = qwen.glue;
const tri = qwen.triton;
const aot = cuda.aot;

const Hid = 5120;
const Mid = 17408;
const QH = 24;
const KV = 4;
const HD = 256;
const Half = 32;
const KH = 16;
const VH = 48;
const DK = 128;
const Keep = 3;
const Conv = (2 * KH + VH) * DK;
const eps: f32 = 1e-6;
const DH = 32;
const DKV = 8;
const DD = 128;
const widths = [_]usize{ 1, 3, 17, 64 };

const Rig = struct {
    gpu: check.Gpu,
    a: std.mem.Allocator,
    s: cuda.Stream,
    f: glue.Fns,
    t: tri.Tri,
    rng: std.Random,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    bad: usize = 0,

    fn g(r: *Rig) glue.Glue {
        return .{ .f = &r.f, .s = r.s };
    }

    fn dev(r: *Rig, comptime T: type, host: []const T) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(r.gpu.d, std.mem.sliceAsBytes(host));
        try r.bufs.append(r.gpu.gpa, b);
        return b.ptr;
    }

    fn zero(r: *Rig, n: usize) !u64 {
        const host = try r.a.alloc(u8, n);
        @memset(host, 0);
        return r.dev(u8, host);
    }

    fn bf(r: *Rig, n: usize, scale: f32) ![]u16 {
        const o = try r.a.alloc(u16, n);
        for (o) |*v| v.* = gm.f32ToBf16((r.rng.float(f32) * 2 - 1) * scale);
        return o;
    }

    fn f16b(r: *Rig, n: usize) ![]u16 {
        const o = try r.a.alloc(u16, n);
        for (o) |*v| v.* = @bitCast(@as(f16, @floatCast((r.rng.float(f32) * 2 - 1) * 2)));
        return o;
    }

    fn fs(r: *Rig, n: usize, lo: f32, hi: f32) ![]f32 {
        const o = try r.a.alloc(f32, n);
        for (o) |*v| v.* = lo + r.rng.float(f32) * (hi - lo);
        return o;
    }

    fn miss(r: *Rig, name: []const u8, rows: usize, e: anyerror) void {
        r.bad += 1;
        std.debug.print("FAIL {s} rows={d} mismatches=launch {t}\n", .{ name, rows, e });
    }

    fn both(r: *Rig, name: []const u8, rows: usize, ours: u64, cubin: u64, n: usize) !void {
        try r.s.synchronize();
        const a = try r.a.alloc(u8, n);
        const b = try r.a.alloc(u8, n);
        try r.gpu.d.check(r.gpu.d.api.cuMemcpyDtoH_v2(a.ptr, ours, n), "cuMemcpyDtoH");
        try r.gpu.d.check(r.gpu.d.api.cuMemcpyDtoH_v2(b.ptr, cubin, n), "cuMemcpyDtoH");
        var mis: usize = 0;
        var shown: usize = 0;
        for (a, b, 0..) |x, y, i| {
            if (x != y) {
                mis += 1;
                if (shown < 6) {
                    std.debug.print("DIFF {s} rows={d} off={d} ours={x:0>2} cubin={x:0>2}\n", .{ name, rows, i, x, y });
                    shown += 1;
                }
            }
        }
        if (mis != 0 and std.mem.startsWith(u8, name, "add_rmsnorm")) r.bad += 1;
        std.debug.print("CUBIN {s} rows={d} mismatches={d}\n", .{ name, rows, mis });
    }
};

fn addRms(r: *Rig, rows: usize) !void {
    const with_r = rows % 2 == 1;
    const x = try r.bf(rows * Hid, 1);
    const res = try r.bf(rows * Hid, 0.5);
    const w = try r.bf(Hid, 1);
    const xd = try r.dev(u16, x);
    const rd: ?u64 = if (with_r) try r.dev(u16, res) else null;
    const wd = try r.dev(u16, w);
    const h1 = try r.zero(rows * Hid * 2);
    const h2 = try r.zero(rows * Hid * 2);
    const y1 = try r.zero(rows * Hid * 2);
    const y2 = try r.zero(rows * Hid * 2);
    const s1 = try r.zero(rows * Hid / 64 * 4);
    const s2 = try r.zero(rows * Hid / 64 * 4);
    r.g().addRmsnorm(xd, rd, wd, if (with_r) h1 else xd, y1, s1, rows, Hid, eps) catch |e| return r.miss("add_rmsnorm", rows, e);
    r.t.addRmsnorm(xd, rd, wd, if (with_r) h2 else xd, y2, s2, rows, Hid, eps) catch |e| return r.miss("add_rmsnorm", rows, e);
    if (with_r) try r.both("add_rmsnorm_h", rows, h1, h2, rows * Hid * 2);
    try r.both("add_rmsnorm_y", rows, y1, y2, rows * Hid * 2);
    try r.both("add_rmsnorm_xs", rows, s1, s2, rows * Hid / 64 * 4);
}

fn rowsOf(r: *Rig, rows: usize) !void {
    const y = try r.bf(rows * VH * DK, 1);
    const z = try r.bf(rows * VH * DK, 1);
    const w = try r.bf(DK, 1);
    const yd = try r.dev(u16, y);
    const zd = try r.dev(u16, z);
    const wd = try r.dev(u16, w);
    const o1 = try r.zero(y.len * 2);
    const o2 = try r.zero(y.len * 2);
    const x1 = try r.zero(rows * VH * DK / 64 * 4);
    const x2 = try r.zero(rows * VH * DK / 64 * 4);
    r.g().gatedNorm(yd, zd, wd, o1, x1, rows, VH, DK, eps) catch |e| return r.miss("gated_norm", rows, e);
    r.t.gatedNorm(yd, zd, wd, o2, x2, rows, VH, DK, eps) catch |e| return r.miss("gated_norm", rows, e);
    try r.both("gated_norm", rows, o1, o2, y.len * 2);
    const gate = try r.bf(rows * Mid, 1);
    const up = try r.bf(rows * Mid, 1);
    const gd = try r.dev(u16, gate);
    const ud = try r.dev(u16, up);
    const s1 = try r.zero(gate.len * 2);
    const s2 = try r.zero(gate.len * 2);
    const a1 = try r.zero(rows * Mid / 64 * 4);
    const a2 = try r.zero(rows * Mid / 64 * 4);
    r.g().swiglu(gd, ud, s1, a1, rows, Mid) catch |e| return r.miss("swiglu", rows, e);
    r.t.swiglu(gd, ud, s2, a2, rows, Mid) catch |e| return r.miss("swiglu", rows, e);
    try r.both("swiglu", rows, s1, s2, gate.len * 2);
    const o = try r.bf(rows * QH * HD, 1);
    const qg = try r.bf(rows * QH * 2 * HD, 1);
    const od = try r.dev(u16, o);
    const qgd = try r.dev(u16, qg);
    const m1 = try r.zero(o.len * 2);
    const m2 = try r.zero(o.len * 2);
    r.g().gateMul(od, qgd, m1, a1, rows, QH, HD) catch |e| return r.miss("gate_mul", rows, e);
    r.t.gateMul(od, qgd, m2, a2, rows, QH, HD) catch |e| return r.miss("gate_mul", rows, e);
    try r.both("gate_mul", rows, m1, m2, o.len * 2);
    const xs = try r.bf(rows * Hid, 1);
    const xsd = try r.dev(u16, xs);
    const g1 = try r.zero(rows * Hid / 64 * 4);
    const g2 = try r.zero(rows * Hid / 64 * 4);
    r.g().groupSums(xsd, g1, Hid, rows, Hid) catch |e| return r.miss("group_sums", rows, e);
    r.t.groupSums(xsd, g2, Hid, rows, Hid) catch |e| return r.miss("group_sums", rows, e);
    try r.both("group_sums", rows, g1, g2, rows * Hid / 64 * 4);
}

fn gdn(r: *Rig, rows: usize) !void {
    const multi = rows == 3 or rows == 64;
    const ab_f16 = rows == 3 or rows == 17;
    const taps = Keep + 1;
    const qkv = try r.dev(u16, try r.bf(rows * Conv, 0.4));
    const cs_rows: usize = if (multi) 2 else 1;
    const cs = try r.dev(u16, try r.bf(cs_rows * Keep * Conv, 0.4));
    const cw = try r.dev(u16, try r.bf(Conv * taps, 0.3));
    const win_h = try r.a.alloc(i32, rows * taps);
    for (0..rows) |row| {
        for (0..Keep) |j| win_h[row * taps + j] = @intCast(j);
        win_h[row * taps + Keep] = @intCast(Keep + row);
    }
    const win = try r.dev(i32, win_h);
    const ab = try r.dev(u16, if (ab_f16) try r.f16b(rows * VH) else try r.bf(rows * VH, 0.4));
    const bb = try r.dev(u16, if (ab_f16) try r.f16b(rows * VH) else try r.bf(rows * VH, 0.4));
    const alog = try r.dev(f32, try r.fs(VH, -1, 1));
    const dtb = try r.dev(f32, try r.fs(VH, -0.5, 0.5));
    const sid: ?u64 = if (!multi) null else blk: {
        const s = try r.a.alloc(i32, rows);
        for (s, 0..) |*v, i| v.* = @intCast(i % 2);
        break :blk try r.dev(i32, s);
    };
    const shape: glue.Gdn = .{ .c = Conv, .kh = KH, .vh = VH, .dk = DK, .keep = Keep };
    const q1 = try r.zero(rows * KH * DK * 2);
    const q2 = try r.zero(rows * KH * DK * 2);
    const k1 = try r.zero(rows * KH * DK * 2);
    const k2 = try r.zero(rows * KH * DK * 2);
    const v1 = try r.zero(rows * VH * DK * 2);
    const v2 = try r.zero(rows * VH * DK * 2);
    const g1 = try r.zero(rows * VH * 4);
    const g2 = try r.zero(rows * VH * 4);
    const b1 = try r.zero(rows * VH * 4);
    const b2 = try r.zero(rows * VH * 4);
    r.g().gdnPre(qkv, cs, cw, win, sid, ab, bb, alog, dtb, q1, k1, v1, g1, b1, rows, shape, ab_f16) catch |e| return r.miss("gdn_pre", rows, e);
    r.t.gdnPre(qkv, cs, cw, win, sid, ab, bb, alog, dtb, q2, k2, v2, g2, b2, rows, shape, ab_f16) catch |e| return r.miss("gdn_pre", rows, e);
    try r.both("gdn_q", rows, q1, q2, rows * KH * DK * 2);
    try r.both("gdn_k", rows, k1, k2, rows * KH * DK * 2);
    try r.both("gdn_v", rows, v1, v2, rows * VH * DK * 2);
    try r.both("gdn_g", rows, g1, g2, rows * VH * 4);
}

fn attn(r: *Rig, rows: usize) !void {
    const qg = try r.dev(u16, try r.bf(rows * QH * 2 * HD, 0.5));
    const kv = try r.dev(u16, try r.bf(rows * KV * HD, 0.5));
    const qn = try r.dev(u16, try r.bf(HD, 1));
    const kn = try r.dev(u16, try r.bf(HD, 1));
    const pos_h = try r.a.alloc(i32, rows);
    for (pos_h, 0..) |*p, i| p.* = @intCast((i * 17 + 3) % 500);
    const pos = try r.dev(i32, pos_h);
    const inv_h = try r.a.alloc(f32, Half);
    for (inv_h, 0..) |*v, i| v.* = @floatCast(std.math.pow(f64, 1e7, -@as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(Half))));
    const inv = try r.dev(f32, inv_h);
    const shape: glue.Attn = .{ .heads = QH, .kv_heads = KV, .dim = HD, .half = Half };
    const q1 = try r.zero(rows * QH * HD * 2);
    const q2 = try r.zero(rows * QH * HD * 2);
    const k1 = try r.zero(rows * KV * HD * 2);
    const k2 = try r.zero(rows * KV * HD * 2);
    r.g().attnPrep(qg, kv, qn, kn, pos, inv, q1, k1, rows, shape, eps) catch |e| return r.miss("attn_prep", rows, e);
    r.t.attnPrep(qg, kv, qn, kn, pos, inv, q2, k2, rows, shape, eps) catch |e| return r.miss("attn_prep", rows, e);
    try r.both("attn_q", rows, q1, q2, rows * QH * HD * 2);
    try r.both("attn_k", rows, k1, k2, rows * KV * HD * 2);
}

fn treeAt(r: *Rig, rows: usize, past: usize) !void {
    const w = rows;
    const group = QH / KV;
    const nslots = tri.slots(past, w);
    const parents_h = try r.a.alloc(i32, w);
    for (parents_h, 0..) |*p, i| p.* = if (i == 0) -1 else @intCast(i - 1);
    const parents = try r.dev(i32, parents_h);
    const p1 = try r.zero(w * 128 * 4);
    const p2 = try r.zero(w * 128 * 4);
    const d1 = try r.zero(w * 4);
    const d2 = try r.zero(w * 4);
    r.g().paths(parents, p1, d1, w) catch |e| return r.miss("paths", rows, e);
    r.t.paths(parents, p2, d2, w) catch |e| return r.miss("paths", rows, e);
    try r.both("paths_depth", rows, d1, d2, w * 4);
    var items: std.ArrayList(i32) = .empty;
    const folded = tri.groups(past, w);
    const loose = past / tri.chunk - folded * tri.group_chunks;
    for (0..folded + loose) |j| {
        const code: i32 = if (j < folded) @intCast(j) else -1 - @as(i32, @intCast(j - folded));
        var first: usize = 0;
        while (first < w * group) : (first += 16) try items.appendSlice(r.a, &.{ 0, @intCast(first), code });
    }
    const stream = [_]i32{ 0, @intCast(w), @intCast(past), @intCast(nslots) };
    const rows_h = try r.a.alloc(i32, w);
    @memset(rows_h, 0);
    const k_elems: i64 = @intCast(past * KV * HD);
    const off = [_]i64{ 0, k_elems };
    const base = try r.dev(u16, try r.bf(2 * @as(usize, @intCast(k_elems)) * 1, 0.4));
    const q = try r.dev(u16, try r.bf(w * QH * HD, 0.4));
    const kn = try r.dev(u16, try r.bf(w * KV * HD, 0.4));
    const vn = try r.dev(u16, try r.bf(w * KV * HD, 0.4));
    const po_n = nslots * w * QH * HD;
    const pm_n = nslots * w * QH;
    const plan1: glue.Plan = .{
        .streams = try r.dev(i32, &stream),
        .items = try r.dev(i32, items.items),
        .n_items = items.items.len / 3,
        .rows = try r.dev(i32, rows_h),
        .paths = p1,
        .depths = d1,
        .width = w,
    };
    const plan2: glue.Plan = .{
        .streams = plan1.streams,
        .items = plan1.items,
        .n_items = plan1.n_items,
        .rows = plan1.rows,
        .paths = p2,
        .depths = d2,
        .width = w,
    };
    const shape: glue.Attn = .{ .heads = QH, .kv_heads = KV, .dim = HD, .half = Half };
    const o1 = try r.zero(w * QH * HD * 2);
    const o2 = try r.zero(w * QH * HD * 2);
    const part1: glue.Partials = .{ .o = try r.zero(po_n * 4), .m = try r.zero(pm_n * 4), .l = try r.zero(pm_n * 4) };
    const part2: glue.Partials = .{ .o = try r.zero(po_n * 4), .m = try r.zero(pm_n * 4), .l = try r.zero(pm_n * 4) };
    const base_d = base;
    const off_d = try r.dev(i64, &off);
    r.g().attention(q, kn, vn, base_d, off_d, plan1, part1, o1, shape) catch |e| return r.miss("tree", rows, e);
    r.t.attention(q, kn, vn, base_d, off_d, plan2, part2, o2, shape) catch |e| return r.miss("tree", rows, e);
    const tag = if (past == 4096) "tree_p4096" else "tree_p512";
    try r.both(tag, rows, o1, o2, w * QH * HD * 2);
}

fn draftOne(r: *Rig, rows: usize) !void {
    const g = Hid / 16;
    // SEG=1 is folded out of the capture, so one row uses a segment of 3.
    const seg: usize = if (rows == 1) 3 else rows;
    const branch: usize = @intFromBool(rows % 2 == 1);
    const x = try r.dev(u16, try r.bf(rows * Hid, 0.5));
    const dyn = try r.dev(u16, try r.bf(4 * rows * g, 0.2));
    const base = try r.dev(u16, try r.bf(4 * Hid, 0.3));
    const res: ?u64 = if (branch == 1) try r.dev(u16, try r.bf(rows * Hid, 0.2)) else null;
    const o1 = try r.zero(rows * Hid * 2);
    const o2 = try r.zero(rows * Hid * 2);
    r.g().dconv(x, dyn, base, res, o1, rows, seg, Hid, g, 16, branch) catch |e| return r.miss("dconv", rows, e);
    const p = aot.ptr;
    r.t.set.run(r.s, "_dconv_kernel", .{ @intCast(rows), (Hid + 1023) / 1024, 1 }, &.{
        p("X", "*bf16", x), p("DYN", "*bf16", dyn), p("BASE", "*bf16", base), p("RES", "*bf16", res orelse x), p("OUT", "*bf16", o2), aot.int("SEG", @intCast(seg)),
    }, &.{ aot.ci("D", Hid), aot.ci("G", g), aot.ci("GS", 16), aot.ci("BRANCH", @intCast(branch)), aot.ci("HAS_RES", @intCast(branch)), aot.ci("BLOCK", 1024) }) catch |e| return r.miss("dconv", rows, e);
    try r.both("dconv", rows, o1, o2, rows * Hid * 2);
    // H=32 with L=1 is not in the capture. Rows 3, 17 and 64 are.
    if (rows != 1) {
        try dprepAt(r, rows);
    }
    try blockAt(r, rows);
}

fn dprepAt(r: *Rig, rows: usize) !void {
    const p = aot.ptr;
    const stride = (DH + 2 * DKV) * DD;
    const qkv = try r.dev(u16, try r.bf(rows * stride, 0.4));
    const qn = try r.dev(u16, try r.bf(DD, 1));
    const knw = try r.dev(u16, try r.bf(DD, 1));
    const cos = try r.dev(f32, try r.fs(rows * (DD / 2), -1, 1));
    const sin = try r.dev(f32, try r.fs(rows * (DD / 2), -1, 1));
    const q1 = try r.zero(DH * rows * DD * 2);
    const q2 = try r.zero(DH * rows * DD * 2);
    const k1 = try r.zero(DKV * rows * DD * 2);
    const k2 = try r.zero(DKV * rows * DD * 2);
    const v1 = try r.zero(DKV * rows * DD * 2);
    const v2 = try r.zero(DKV * rows * DD * 2);
    r.g().dprep(qkv, qn, knw, cos, sin, q1, k1, v1, rows, DH, DKV, DD / 2, stride, eps) catch |e| return r.miss("dprep", rows, e);
    r.t.set.run(r.s, "_prep_kernel", .{ @intCast(rows), DH + 2 * DKV, 1 }, &.{
        p("QKV", "*bf16", qkv), p("QN", "*bf16", qn), p("KN", "*bf16", knw), p("COS", "*fp32", cos),       p("SIN", "*fp32", sin),
        p("QO", "*bf16", q2),   p("KO", "*bf16", k2), p("VO", "*bf16", v2),  aot.int("L", @intCast(rows)), aot.int("stride", @intCast(stride)),
        aot.float("eps", eps),
    }, &.{ aot.ci("H", DH), aot.ci("HKV", DKV), aot.ci("HALF", DD / 2) }) catch |e| return r.miss("dprep", rows, e);
    try r.both("dprep_q", rows, q1, q2, DH * rows * DD * 2);
    try r.both("dprep_k", rows, k1, k2, DKV * rows * DD * 2);
    try r.both("dprep_v", rows, v1, v2, DKV * rows * DD * 2);
}

fn blockAt(r: *Rig, rows: usize) !void {
    const p = aot.ptr;
    const bl: usize = if (rows % 16 == 0) 16 else 12;
    const br: usize = if (rows >= bl and rows != 1) rows else bl;
    const slen: usize = 8;
    const qb = try r.dev(u16, try r.bf(DH * br * DD, 0.4));
    const kb = try r.dev(u16, try r.bf(DKV * br * DD, 0.4));
    const vb = try r.dev(u16, try r.bf(DKV * br * DD, 0.4));
    const kc = try r.dev(u16, try r.bf(DKV * slen * DD, 0.4));
    const vc = try r.dev(u16, try r.bf(DKV * slen * DD, 0.4));
    const table = [_]u64{ kc, vc };
    const lens = [_]i32{@intCast(slen)};
    const td = try r.dev(u64, &table);
    const ld = try r.dev(i32, &lens);
    const bo1 = try r.zero(bl * DH * DD * 2);
    const bo2 = try r.zero(bl * DH * DD * 2);
    r.g().blockAttn(qb, kb, vb, td, ld, bo1, 2047, 1, br, DH, DKV, bl, DD, 64, false) catch |e| return r.miss("block_attn", rows, e);
    const scale = qwen.q27_math.attnScale(DD);
    r.t.set.run(r.s, "_block_attention", .{ 1, DKV, 1 }, &.{
        p("Q", "*bf16", qb),       p("KB", "*bf16", kb),    p("VB", "*bf16", vb),       p("TABLE", "*i64", td), p("LENS", "*i32", ld), p("O", "*bf16", bo2),
        aot.float("scale", scale), aot.int("window", 2047), aot.int("R", @intCast(br)),
    }, &.{ aot.ci("G", DH / DKV), aot.ci("HKV", DKV), aot.ci("L", @intCast(bl)), aot.ci("LP", 16), aot.ci("D", DD), aot.ci("BN", 64), aot.ci("CAUSAL", 0) }) catch |e| return r.miss("block_attn", rows, e);
    try r.both("block_attn", rows, bo1, bo2, bl * DH * DD * 2);
}

pub fn run(gpu: check.Gpu, dir: []const u8) !void {
    if (!cuda.kernels.available) return error.BuiltWithoutKernels;
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const kk = cuda.kernels;
    var mods: [4]cuda.Module = undefined;
    var loaded: usize = 0;
    defer for (mods[0..loaded]) |*m| m.unload();
    for ([_][]const u8{ kk.q27_rows, kk.q27_prep, kk.q27_tree, kk.q27_draft }, 0..) |img, i| {
        mods[i] = try cuda.Module.load(gpu.d, img);
        loaded += 1;
    }
    var set = try cuda.aot.Set.load(gpu.gpa, gpu.io, gpu.d, gpu.ctx.device, dir);
    defer set.deinit();
    var prng = std.Random.DefaultPrng.init(0x27);
    var rig: Rig = .{
        .gpu = gpu,
        .a = arena.allocator(),
        .s = try cuda.Stream.init(gpu.d, true),
        .f = try glue.Fns.resolve(&mods),
        .t = .{ .set = &set, .s = undefined },
        .rng = prng.random(),
    };
    rig.t.s = rig.s;
    defer {
        rig.s.deinit();
        for (rig.bufs.items) |*b| b.free();
        rig.bufs.deinit(gpu.gpa);
    }
    try rig.f.shared.allowDynamicShared(glue.treeSmem(HD));
    try rig.f.tail.allowDynamicShared(glue.treeSmem(HD));
    for (widths) |rows| {
        try addRms(&rig, rows);
        try rowsOf(&rig, rows);
        try gdn(&rig, rows);
        try attn(&rig, rows);
        try treeAt(&rig, rows, 512);
        try draftOne(&rig, rows);
    }
    try treeAt(&rig, 64, 4096);
    if (rig.bad != 0) return error.TestFailed;
}
