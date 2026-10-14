//! The 27B glue kernels against their host references. Every output byte, at 1, 3, 17 and 64 rows.

const std = @import("std");
const cuda = @import("cuda");
const qwen = @import("qwen27");
const check = @import("check.zig");
const gm = @import("nemotron").glue_math;

const glue = qwen.glue;
const ref = qwen.glue_ref;
const tree = qwen.glue_tree_ref;
const draft = qwen.glue_draft_ref;
const tri = qwen.triton;

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

    fn cmp(r: *Rig, name: []const u8, rows: usize, ptr: u64, want: anytype, proof: bool) !void {
        const raw = std.mem.sliceAsBytes(want);
        try r.s.synchronize();
        const got = try r.a.alloc(u8, raw.len);
        try r.gpu.d.check(r.gpu.d.api.cuMemcpyDtoH_v2(got.ptr, ptr, raw.len), "cuMemcpyDtoH");
        var n: usize = 0;
        for (got, raw) |x, y| n += @intFromBool(x != y);
        const dead = proof and std.mem.allEqual(u8, raw, 0);
        if (dead or n != 0) {
            r.bad += 1;
            std.debug.print("FAIL {s} rows={d} mismatches={d}\n", .{ name, rows, if (dead) raw.len else n });
            return;
        }
        std.debug.print("PASS {s} rows={d} mismatches=0\n", .{ name, rows });
    }
};

fn addRms(r: *Rig, rows: usize) !void {
    const gl = r.g();
    const with_r = rows % 2 == 1;
    const x = try r.bf(rows * Hid, 1.5);
    const res = try r.bf(rows * Hid, 0.5);
    const w = try r.bf(Hid, 1);
    const h = try r.a.alloc(u16, rows * Hid);
    @memset(h, 0x1234);
    const y = try r.a.alloc(u16, rows * Hid);
    const xs = try r.a.alloc(f32, rows * Hid / 64);
    ref.addRmsnorm(x, if (with_r) res else null, w, h, y, xs, rows, Hid, eps);
    const xd = try r.dev(u16, x);
    const hd = try r.dev(u16, h);
    const yd = try r.zero(y.len * 2);
    const xsd = try r.zero(xs.len * 4);
    try gl.addRmsnorm(xd, if (with_r) try r.dev(u16, res) else null, try r.dev(u16, w), if (with_r) hd else xd, yd, xsd, rows, Hid, eps);
    if (with_r) try r.cmp("add_rmsnorm_h", rows, hd, h, true);
    try r.cmp("add_rmsnorm_y", rows, yd, y, true);
    try r.cmp("add_rmsnorm_xs", rows, xsd, xs, true);
}

fn gated(r: *Rig, rows: usize) !void {
    const y = try r.bf(rows * VH * DK, 1);
    const z = try r.bf(rows * VH * DK, 1);
    const w = try r.bf(DK, 1);
    const out = try r.a.alloc(u16, y.len);
    const xs = try r.a.alloc(f32, rows * VH * DK / 64);
    ref.gatedNorm(y, z, w, out, xs, rows, VH, DK, eps);
    const od = try r.zero(out.len * 2);
    const xsd = try r.zero(xs.len * 4);
    try r.g().gatedNorm(try r.dev(u16, y), try r.dev(u16, z), try r.dev(u16, w), od, xsd, rows, VH, DK, eps);
    try r.cmp("gated_norm", rows, od, out, true);
    try r.cmp("gated_norm_xs", rows, xsd, xs, true);
}

fn swiglu(r: *Rig, rows: usize) !void {
    const gate = try r.bf(rows * Mid, 1);
    const up = try r.bf(rows * Mid, 1);
    const out = try r.a.alloc(u16, gate.len);
    const xs = try r.a.alloc(f32, rows * Mid / 64);
    ref.swiglu(gate, up, out, xs, rows, Mid);
    const od = try r.zero(out.len * 2);
    const xsd = try r.zero(xs.len * 4);
    try r.g().swiglu(try r.dev(u16, gate), try r.dev(u16, up), od, xsd, rows, Mid);
    try r.cmp("swiglu", rows, od, out, true);
    try r.cmp("swiglu_xs", rows, xsd, xs, true);
}

fn gateMul(r: *Rig, rows: usize) !void {
    const o = try r.bf(rows * QH * HD, 1);
    const qg = try r.bf(rows * QH * 2 * HD, 1);
    const out = try r.a.alloc(u16, o.len);
    const xs = try r.a.alloc(f32, rows * QH * HD / 64);
    ref.gateMul(o, qg, out, xs, rows, QH, HD);
    const od = try r.zero(out.len * 2);
    const xsd = try r.zero(xs.len * 4);
    try r.g().gateMul(try r.dev(u16, o), try r.dev(u16, qg), od, xsd, rows, QH, HD);
    try r.cmp("gate_mul", rows, od, out, true);
    try r.cmp("gate_mul_xs", rows, xsd, xs, true);
}

fn sums(r: *Rig, rows: usize) !void {
    const x = try r.bf(rows * Hid, 1);
    const xs = try r.a.alloc(f32, rows * Hid / 64);
    ref.groupSumsRows(x, xs, Hid, rows, Hid);
    const xsd = try r.zero(xs.len * 4);
    try r.g().groupSums(try r.dev(u16, x), xsd, Hid, rows, Hid);
    try r.cmp("group_sums", rows, xsd, xs, true);
}

fn embed(r: *Rig, rows: usize) !void {
    const toks = 8;
    const words = try r.a.alloc(u32, toks * Hid / 8);
    for (words) |*w| w.* = r.rng.int(u32);
    const sc = try r.bf(toks * Hid / 64, 0.05);
    const bi = try r.bf(toks * Hid / 64, 0.05);
    const ids = try r.a.alloc(i32, rows);
    for (ids) |*i| i.* = r.rng.intRangeLessThan(i32, 0, toks);
    const out = try r.a.alloc(u16, rows * Hid);
    ref.embed(ids, words, sc, bi, out, Hid);
    const od = try r.zero(out.len * 2);
    try r.g().embed(try r.dev(i32, ids), try r.dev(u32, words), try r.dev(u16, sc), try r.dev(u16, bi), od, rows, Hid);
    try r.cmp("embed", rows, od, out, true);
    const table = try r.bf(toks * Hid, 1);
    const b16 = try r.a.alloc(u16, rows * Hid);
    ref.b16Embed(ids, table, b16, Hid);
    const bd = try r.zero(b16.len * 2);
    try r.g().b16Embed(try r.dev(i32, ids), try r.dev(u16, table), bd, rows, Hid);
    try r.cmp("b16_embed", rows, bd, b16, true);
}

fn cast(r: *Rig, rows: usize) !void {
    const x = try r.bf(rows * Hid, 2);
    const y = try r.a.alloc(f16, x.len);
    ref.bf16ToF16(x, y);
    const yd = try r.zero(y.len * 2);
    try r.g().bf16ToF16(try r.dev(u16, x), yd, x.len);
    try r.cmp("bf16_to_f16", rows, yd, y, true);
}

fn gdn(r: *Rig, rows: usize) !void {
    const multi = rows == 3 or rows == 64;
    const ab_f16 = rows == 3 or rows == 17;
    const taps = Keep + 1;
    const qkv = try r.bf(rows * Conv, 0.4);
    const cs_rows: usize = if (multi) 2 else 1;
    const cs = try r.bf(cs_rows * Keep * Conv, 0.4);
    const cw = try r.bf(Conv * taps, 0.3);
    const win = try r.a.alloc(i32, rows * taps);
    for (0..rows) |row| {
        for (0..Keep) |j| win[row * taps + j] = @intCast(j);
        win[row * taps + Keep] = @intCast(Keep + row);
    }
    const ab = if (ab_f16) try r.f16b(rows * VH) else try r.bf(rows * VH, 0.4);
    const bb = if (ab_f16) try r.f16b(rows * VH) else try r.bf(rows * VH, 0.4);
    const alog = try r.fs(VH, -1, 1);
    const dtb = try r.fs(VH, -0.5, 0.5);
    const sid_buf: ?[]i32 = if (!multi) null else blk: {
        const s = try r.a.alloc(i32, rows);
        for (s, 0..) |*v, i| v.* = @intCast(i % 2);
        break :blk s;
    };
    const q = try r.a.alloc(u16, rows * KH * DK);
    const k = try r.a.alloc(u16, rows * KH * DK);
    const v = try r.a.alloc(u16, rows * VH * DK);
    const gv = try r.a.alloc(f32, rows * VH);
    const beta = try r.a.alloc(f32, rows * VH);
    ref.gdnPre(qkv, cs, cw, win, ab, bb, alog, dtb, q, k, v, gv, beta, sid_buf, rows, Conv, KH, VH, DK, Keep, ab_f16);
    const shape: glue.Gdn = .{ .c = Conv, .kh = KH, .vh = VH, .dk = DK, .keep = Keep };
    const qd = try r.zero(q.len * 2);
    const kd = try r.zero(k.len * 2);
    const vd = try r.zero(v.len * 2);
    const gd = try r.zero(gv.len * 4);
    const bd = try r.zero(beta.len * 4);
    try r.g().gdnPre(try r.dev(u16, qkv), try r.dev(u16, cs), try r.dev(u16, cw), try r.dev(i32, win), if (sid_buf) |s| try r.dev(i32, s) else null, try r.dev(u16, ab), try r.dev(u16, bb), try r.dev(f32, alog), try r.dev(f32, dtb), qd, kd, vd, gd, bd, rows, shape, ab_f16);
    try r.cmp("gdn_q", rows, qd, q, true);
    try r.cmp("gdn_k", rows, kd, k, true);
    try r.cmp("gdn_v", rows, vd, v, true);
    try r.cmp("gdn_g", rows, gd, gv, true);
    try r.cmp("gdn_beta", rows, bd, beta, true);
}

fn invFreq(a: std.mem.Allocator, half: usize) ![]f32 {
    const o = try a.alloc(f32, half);
    for (o, 0..) |*v, i| v.* = @floatCast(std.math.pow(f64, 1e7, -@as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(half))));
    return o;
}

fn attn(r: *Rig, rows: usize) !void {
    const mrope = rows == 3 or rows == 64;
    const qg = try r.bf(rows * QH * 2 * HD, 0.5);
    const kv = try r.bf(rows * KV * HD, 0.5);
    const qn = try r.bf(HD, 1);
    const kn = try r.bf(HD, 1);
    const pos = try r.a.alloc(i32, if (mrope) rows * 3 else rows);
    for (pos, 0..) |*p, i| p.* = @intCast((i * 17 + 3) % 500);
    const inv = try invFreq(r.a, Half);
    const qo = try r.a.alloc(u16, rows * QH * HD);
    const ko = try r.a.alloc(u16, rows * KV * HD);
    ref.attnPrep(qg, kv, qn, kn, pos, inv, qo, ko, rows, QH, KV, HD, Half, eps, if (mrope) rows else 0, 11, 10);
    const qod = try r.zero(qo.len * 2);
    const kod = try r.zero(ko.len * 2);
    const shape: glue.Attn = .{ .heads = QH, .kv_heads = KV, .dim = HD, .half = Half };
    const pd = try r.dev(i32, pos);
    const id = try r.dev(f32, inv);
    const qgd = try r.dev(u16, qg);
    const kvd = try r.dev(u16, kv);
    const qnd = try r.dev(u16, qn);
    const knd = try r.dev(u16, kn);
    if (mrope) try r.g().attnPrepAt(qgd, kvd, qnd, knd, pd, id, qod, kod, rows, shape, eps, rows, 11, 10) else try r.g().attnPrep(qgd, kvd, qnd, knd, pd, id, qod, kod, rows, shape, eps);
    try r.cmp("attn_q", rows, qod, qo, true);
    try r.cmp("attn_k", rows, kod, ko, true);
}

fn oneTree(r: *Rig, rows: usize, past: usize, tag: []const u8) !void {
    const w = rows;
    const group = QH / KV;
    const nslots = tri.slots(past, w);
    const parents = try r.a.alloc(i32, w);
    for (parents, 0..) |*p, i| p.* = if (i == 0) -1 else @intCast(i - 1);
    const path = try r.a.alloc(i32, w * 128);
    const depths = try r.a.alloc(i32, w);
    tree.paths(parents, path, depths, w);
    const path_d = try r.zero(path.len * 4);
    const dep_d = try r.zero(depths.len * 4);
    try r.g().paths(try r.dev(i32, parents), path_d, dep_d, w);
    try r.cmp("paths", rows, path_d, path, false);
    try r.cmp("paths_depth", rows, dep_d, depths, true);
    var items: std.ArrayList(i32) = .empty;
    const folded = tri.groups(past, w);
    const loose = past / tri.chunk - folded * tri.group_chunks;
    for (0..folded + loose) |j| {
        const code: i32 = if (j < folded) @intCast(j) else -1 - @as(i32, @intCast(j - folded));
        var first: usize = 0;
        while (first < w * group) : (first += 16) try items.appendSlice(r.a, &.{ 0, @intCast(first), code });
    }
    const stream = [_]i32{ 0, @intCast(w), @intCast(past), @intCast(nslots) };
    const rows_of = try r.a.alloc(i32, w);
    @memset(rows_of, 0);
    const k_elems = past * KV * HD;
    const base = try r.bf(2 * past * KV * HD, 0.4);
    const q = try r.bf(w * QH * HD, 0.4);
    const kn = try r.bf(w * KV * HD, 0.4);
    const vn = try r.bf(w * KV * HD, 0.4);
    const off = [_]i64{ 0, @intCast(k_elems) };
    const po_n = nslots * w * QH * HD;
    const pm_n = nslots * w * QH;
    const po = try r.a.alloc(f32, po_n);
    const pm = try r.a.alloc(f32, pm_n);
    const pl = try r.a.alloc(f32, pm_n);
    @memset(po, 0);
    @memset(pm, 0);
    @memset(pl, 0);
    const scale = qwen.q27_math.attnScale(HD);
    const item_s = items.items;
    tree.shared(q, base, &off, &stream, item_s, po, pm, pl, w, QH, KV, HD, scale);
    tree.tail(q, kn, vn, base, &off, &stream, rows_of, path, depths, po, pm, pl, w, QH, KV, HD, scale);
    const out = try r.a.alloc(u16, w * QH * HD);
    tree.merge(po, pm, pl, out, &stream, rows_of, w, QH, KV, HD);
    const pod = try r.zero(po_n * 4);
    const pmd = try r.zero(pm_n * 4);
    const pld = try r.zero(pm_n * 4);
    const outd = try r.zero(out.len * 2);
    const plan: glue.Plan = .{
        .streams = try r.dev(i32, &stream),
        .items = try r.dev(i32, item_s),
        .n_items = item_s.len / 3,
        .rows = try r.dev(i32, rows_of),
        .paths = try r.dev(i32, path),
        .depths = try r.dev(i32, depths),
        .width = w,
    };
    try r.g().attention(try r.dev(u16, q), try r.dev(u16, kn), try r.dev(u16, vn), try r.dev(u16, base), try r.dev(i64, &off), plan, .{ .o = pod, .m = pmd, .l = pld }, outd, .{ .heads = QH, .kv_heads = KV, .dim = HD, .half = Half });
    const name_o = try std.fmt.allocPrint(r.a, "tree_merge_{s}", .{tag});
    const name_p = try std.fmt.allocPrint(r.a, "tree_partial_{s}", .{tag});
    try r.cmp(name_o, rows, outd, out, true);
    try r.cmp(name_p, rows, pod, po, true);
}

fn dconv(r: *Rig, rows: usize) !void {
    const g = Hid / 16;
    const seg: usize = if (rows == 1) 1 else rows;
    const dyn_n = (4 * rows) * g;
    const x = try r.bf(rows * Hid, 0.5);
    const dyn = try r.bf(dyn_n, 0.2);
    const base = try r.bf(4 * Hid, 0.3);
    const res = try r.bf(rows * Hid, 0.2);
    const with = rows % 2 == 1;
    const out = try r.a.alloc(u16, x.len);
    draft.dconv(x, dyn, base, if (with) res else null, out, rows, seg, Hid, g, 16, @intFromBool(with));
    const od = try r.zero(out.len * 2);
    try r.g().dconv(try r.dev(u16, x), try r.dev(u16, dyn), try r.dev(u16, base), if (with) try r.dev(u16, res) else null, od, rows, seg, Hid, g, 16, @intFromBool(with));
    try r.cmp("dconv", rows, od, out, true);
}

fn dprep(r: *Rig, rows: usize) !void {
    const stride = (DH + 2 * DKV) * DD;
    const qkv = try r.bf(rows * stride, 0.4);
    const qn = try r.bf(DD, 1);
    const kn = try r.bf(DD, 1);
    const cos = try r.fs(rows * (DD / 2), -1, 1);
    const sin = try r.fs(rows * (DD / 2), -1, 1);
    const qo = try r.a.alloc(u16, DH * rows * DD);
    const ko = try r.a.alloc(u16, DKV * rows * DD);
    const vo = try r.a.alloc(u16, DKV * rows * DD);
    draft.dprep(qkv, qn, kn, cos, sin, qo, ko, vo, rows, DH, DKV, DD / 2, stride, eps);
    const qod = try r.zero(qo.len * 2);
    const kod = try r.zero(ko.len * 2);
    const vod = try r.zero(vo.len * 2);
    try r.g().dprep(try r.dev(u16, qkv), try r.dev(u16, qn), try r.dev(u16, kn), try r.dev(f32, cos), try r.dev(f32, sin), qod, kod, vod, rows, DH, DKV, DD / 2, stride, eps);
    try r.cmp("dprep_q", rows, qod, qo, true);
    try r.cmp("dprep_k", rows, kod, ko, true);
    try r.cmp("dprep_v", rows, vod, vo, true);
}

fn block(r: *Rig, rows: usize) !void {
    const L = rows;
    const R = L;
    const slen: usize = 8;
    const q = try r.bf(DH * R * DD, 0.4);
    const kb = try r.bf(DKV * R * DD, 0.4);
    const vb = try r.bf(DKV * R * DD, 0.4);
    const kc = try r.bf(DKV * slen * DD, 0.4);
    const vc = try r.bf(DKV * slen * DD, 0.4);
    const lens = [_]i32{@intCast(slen)};
    const kcs = [_][]const u16{kc};
    const vcs = [_][]const u16{vc};
    const out = try r.a.alloc(u16, L * DH * DD);
    const scale = qwen.q27_math.attnScale(DD);
    draft.blockAttn(q, kb, vb, &kcs, &vcs, &lens, out, scale, 2047, R, DH, DKV, L, DD, 64, false);
    const kcd = try r.dev(u16, kc);
    const vcd = try r.dev(u16, vc);
    const table = [_]u64{ kcd, vcd };
    const od = try r.zero(out.len * 2);
    try r.g().blockAttn(try r.dev(u16, q), try r.dev(u16, kb), try r.dev(u16, vb), try r.dev(u64, &table), try r.dev(i32, &lens), od, 2047, 1, R, DH, DKV, L, DD, 64, false);
    try r.cmp("block_attn", rows, od, out, true);
}

pub fn run(gpu: check.Gpu) !void {
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
    var prng = std.Random.DefaultPrng.init(0x27);
    var rig: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = try cuda.Stream.init(gpu.d, true), .f = try glue.Fns.resolve(&mods), .rng = prng.random() };
    defer {
        rig.s.deinit();
        for (rig.bufs.items) |*b| b.free();
        rig.bufs.deinit(gpu.gpa);
    }
    try rig.f.shared.allowDynamicShared(glue.treeSmem(HD));
    try rig.f.tail.allowDynamicShared(glue.treeSmem(HD));
    for (widths) |rows| {
        try addRms(&rig, rows);
        try gated(&rig, rows);
        try swiglu(&rig, rows);
        try gateMul(&rig, rows);
        try sums(&rig, rows);
        try embed(&rig, rows);
        try cast(&rig, rows);
        try gdn(&rig, rows);
        try attn(&rig, rows);
        try oneTree(&rig, rows, 512, "p512");
        try dconv(&rig, rows);
        try dprep(&rig, rows);
        try block(&rig, rows);
    }
    try oneTree(&rig, 64, 4096, "p4096");
    if (rig.bad != 0) return error.TestFailed;
}
