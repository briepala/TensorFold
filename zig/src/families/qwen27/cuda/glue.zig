//! The 27B's own glue: q27_rows, q27_prep, q27_tree and q27_draft, by their extern "C" names.

const std = @import("std");
const cuda = @import("cuda");
const tri = @import("triton.zig");
const math = @import("../q27_math.zig");

pub const Gdn = tri.Gdn;
pub const Attn = tri.Attn;
pub const Plan = tri.Tri.Plan;
pub const Partials = tri.Tri.Partials;

/// Dynamic shared memory for one 64-key tile of head dim `d`: keys, values, and the tile's valid flags.
pub fn treeSmem(d: usize) u32 {
    return @intCast(2 * 64 * d * 2 + 64 * 4);
}

/// The four modules, rows then prep then tree then draft.
pub const Fns = struct {
    add_rmsnorm: cuda.Function,
    gated_norm: cuda.Function,
    swiglu: cuda.Function,
    gate_mul: cuda.Function,
    group_sums: cuda.Function,
    embed: cuda.Function,
    b16_embed: cuda.Function,
    bf16_to_f16: cuda.Function,
    gdn_pre: cuda.Function,
    attn_prep: cuda.Function,
    paths: cuda.Function,
    shared: cuda.Function,
    tail: cuda.Function,
    merge: cuda.Function,
    dconv: cuda.Function,
    dprep: cuda.Function,
    block_attn: cuda.Function,

    pub fn resolve(mods: []const cuda.Module) !Fns {
        return .{
            .add_rmsnorm = try mods[0].function("tf_q27_add_rmsnorm"),
            .gated_norm = try mods[0].function("tf_q27_gated_norm"),
            .swiglu = try mods[0].function("tf_q27_swiglu"),
            .gate_mul = try mods[0].function("tf_q27_gate_mul"),
            .group_sums = try mods[0].function("tf_q27_group_sums"),
            .embed = try mods[0].function("tf_q27_embed"),
            .b16_embed = try mods[0].function("tf_q27_b16_embed"),
            .bf16_to_f16 = try mods[0].function("tf_q27_bf16_to_f16"),
            .gdn_pre = try mods[1].function("tf_q27_gdn_pre"),
            .attn_prep = try mods[1].function("tf_q27_attn_prep"),
            .paths = try mods[2].function("tf_q27_paths"),
            .shared = try mods[2].function("tf_q27_shared"),
            .tail = try mods[2].function("tf_q27_tail"),
            .merge = try mods[2].function("tf_q27_merge"),
            .dconv = try mods[3].function("tf_q27_dconv"),
            .dprep = try mods[3].function("tf_q27_dprep"),
            .block_attn = try mods[3].function("tf_q27_block_attn"),
        };
    }
};

fn i(x: usize) c_int {
    return @intCast(x);
}

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

pub const Glue = struct {
    f: *const Fns,
    s: cuda.Stream,

    fn go(g: Glue, f: cuda.Function, grid: [3]usize, block: u32, shared: u32, a: *cuda.Args) !void {
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = block }, .shared = shared }, g.s, a);
    }

    pub fn embed(g: Glue, ids: u64, w: u64, s: u64, b: u64, out: u64, rows: usize, d: usize) !void {
        var a: cuda.Args = .{};
        a.add(ids);
        a.add(w);
        a.add(s);
        a.add(b);
        a.add(out);
        a.add(i(d));
        try g.go(g.f.embed, .{ rows, 1, 1 }, 1, 0, &a);
    }

    pub fn b16Embed(g: Glue, ids: u64, table: u64, out: u64, rows: usize, d: usize) !void {
        var a: cuda.Args = .{};
        a.add(ids);
        a.add(table);
        a.add(out);
        a.add(i(d));
        try g.go(g.f.b16_embed, .{ rows, 1, 1 }, 32, 0, &a);
    }

    pub fn bf16ToF16(g: Glue, x: u64, y: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(y);
        a.add(@as(i64, @intCast(n)));
        try g.go(g.f.bf16_to_f16, .{ @min(cdiv(n, 256), 4096), 1, 1 }, 256, 0, &a);
    }

    pub fn addRmsnorm(g: Glue, x: u64, r: ?u64, w: u64, h: u64, y: u64, xs: u64, rows: usize, d: usize, eps: f32) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(r orelse 0);
        a.add(w);
        a.add(h);
        a.add(y);
        a.add(xs);
        a.add(eps);
        a.add(i(d));
        a.add(@as(c_int, @intFromBool(r != null)));
        // 256 threads: the width-5120 reduction. Any other width ignores every thread but 0.
        try g.go(g.f.add_rmsnorm, .{ rows, 1, 1 }, 256, 0, &a);
    }

    pub fn gdnPre(g: Glue, qkv: u64, cs: u64, cw: u64, win: u64, sid: ?u64, a: u64, b: u64, alog: u64, dtb: u64, q: u64, k: u64, v: u64, gv: u64, beta: u64, rows: usize, shape: Gdn, ab_f16: bool) !void {
        var args: cuda.Args = .{};
        args.add(qkv);
        args.add(cs);
        args.add(cw);
        args.add(win);
        args.add(a);
        args.add(b);
        args.add(alog);
        args.add(dtb);
        args.add(q);
        args.add(k);
        args.add(v);
        args.add(gv);
        args.add(beta);
        args.add(sid orelse 0);
        args.add(i(shape.c));
        args.add(i(shape.kh));
        args.add(i(shape.vh));
        args.add(i(shape.dk));
        args.add(i(shape.keep));
        args.add(@as(c_int, @intFromBool(sid != null)));
        args.add(@as(c_int, @intFromBool(ab_f16)));
        // 64 threads when the head dim is the captured 128-wide pair reduction.
        try g.go(g.f.gdn_pre, .{ rows, 2 * shape.kh + shape.vh, 1 }, if (shape.dk == 128) 64 else 1, 0, &args);
    }

    pub fn gatedNorm(g: Glue, y: u64, z: u64, w: u64, out: u64, xs: u64, rows: usize, vh: usize, dv: usize, eps: f32) !void {
        var a: cuda.Args = .{};
        a.add(y);
        a.add(z);
        a.add(w);
        a.add(out);
        a.add(xs);
        a.add(eps);
        a.add(i(vh));
        a.add(i(dv));
        // 32 threads when the head is the captured 128-wide butterfly. Any other width uses thread 0.
        try g.go(g.f.gated_norm, .{ rows, vh, 1 }, if (dv == 128) 32 else 1, 0, &a);
    }

    pub fn swiglu(g: Glue, gate: u64, up: u64, out: u64, xs: u64, rows: usize, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(gate);
        a.add(up);
        a.add(out);
        a.add(xs);
        a.add(i(n));
        try g.go(g.f.swiglu, .{ rows, 1, 1 }, 1, 0, &a);
    }

    pub fn attnPrep(g: Glue, qg: u64, kv: u64, qn: u64, kn: u64, pos: u64, inv: u64, qout: u64, kout: u64, rows: usize, shape: Attn, eps: f32) !void {
        return g.attnPrepAt(qg, kv, qn, kn, pos, inv, qout, kout, rows, shape, eps, 0, 11, 10);
    }

    pub fn attnPrepAt(g: Glue, qg: u64, kv: u64, qn: u64, kn: u64, pos: u64, inv: u64, qout: u64, kout: u64, rows: usize, shape: Attn, eps: f32, mrope_rows: usize, hsec: usize, wsec: usize) !void {
        var a: cuda.Args = .{};
        a.add(qg);
        a.add(kv);
        a.add(qn);
        a.add(kn);
        a.add(pos);
        a.add(inv);
        a.add(qout);
        a.add(kout);
        a.add(eps);
        a.add(i(shape.heads));
        a.add(i(shape.kv_heads));
        a.add(i(shape.dim));
        a.add(i(shape.half));
        a.add(i(mrope_rows));
        a.add(i(hsec));
        a.add(i(wsec));
        try g.go(g.f.attn_prep, .{ rows, shape.heads + shape.kv_heads, 1 }, 1, 0, &a);
    }

    pub fn gateMul(g: Glue, o: u64, qg: u64, out: u64, xs: u64, rows: usize, heads: usize, dim: usize) !void {
        var a: cuda.Args = .{};
        a.add(o);
        a.add(qg);
        a.add(out);
        a.add(xs);
        a.add(i(heads));
        a.add(i(dim));
        try g.go(g.f.gate_mul, .{ rows, heads, 1 }, 1, 0, &a);
    }

    pub fn groupSums(g: Glue, x: u64, xs: u64, ld: usize, rows: usize, k: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(xs);
        a.add(i(ld));
        a.add(i(k));
        try g.go(g.f.group_sums, .{ rows, 1, 1 }, 1, 0, &a);
    }

    pub fn paths(g: Glue, parents: u64, out: u64, depths: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(parents);
        a.add(out);
        a.add(depths);
        a.add(i(rows));
        try g.go(g.f.paths, .{ cdiv(rows, 128), 1, 1 }, 128, 0, &a);
    }

    pub fn attention(g: Glue, q: u64, kn: u64, vn: u64, base: u64, offs: u64, plan: Plan, part: Partials, out: u64, shape: Attn) !void {
        const grp = shape.heads / shape.kv_heads;
        const scale = math.attnScale(shape.dim);
        const w = plan.width;
        const smem = treeSmem(shape.dim);
        if (plan.n_items > 0) {
            var a: cuda.Args = .{};
            a.add(q);
            a.add(base);
            a.add(offs);
            a.add(plan.streams);
            a.add(plan.items);
            a.add(part.o);
            a.add(part.m);
            a.add(part.l);
            a.add(i(w));
            a.add(i(shape.heads));
            a.add(i(shape.kv_heads));
            a.add(i(grp));
            a.add(i(shape.dim));
            a.add(scale);
            try g.go(g.f.shared, .{ plan.n_items * shape.kv_heads, 1, 1 }, 512, smem, &a);
        }
        const tails = 1 + (tri.max_nodes + tri.chunk - 1) / tri.chunk;
        var b: cuda.Args = .{};
        b.add(q);
        b.add(kn);
        b.add(vn);
        b.add(base);
        b.add(offs);
        b.add(plan.streams);
        b.add(plan.rows);
        b.add(plan.paths);
        b.add(plan.depths);
        b.add(part.o);
        b.add(part.m);
        b.add(part.l);
        b.add(i(w));
        b.add(i(shape.heads));
        b.add(i(shape.kv_heads));
        b.add(i(grp));
        b.add(i(shape.dim));
        b.add(scale);
        try g.go(g.f.tail, .{ w, shape.kv_heads, tails }, 512, smem, &b);
        var c: cuda.Args = .{};
        c.add(part.o);
        c.add(part.m);
        c.add(part.l);
        c.add(out);
        c.add(plan.streams);
        c.add(plan.rows);
        c.add(i(w));
        c.add(i(shape.heads));
        c.add(i(shape.kv_heads));
        c.add(i(grp));
        c.add(i(shape.dim));
        try g.go(g.f.merge, .{ w, shape.kv_heads, 1 }, 512, 0, &c);
    }

    pub fn dconv(g: Glue, in: u64, dyn: u64, base: u64, res: ?u64, out: u64, rows: usize, seg: usize, d: usize, groups: usize, gs: usize, branch: usize) !void {
        var a: cuda.Args = .{};
        a.add(in);
        a.add(dyn);
        a.add(base);
        a.add(res orelse 0);
        a.add(out);
        a.add(i(seg));
        a.add(i(d));
        a.add(i(groups));
        a.add(i(gs));
        a.add(i(branch));
        a.add(@as(c_int, @intFromBool(res != null)));
        try g.go(g.f.dconv, .{ rows, 1, 1 }, 1, 0, &a);
    }

    pub fn dprep(g: Glue, qkv: u64, qn: u64, kn: u64, cos: u64, sin: u64, qo: u64, ko: u64, vo: u64, rows: usize, h: usize, hkv: usize, half: usize, stride: usize, eps: f32) !void {
        var a: cuda.Args = .{};
        a.add(qkv);
        a.add(qn);
        a.add(kn);
        a.add(cos);
        a.add(sin);
        a.add(qo);
        a.add(ko);
        a.add(vo);
        a.add(eps);
        a.add(i(h));
        a.add(i(hkv));
        a.add(i(half));
        a.add(i(rows));
        a.add(i(stride));
        // 32 threads for the captured half of 64. A shorter head keeps the one-thread loop.
        try g.go(g.f.dprep, .{ rows, h + 2 * hkv, 1 }, if (half == 64) 32 else 1, 0, &a);
    }

    pub fn blockAttn(g: Glue, q: u64, kb: u64, vb: u64, table: u64, lens: u64, out: u64, window: usize, streams: usize, R: usize, query_heads: usize, hkv: usize, L: usize, dim: usize, bn: usize, causal: bool) !void {
        const scale = math.attnScale(dim);
        var a: cuda.Args = .{};
        a.add(q);
        a.add(kb);
        a.add(vb);
        a.add(table);
        a.add(lens);
        a.add(out);
        a.add(scale);
        a.add(i(window));
        a.add(i(R));
        a.add(i(query_heads / hkv));
        a.add(i(hkv));
        a.add(i(L));
        a.add(i(dim));
        a.add(i(bn));
        a.add(@as(c_int, @intFromBool(causal)));
        try g.go(g.f.block_attn, .{ streams, hkv, L }, 1, 0, &a);
    }
};

test "tree shared memory at head dim 256 is the two tiles plus flags" {
    try std.testing.expectEqual(@as(u32, 65792), treeSmem(256));
}
