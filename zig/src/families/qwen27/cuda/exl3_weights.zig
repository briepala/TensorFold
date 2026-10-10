//! The 27B's EXL3 weights on the GPU, filling the family's shared Weights: the pack scanned once
//! (every group's width, codebook and scales derived as the format specifies), trellis groups
//! uploaded as stored, bf16 norms and the bf16 embedding table. Projections are `.exl3` QLinears,
//! so the engine, forward, lanes and drafter stay format-blind.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const source = @import("nemotron").source;
const c = @import("shape.zig");
const kern = @import("kernels.zig");
const wts = @import("weights.zig");
const exl3 = cuda.exl3;
const Tensor = core.checkpoint.Tensor;

/// The 27B's EXL3 packs store the text model under `model.language_model.`.
const prefix = "model.language_model.";

/// The checkpoint as the scanner's View: names and tensors by header name.
fn viewGet(ptr: *anyopaque, name: []const u8) anyerror!exl3.Entry {
    const ck: *core.Checkpoint = @ptrCast(@alignCast(ptr));
    return entryOf(try ck.get(name));
}

fn viewNames(ptr: *anyopaque, a: std.mem.Allocator) anyerror![][]const u8 {
    const ck: *core.Checkpoint = @ptrCast(@alignCast(ptr));
    var out: std.ArrayList([]const u8) = .empty;
    errdefer out.deinit(a);
    for (ck.files.items) |*f| {
        var it = f.names.iterator();
        while (it.next()) |e| try out.append(a, e.key_ptr.*);
    }
    return out.toOwnedSlice(a);
}

/// A checkpoint Tensor as the scan module's Entry.
fn entryOf(t: Tensor) !exl3.Entry {
    const dt: exl3.DType = switch (t.dtype) {
        .f16 => .f16,
        .i16 => .i16,
        else => .other,
    };
    var e: exl3.Entry = .{ .dtype = dt, .shape = .{ 1, 1, 1 }, .rank = 1, .bytes = t.bytes };
    switch (t.shape.len) {
        1 => e.shape[0] = t.shape[0],
        2 => {
            e.shape[0] = t.shape[1];
            e.shape[1] = t.shape[0];
        },
        3 => e.shape = .{ t.shape[2], t.shape[1], t.shape[0] },
        else => return error.BadTrellisShape,
    }
    e.rank = t.shape.len;
    return e;
}

const Loader = struct {
    gpa: std.mem.Allocator,
    ops: kern.Ops,
    w: *wts.Weights,
    src: *source.Source,
    lin: *const exl3.Linear,
    arena: std.mem.Allocator,
    host: std.ArrayList(u8) = .empty,

    fn alloc(L: *Loader, len: usize) !u64 {
        const b = try cuda.DeviceBuffer.alloc(L.ops.k.d, len);
        try L.w.buffers.append(L.gpa, b);
        return b.ptr;
    }

    fn staging(L: *Loader, len: usize) ![]u8 {
        try L.src.flush();
        try L.ops.s.synchronize();
        try L.host.resize(L.gpa, len);
        return L.host.items;
    }

    fn raw(L: *Loader, t: Tensor) !u64 {
        const ptr = try L.alloc(t.bytes.len);
        try L.src.upload(ptr, t.bytes);
        return ptr;
    }

    fn widen(L: *Loader, t: Tensor) !u64 {
        if (t.dtype == .f32) return L.raw(t);
        if (t.dtype != .bf16) return error.UnexpectedTensor;
        const n = t.bytes.len / 2;
        const out = std.mem.bytesAsSlice(u32, try L.staging(n * 4));
        const in = try L.src.view(t.bytes);
        for (out, 0..) |*o, i| o.* = @as(u32, std.mem.readInt(u16, in[2 * i ..][0..2], .little)) << 16;
        const ptr = try L.alloc(n * 4);
        try L.ops.upload(ptr, L.host.items);
        return ptr;
    }

    fn vector(L: *Loader, ck: *core.Checkpoint, name: []const u8, len: usize) !u64 {
        return L.raw(try ck.expect(name, .bf16, &.{len}));
    }

    /// One scanned group uploaded as stored; the QLinear carries the launching module.
    fn group(L: *Loader, g: exl3.Group) !kern.QLinear {
        const w = try exl3.upload(L.ops.k.d, L.arena, g);
        try L.w.buffers.append(L.gpa, w.trellis);
        try L.w.buffers.append(L.gpa, w.suh);
        try L.w.buffers.append(L.gpa, w.svh);
        if (w.bias.ptr != 0) try L.w.buffers.append(L.gpa, w.bias);
        return .{ .exl3 = .{ .w = w, .lin = L.lin } };
    }

    fn delta(L: *Loader, ck: *core.Checkpoint, by_name: *std.StringHashMap(exl3.Group), p: []const u8, g: c.Geometry) !wts.Delta {
        var b1: [224]u8 = undefined;
        var d: wts.Delta = undefined;
        d.qkv = try L.group(by_name.get(try std.fmt.bufPrint(&b1, "{s}linear_attn.in_proj_qkv", .{p})).?);
        d.z = try L.group(by_name.get(try std.fmt.bufPrint(&b1, "{s}linear_attn.in_proj_z", .{p})).?);
        d.b = try L.group(by_name.get(try std.fmt.bufPrint(&b1, "{s}linear_attn.in_proj_b", .{p})).?);
        d.a = try L.group(by_name.get(try std.fmt.bufPrint(&b1, "{s}linear_attn.in_proj_a", .{p})).?);
        d.out = try L.group(by_name.get(try std.fmt.bufPrint(&b1, "{s}linear_attn.out_proj", .{p})).?);
        d.conv = try L.raw(try ck.expect(try std.fmt.bufPrint(&b1, "{s}linear_attn.conv1d.weight", .{p}), .bf16, &.{ g.convDim(), c.conv_taps, 1 }));
        d.a_log = try L.widen(try ck.get(try std.fmt.bufPrint(&b1, "{s}linear_attn.A_log", .{p})));
        d.dt_bias = try L.widen(try ck.get(try std.fmt.bufPrint(&b1, "{s}linear_attn.dt_bias", .{p})));
        d.norm = try L.vector(ck, try std.fmt.bufPrint(&b1, "{s}linear_attn.norm.weight", .{p}), c.linear_dim);
        return d;
    }

    fn attention(L: *Loader, ck: *core.Checkpoint, by_name: *std.StringHashMap(exl3.Group), p: []const u8, g: c.Geometry) !wts.Attention {
        _ = g; // the attention shapes the scanned groups already carry
        var b1: [224]u8 = undefined;
        return .{
            .q = try L.group(by_name.get(try std.fmt.bufPrint(&b1, "{s}self_attn.q_proj", .{p})).?),
            .k = try L.group(by_name.get(try std.fmt.bufPrint(&b1, "{s}self_attn.k_proj", .{p})).?),
            .v = try L.group(by_name.get(try std.fmt.bufPrint(&b1, "{s}self_attn.v_proj", .{p})).?),
            .o = try L.group(by_name.get(try std.fmt.bufPrint(&b1, "{s}self_attn.o_proj", .{p})).?),
            .q_norm = try L.vector(ck, try std.fmt.bufPrint(&b1, "{s}self_attn.q_norm.weight", .{p}), c.head_dim),
            .k_norm = try L.vector(ck, try std.fmt.bufPrint(&b1, "{s}self_attn.k_norm.weight", .{p}), c.head_dim),
        };
    }
};

/// The EXL3 load: fills the shared Weights so every engine path stays format-blind.
pub fn load(gpa: std.mem.Allocator, io: std.Io, ops: kern.Ops, lin: *const exl3.Linear, dir: []const u8, cfg: c.Config) !wts.Weights {
    const g = cfg.g;
    var w: wts.Weights = .{ .gpa = gpa, .g = g, .eos = cfg.eos, .eos_count = cfg.eos_count };
    errdefer w.deinit();
    w.scales_arena = std.heap.ArenaAllocator.init(gpa);
    var src = try source.Source.init(gpa, .{ .d = ops.k.d, .s = ops.s, .discrete = ops.k.discrete });
    defer src.deinit();
    var L: Loader = .{ .gpa = gpa, .ops = ops, .w = &w, .src = &src, .lin = lin, .arena = w.scales_arena.allocator() };
    defer {
        src.flush() catch {};
        ops.s.synchronize() catch {};
        L.host.deinit(gpa);
    }
    var ck = try core.Checkpoint.openModelPrefix(gpa, io, dir, null);
    defer ck.close();
    try src.add(&ck);

    // the pack scanned once: every group's width, codebook and scales by the format's own rules
    const groups = try exl3.scan(gpa, .{ .ptr = &ck, .getFn = viewGet, .namesFn = viewNames });
    defer gpa.free(groups);
    var by_name = std.StringHashMap(exl3.Group).init(gpa);
    defer by_name.deinit();
    for (groups) |gr| try by_name.put(gr.prefix, gr);

    w.layers = try gpa.alloc(wts.Layer, g.layers);
    for (w.layers, 0..) |*layer, i| {
        var pb: [64]u8 = undefined;
        var nb: [224]u8 = undefined;
        const p = try std.fmt.bufPrint(&pb, "model.layers.{d}.", .{i});
        layer.input_norm = try L.vector(&ck, try std.fmt.bufPrint(&nb, "{s}input_layernorm.weight", .{p}), g.hidden);
        layer.mixer = if (c.linear(i)) .{ .delta = try L.delta(&ck, &by_name, p, g) } else .{ .attention = try L.attention(&ck, &by_name, p, g) };
        layer.gate = try L.group(by_name.get(try std.fmt.bufPrint(&nb, "{s}mlp.gate_proj", .{p})).?);
        layer.up = try L.group(by_name.get(try std.fmt.bufPrint(&nb, "{s}mlp.up_proj", .{p})).?);
        layer.down = try L.group(by_name.get(try std.fmt.bufPrint(&nb, "{s}mlp.down_proj", .{p})).?);
        layer.post_norm = try L.vector(&ck, try std.fmt.bufPrint(&nb, "{s}post_attention_layernorm.weight", .{p}), g.hidden);
    }
    w.embed = .{ .b16 = .{ .w = try L.raw(try ck.expect(prefix ++ "model.embed_tokens.weight", .bf16, &.{ c.vocab, g.hidden })), .d = g.hidden } };
    w.norm = try L.vector(&ck, "model.norm.weight", g.hidden);
    w.head = try L.group(by_name.get("lm_head").?);
    if (ck.unused() != 0) return error.UnusedCheckpointTensors;
    var inv: [c.rotary_dim / 2]f32 = undefined;
    wts.invFreq(&inv);
    w.inv_freq = try L.alloc(@sizeOf(@TypeOf(inv)));
    try ops.upload(w.inv_freq, std.mem.asBytes(&inv));
    try src.flush();
    try ops.s.synchronize();
    return w;
}
