//! EXL3 checkpoint scanning: the groups a safetensors index holds, their widths, codebooks and scales.

const std = @import("std");

pub const Codebook = enum(u8) { inst3 = 0, mcg = 1, mul1 = 2 };

pub const Width = struct {
    /// Whole halves use twice the bit count; mul1 also admits odd half-bit units.
    halves: u16,

    pub fn bits(w: Width) f64 {
        return @as(f64, @floatFromInt(w.halves)) / 2.0;
    }

    /// The tile's 32-bit words: 4 a half (16 * bits int16 words are 8 * bits bytes).
    pub fn tileWords(w: Width) usize {
        return @as(usize, w.halves) * 4;
    }
};

/// A raw safetensors view the scanner reads: names in file order, each with dtype, shape and bytes.
pub const View = struct {
    ptr: *anyopaque,
    getFn: *const fn (ptr: *anyopaque, name: []const u8) anyerror!Entry,
    namesFn: *const fn (ptr: *anyopaque, a: std.mem.Allocator) anyerror![][]const u8,

    pub fn get(v: View, name: []const u8) !Entry {
        return v.getFn(v.ptr, name);
    }

    pub fn names(v: View, a: std.mem.Allocator) ![][]const u8 {
        return v.namesFn(v.ptr, a);
    }
};

/// One tensor the scanner saw: its header facts and a borrowed byte view.
pub const Entry = struct {
    dtype: DType,
    shape: [3]usize,
    rank: usize,
    bytes: []const u8,
};

pub const DType = enum { f16, i16, other };

/// One quantized group's metadata, from the headers alone.
pub const Group = struct {
    prefix: []const u8,
    trellis: Entry,
    suh: Entry,
    svh: Entry,
    bias: ?Entry,
    codebook: Codebook,
    k: usize,
    n: usize,
    width: Width,
};

/// Every EXL3 group in the checkpoint, name order; the plain tensors stay with the caller's view.
pub fn scan(a: std.mem.Allocator, v: View) ![]Group {
    const names_ = try v.names(a);
    defer a.free(names_);
    var out: std.ArrayList(Group) = .empty;
    errdefer out.deinit(a);
    var prefixes: std.StringHashMapUnmanaged(void) = .empty;
    defer prefixes.deinit(a);
    for (names_) |name| {
        if (!std.mem.endsWith(u8, name, ".trellis")) continue;
        const prefix = name[0 .. name.len - ".trellis".len];
        if ((try prefixes.getOrPut(a, prefix)).found_existing) continue;
        const trellis = try v.get(name);
        const codebook: Codebook = codebookOf(v, prefix);
        const width = try widthOf(trellis);
        if (width.halves % 2 == 1 and codebook != .mul1) return error.HalfBitsNeedMul1;
        const suh = try scaleOf(v, prefix, "suh", "su");
        const svh = try scaleOf(v, prefix, "svh", "sv");
        var buf: [512]u8 = undefined;
        const bias_name = std.fmt.bufPrint(&buf, "{s}.bias", .{prefix}) catch return error.NameTooLong;
        const bias = optionalTensor(v, bias_name);
        try out.append(a, .{
            .prefix = prefix,
            .trellis = trellis,
            .suh = suh,
            .svh = svh,
            .bias = bias,
            .codebook = codebook,
            .k = 16 * trellis.shape[0],
            .n = 16 * trellis.shape[1],
            .width = width,
        });
    }
    return out.toOwnedSlice(a);
}

fn codebookOf(v: View, prefix: []const u8) Codebook {
    var buf: [512]u8 = undefined;
    if (markerAt(v, std.fmt.bufPrint(&buf, "{s}.mul1", .{prefix}) catch return .inst3)) return .mul1;
    var buf2: [512]u8 = undefined;
    if (markerAt(v, std.fmt.bufPrint(&buf2, "{s}.mcg", .{prefix}) catch return .inst3)) return .mcg;
    return .inst3;
}

fn markerAt(v: View, name: []const u8) bool {
    const t = v.get(name) catch return false;
    return t.bytes.len == 0;
}

fn widthOf(trellis: Entry) !Width {
    if (trellis.rank != 3 or trellis.shape[0] == 0 or trellis.shape[1] == 0) return error.BadTrellis;
    const words = trellis.shape[2];
    if (words == 0 or words % 8 != 0 or trellis.dtype != .i16) return error.BadTrellis;
    return .{ .halves = @intCast(words / 8) };
}

fn scaleOf(v: View, prefix: []const u8, wide: []const u8, packed_signs: []const u8) !Entry {
    var buf: [512]u8 = undefined;
    const wide_full = std.fmt.bufPrint(&buf, "{s}.{s}", .{ prefix, wide }) catch return error.NameTooLong;
    if (v.get(wide_full)) |t| return t else |_| {}
    var buf2: [512]u8 = undefined;
    const packed_full = std.fmt.bufPrint(&buf2, "{s}.{s}", .{ prefix, packed_signs }) catch return error.NameTooLong;
    return v.get(packed_full) catch error.MissingScale;
}

fn optionalTensor(v: View, name: []const u8) ?Entry {
    return v.get(name) catch null;
}

/// The packed sign words of an older checkpoint expanded to fp16 signs (format.py's unpack_signs).
pub fn unpackSigns(packed_words: []const i16, out: []f16) void {
    for (packed_words, 0..) |word, i| {
        const w: u16 = @bitCast(word);
        const bits = @as(u16, 1) << 15;
        const base: u16 = 0x3C00; // +1.0 in fp16
        for (0..16) |j| {
            out[i * 16 + j] = @bitCast(if ((w >> @intCast(j)) & 1 != 0) base | bits else base);
        }
    }
}

test "unpackSigns makes +1/-1 fp16s" {
    const words = [_]i16{ 0b0000000100000000, -1 };
    var out: [32]f16 = undefined;
    unpackSigns(&words, &out);
    try std.testing.expectEqual(@as(f16, 1.0), out[0]);
    try std.testing.expectEqual(@as(f16, -1.0), out[8]);
    try std.testing.expectEqual(@as(f16, -1.0), out[16]);
    try std.testing.expectEqual(@as(f16, -1.0), out[31]);
}

test "scan reads a small synthetic checkpoint" {
    const a = std.testing.allocator;
    const Fake = struct {
        names: []const []const u8,
        fn get(ptr: *anyopaque, name: []const u8) anyerror!Entry {
            const byte_count = 2 * 8 * 24 * 2;
            _ = byte_count;
            const f: *@This() = @ptrCast(@alignCast(ptr));
            for (f.names) |n| if (std.mem.eql(u8, n, name)) {
                if (std.mem.endsWith(u8, name, ".mul1")) return .{ .dtype = .other, .shape = .{ 0, 0, 0 }, .rank = 0, .bytes = &.{} };
                if (std.mem.endsWith(u8, name, ".trellis")) {
                    // K/16 = 2, N/16 = 8, 16*bits words = 24 (1.5 bits): shape [2, 8, 24]
                    const S = struct {
                        var zeros: [768]u8 = @splat(0);
                    };
                    return .{ .dtype = .i16, .shape = .{ 2, 8, 24 }, .rank = 3, .bytes = &S.zeros };
                }
                if (std.mem.endsWith(u8, name, ".suh") or std.mem.endsWith(u8, name, ".svh")) {
                    const S = struct {
                        var zeros: [64]u8 = @splat(0);
                    };
                    return .{ .dtype = .f16, .shape = .{ 32, 0, 0 }, .rank = 1, .bytes = &S.zeros };
                }
                return .{ .dtype = .other, .shape = .{ 0, 0, 0 }, .rank = 0, .bytes = &.{} };
            };
            return error.MissingTensor;
        }
        fn namesOf(ptr: *anyopaque, alloc: std.mem.Allocator) anyerror![][]const u8 {
            const f: *@This() = @ptrCast(@alignCast(ptr));
            const out = try alloc.alloc([]const u8, f.names.len);
            for (f.names, 0..) |n, i| out[i] = n;
            return out;
        }
    };
    var fake = Fake{ .names = &.{ "model.layers.0.q_proj.trellis", "model.layers.0.q_proj.suh", "model.layers.0.q_proj.svh", "model.layers.0.q_proj.mul1", "model.embed.weight" } };
    const v = View{ .ptr = &fake, .getFn = Fake.get, .namesFn = Fake.namesOf };
    const groups = try scan(a, v);
    defer a.free(groups);
    try std.testing.expectEqual(@as(usize, 1), groups.len);
    try std.testing.expectEqualStrings("model.layers.0.q_proj", groups[0].prefix);
    try std.testing.expectEqual(Codebook.mul1, groups[0].codebook);
    try std.testing.expectEqual(@as(usize, 32), groups[0].k);
    try std.testing.expectEqual(@as(usize, 128), groups[0].n);
    try std.testing.expectEqual(@as(u16, 3), groups[0].width.halves);
}
