//! EXL3 trellis projections on CUDA: the group scan, the device upload and the row-invariant decode launches.

const std = @import("std");
const Driver = @import("driver.zig").Driver;
const Module = @import("module.zig").Module;
const memory = @import("memory.zig");
const stream_ = @import("stream.zig");
const launch = @import("launch.zig");
const kernels = @import("kernels.zig");
const format = @import("exl3_scan.zig");
pub const Codebook = format.Codebook;
pub const Entry = format.Entry;
pub const View = format.View;
pub const DType = format.DType;
pub const Width = format.Width;
pub const Group = format.Group;
pub const scan = format.scan;

/// (K splits, warps a program) for a K x N layer: linear.py's plan, the shape's alone.
pub const Plan = struct { sk: u32, wk: u32 };

pub fn plan(k: usize, n: usize, blocks: usize, min_tiles: usize) Plan {
    const kt = k / 16;
    const nb = n / 128;
    var sk: u32 = 1;
    var wk: u32 = if (nb >= 64) 8 else 4;
    if (nb < 64) {
        while (nb * sk < blocks and sk < 64) sk *= 2;
    }
    while ((wk > 4 or sk > 1) and (kt % (sk * wk) != 0 or kt / (sk * wk) < min_tiles)) {
        if (wk > 4) {
            wk = 4;
        } else if (sk > 1) {
            sk /= 2;
        } else {
            break;
        }
    }
    return .{ .sk = sk, .wk = wk };
}

/// Qwen3.8-27B teacher's per-shape split. Generic plan() picks a different (sk, wk) on these shapes.
pub fn qwenPlan(halves: u16, k: usize, n: usize) Plan {
    if (halves == 6) {
        if (k == 17408 and n == 5120) return .{ .sk = 4, .wk = 2 };
        if (k == 5120 and n == 10240) return .{ .sk = 2, .wk = 2 };
        if (k == 5120 and n == 6144) return .{ .sk = 1, .wk = 8 };
        if (k == 6144 and n == 5120) return .{ .sk = 4, .wk = 2 };
    }
    if (halves == 8) {
        if (k == 17408 and n == 5120) return .{ .sk = 1, .wk = 8 };
        if (k == 5120 and n == 10240) return .{ .sk = 5, .wk = 4 };
        if (k == 5120 and n == 6144) return .{ .sk = 5, .wk = 2 };
        if (k == 6144 and n == 5120) return .{ .sk = 16, .wk = 2 };
    }
    return plan(k, n, 192, 8);
}

const Sym = struct { k2: u16, cb: u8, wk: u8, name: [:0]const u8 };

// exl3 linear: every instantiation the fatbin exports, as cuobjdump lists them (k2, codebook, warps).
const lin_syms = [_]Sym{
    .{ .k2 = 2, .cb = 0, .wk = 4, .name = "_ZN14tf_exl3_linear13linear_kernelILi2ELi0ELi4EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 2, .cb = 0, .wk = 8, .name = "_ZN14tf_exl3_linear13linear_kernelILi2ELi0ELi8EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 2, .cb = 1, .wk = 4, .name = "_ZN14tf_exl3_linear13linear_kernelILi2ELi1ELi4EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 2, .cb = 1, .wk = 8, .name = "_ZN14tf_exl3_linear13linear_kernelILi2ELi1ELi8EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 2, .cb = 2, .wk = 4, .name = "_ZN14tf_exl3_linear13linear_kernelILi2ELi2ELi4EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 2, .cb = 2, .wk = 8, .name = "_ZN14tf_exl3_linear13linear_kernelILi2ELi2ELi8EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 3, .cb = 2, .wk = 4, .name = "_ZN14tf_exl3_linear13linear_kernelILi3ELi2ELi4EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 3, .cb = 2, .wk = 8, .name = "_ZN14tf_exl3_linear13linear_kernelILi3ELi2ELi8EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 4, .cb = 0, .wk = 4, .name = "_ZN14tf_exl3_linear13linear_kernelILi4ELi0ELi4EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 4, .cb = 0, .wk = 8, .name = "_ZN14tf_exl3_linear13linear_kernelILi4ELi0ELi8EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 4, .cb = 1, .wk = 4, .name = "_ZN14tf_exl3_linear13linear_kernelILi4ELi1ELi4EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 4, .cb = 1, .wk = 8, .name = "_ZN14tf_exl3_linear13linear_kernelILi4ELi1ELi8EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 4, .cb = 2, .wk = 4, .name = "_ZN14tf_exl3_linear13linear_kernelILi4ELi2ELi4EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 4, .cb = 2, .wk = 8, .name = "_ZN14tf_exl3_linear13linear_kernelILi4ELi2ELi8EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 5, .cb = 2, .wk = 4, .name = "_ZN14tf_exl3_linear13linear_kernelILi5ELi2ELi4EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 5, .cb = 2, .wk = 8, .name = "_ZN14tf_exl3_linear13linear_kernelILi5ELi2ELi8EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 6, .cb = 0, .wk = 4, .name = "_ZN14tf_exl3_linear13linear_kernelILi6ELi0ELi4EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 6, .cb = 0, .wk = 8, .name = "_ZN14tf_exl3_linear13linear_kernelILi6ELi0ELi8EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 6, .cb = 1, .wk = 4, .name = "_ZN14tf_exl3_linear13linear_kernelILi6ELi1ELi4EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 6, .cb = 1, .wk = 8, .name = "_ZN14tf_exl3_linear13linear_kernelILi6ELi1ELi8EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 6, .cb = 2, .wk = 2, .name = "_ZN14tf_exl3_linear13linear_kernelILi6ELi2ELi2EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 6, .cb = 2, .wk = 4, .name = "_ZN14tf_exl3_linear13linear_kernelILi6ELi2ELi4EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 6, .cb = 2, .wk = 8, .name = "_ZN14tf_exl3_linear13linear_kernelILi6ELi2ELi8EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 7, .cb = 2, .wk = 4, .name = "_ZN14tf_exl3_linear13linear_kernelILi7ELi2ELi4EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 7, .cb = 2, .wk = 8, .name = "_ZN14tf_exl3_linear13linear_kernelILi7ELi2ELi8EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 8, .cb = 0, .wk = 4, .name = "_ZN14tf_exl3_linear13linear_kernelILi8ELi0ELi4EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 8, .cb = 0, .wk = 8, .name = "_ZN14tf_exl3_linear13linear_kernelILi8ELi0ELi8EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 8, .cb = 1, .wk = 4, .name = "_ZN14tf_exl3_linear13linear_kernelILi8ELi1ELi4EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 8, .cb = 1, .wk = 8, .name = "_ZN14tf_exl3_linear13linear_kernelILi8ELi1ELi8EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 8, .cb = 2, .wk = 2, .name = "_ZN14tf_exl3_linear13linear_kernelILi8ELi2ELi2EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 8, .cb = 2, .wk = 4, .name = "_ZN14tf_exl3_linear13linear_kernelILi8ELi2ELi4EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 8, .cb = 2, .wk = 8, .name = "_ZN14tf_exl3_linear13linear_kernelILi8ELi2ELi8EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 10, .cb = 0, .wk = 4, .name = "_ZN14tf_exl3_linear13linear_kernelILi10ELi0ELi4EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 10, .cb = 0, .wk = 8, .name = "_ZN14tf_exl3_linear13linear_kernelILi10ELi0ELi8EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 10, .cb = 1, .wk = 4, .name = "_ZN14tf_exl3_linear13linear_kernelILi10ELi1ELi4EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 10, .cb = 1, .wk = 8, .name = "_ZN14tf_exl3_linear13linear_kernelILi10ELi1ELi8EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 10, .cb = 2, .wk = 4, .name = "_ZN14tf_exl3_linear13linear_kernelILi10ELi2ELi4EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 10, .cb = 2, .wk = 8, .name = "_ZN14tf_exl3_linear13linear_kernelILi10ELi2ELi8EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 12, .cb = 0, .wk = 4, .name = "_ZN14tf_exl3_linear13linear_kernelILi12ELi0ELi4EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 12, .cb = 0, .wk = 8, .name = "_ZN14tf_exl3_linear13linear_kernelILi12ELi0ELi8EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 12, .cb = 1, .wk = 4, .name = "_ZN14tf_exl3_linear13linear_kernelILi12ELi1ELi4EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 12, .cb = 1, .wk = 8, .name = "_ZN14tf_exl3_linear13linear_kernelILi12ELi1ELi8EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 12, .cb = 2, .wk = 4, .name = "_ZN14tf_exl3_linear13linear_kernelILi12ELi2ELi4EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 12, .cb = 2, .wk = 8, .name = "_ZN14tf_exl3_linear13linear_kernelILi12ELi2ELi8EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 14, .cb = 0, .wk = 4, .name = "_ZN14tf_exl3_linear13linear_kernelILi14ELi0ELi4EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 14, .cb = 0, .wk = 8, .name = "_ZN14tf_exl3_linear13linear_kernelILi14ELi0ELi8EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 14, .cb = 1, .wk = 4, .name = "_ZN14tf_exl3_linear13linear_kernelILi14ELi1ELi4EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 14, .cb = 1, .wk = 8, .name = "_ZN14tf_exl3_linear13linear_kernelILi14ELi1ELi8EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 14, .cb = 2, .wk = 4, .name = "_ZN14tf_exl3_linear13linear_kernelILi14ELi2ELi4EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 14, .cb = 2, .wk = 8, .name = "_ZN14tf_exl3_linear13linear_kernelILi14ELi2ELi8EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 16, .cb = 0, .wk = 4, .name = "_ZN14tf_exl3_linear13linear_kernelILi16ELi0ELi4EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 16, .cb = 0, .wk = 8, .name = "_ZN14tf_exl3_linear13linear_kernelILi16ELi0ELi8EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 16, .cb = 1, .wk = 4, .name = "_ZN14tf_exl3_linear13linear_kernelILi16ELi1ELi4EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 16, .cb = 1, .wk = 8, .name = "_ZN14tf_exl3_linear13linear_kernelILi16ELi1ELi8EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 16, .cb = 2, .wk = 4, .name = "_ZN14tf_exl3_linear13linear_kernelILi16ELi2ELi4EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
    .{ .k2 = 16, .cb = 2, .wk = 8, .name = "_ZN14tf_exl3_linear13linear_kernelILi16ELi2ELi8EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii" },
};

// unpack: W_q [K, N] fp16 from either layout's words, one warp a tile.
const unp_syms = [_]Sym{
    .{ .k2 = 2, .cb = 0, .wk = 0, .name = "_ZN14tf_exl3_linear13unpack_kernelILi2ELi0EEEvPKjP6__halfill" },
    .{ .k2 = 2, .cb = 1, .wk = 0, .name = "_ZN14tf_exl3_linear13unpack_kernelILi2ELi1EEEvPKjP6__halfill" },
    .{ .k2 = 2, .cb = 2, .wk = 0, .name = "_ZN14tf_exl3_linear13unpack_kernelILi2ELi2EEEvPKjP6__halfill" },
    .{ .k2 = 3, .cb = 2, .wk = 0, .name = "_ZN14tf_exl3_linear13unpack_kernelILi3ELi2EEEvPKjP6__halfill" },
    .{ .k2 = 4, .cb = 0, .wk = 0, .name = "_ZN14tf_exl3_linear13unpack_kernelILi4ELi0EEEvPKjP6__halfill" },
    .{ .k2 = 4, .cb = 1, .wk = 0, .name = "_ZN14tf_exl3_linear13unpack_kernelILi4ELi1EEEvPKjP6__halfill" },
    .{ .k2 = 4, .cb = 2, .wk = 0, .name = "_ZN14tf_exl3_linear13unpack_kernelILi4ELi2EEEvPKjP6__halfill" },
    .{ .k2 = 5, .cb = 2, .wk = 0, .name = "_ZN14tf_exl3_linear13unpack_kernelILi5ELi2EEEvPKjP6__halfill" },
    .{ .k2 = 6, .cb = 0, .wk = 0, .name = "_ZN14tf_exl3_linear13unpack_kernelILi6ELi0EEEvPKjP6__halfill" },
    .{ .k2 = 6, .cb = 1, .wk = 0, .name = "_ZN14tf_exl3_linear13unpack_kernelILi6ELi1EEEvPKjP6__halfill" },
    .{ .k2 = 6, .cb = 2, .wk = 0, .name = "_ZN14tf_exl3_linear13unpack_kernelILi6ELi2EEEvPKjP6__halfill" },
    .{ .k2 = 7, .cb = 2, .wk = 0, .name = "_ZN14tf_exl3_linear13unpack_kernelILi7ELi2EEEvPKjP6__halfill" },
    .{ .k2 = 8, .cb = 0, .wk = 0, .name = "_ZN14tf_exl3_linear13unpack_kernelILi8ELi0EEEvPKjP6__halfill" },
    .{ .k2 = 8, .cb = 1, .wk = 0, .name = "_ZN14tf_exl3_linear13unpack_kernelILi8ELi1EEEvPKjP6__halfill" },
    .{ .k2 = 8, .cb = 2, .wk = 0, .name = "_ZN14tf_exl3_linear13unpack_kernelILi8ELi2EEEvPKjP6__halfill" },
    .{ .k2 = 10, .cb = 0, .wk = 0, .name = "_ZN14tf_exl3_linear13unpack_kernelILi10ELi0EEEvPKjP6__halfill" },
    .{ .k2 = 10, .cb = 1, .wk = 0, .name = "_ZN14tf_exl3_linear13unpack_kernelILi10ELi1EEEvPKjP6__halfill" },
    .{ .k2 = 10, .cb = 2, .wk = 0, .name = "_ZN14tf_exl3_linear13unpack_kernelILi10ELi2EEEvPKjP6__halfill" },
    .{ .k2 = 12, .cb = 0, .wk = 0, .name = "_ZN14tf_exl3_linear13unpack_kernelILi12ELi0EEEvPKjP6__halfill" },
    .{ .k2 = 12, .cb = 1, .wk = 0, .name = "_ZN14tf_exl3_linear13unpack_kernelILi12ELi1EEEvPKjP6__halfill" },
    .{ .k2 = 12, .cb = 2, .wk = 0, .name = "_ZN14tf_exl3_linear13unpack_kernelILi12ELi2EEEvPKjP6__halfill" },
    .{ .k2 = 14, .cb = 0, .wk = 0, .name = "_ZN14tf_exl3_linear13unpack_kernelILi14ELi0EEEvPKjP6__halfill" },
    .{ .k2 = 14, .cb = 1, .wk = 0, .name = "_ZN14tf_exl3_linear13unpack_kernelILi14ELi1EEEvPKjP6__halfill" },
    .{ .k2 = 14, .cb = 2, .wk = 0, .name = "_ZN14tf_exl3_linear13unpack_kernelILi14ELi2EEEvPKjP6__halfill" },
    .{ .k2 = 16, .cb = 0, .wk = 0, .name = "_ZN14tf_exl3_linear13unpack_kernelILi16ELi0EEEvPKjP6__halfill" },
    .{ .k2 = 16, .cb = 1, .wk = 0, .name = "_ZN14tf_exl3_linear13unpack_kernelILi16ELi1EEEvPKjP6__halfill" },
    .{ .k2 = 16, .cb = 2, .wk = 0, .name = "_ZN14tf_exl3_linear13unpack_kernelILi16ELi2EEEvPKjP6__halfill" },
};

fn find(comptime table: []const Sym, k2: u16, cb: u8, wk: u8) ?[:0]const u8 {
    inline for (table) |s| {
        if (s.k2 == k2 and s.cb == cb and s.wk == wk) return s.name;
    }
    return null;
}

/// The loaded exl3_linear module with its kernels resolved once.
pub const Linear = struct {
    d: *const Driver,
    mod: Module,
    rot_in: @import("module.zig").Function,
    lin: [lin_syms.len]@import("module.zig").Function,
    unp: [unp_syms.len]@import("module.zig").Function,
    counters: memory.DeviceBuffer,
    own_mod: ?Module = null, // set by load(); fromModule callers keep their module

    pub fn load(d: *const Driver) !Linear {
        if (!kernels.available) return error.NoKernels;
        const mod = try Module.load(d, kernels.exl3_linear);
        var out = try Linear.fromModule(d, &mod);
        out.own_mod = mod;
        return out;
    }

    /// The same resolution over a module the caller loaded and owns: this Linear never unloads it.
    pub fn fromModule(d: *const Driver, mod: *const Module) !Linear {
        var out: Linear = undefined;
        out.d = d;
        out.rot_in = try mod.function(kernels.exl3_symbols.rot_in);
        const counter_bytes = 8 * 1024;
        out.counters = try memory.DeviceBuffer.alloc(d, counter_bytes);
        errdefer out.counters.free();
        out.counters.fill32(0, null) catch {};
        inline for (lin_syms, 0..) |s, i| out.lin[i] = try mod.function(s.name);
        inline for (unp_syms, 0..) |s, i| out.unp[i] = try mod.function(s.name);
        return out;
    }

    pub fn deinit(l: *Linear) void {
        l.counters.free();
        if (l.own_mod) |*m| m.unload();
    }

    /// xh [M, K] fp16 = (x * suh) rotated by H / sqrt(128): rot_in_kernel, one block a 4-block span and row.
    pub fn rotIn(l: *const Linear, s: stream_.Stream, x: memory.DeviceBuffer, suh: memory.DeviceBuffer, xh: memory.DeviceBuffer, m: usize, k: usize) !void {
        const blocks = (k / 128 + 3) / 4;
        var args = launch.Args{};
        args.add(x.ptr);
        args.add(@as(c_int, 1)); // BF16 inputs everywhere in the native engine
        args.add(suh.ptr);
        args.add(xh.ptr);
        args.add(@as(c_int, @intCast(k)));
        if (std.c.getenv("TF_TRACE_LAUNCH") != null) std.debug.print("[rotIn] blocks={d} m={d} k={d}\n", .{ blocks, m, k });
        try launch.launch(l.rot_in, .{ .grid = .{ .x = @intCast(blocks), .y = @intCast(m), .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 } }, s, &args);
    }

    /// y [M, N] bf16 = xh @ W + bias by the trellis GEMV; a row's bits are its own at every width.
    pub fn matmul(
        l: *const Linear,
        s: stream_.Stream,
        w: Weight,
        xh: memory.DeviceBuffer,
        y: memory.DeviceBuffer,
        z: ?memory.DeviceBuffer,
        m: usize,
    ) !void {
        const p = w.split();
        const f = try l.function(p.wk, w.codebook, w.width.halves);
        const shared: u32 = @intCast(p.wk * @min(m, 8) * 128 * 4);
        if (shared > 48 * 1024) try f.allowDynamicShared(shared);
        var args = launch.Args{};
        args.add(xh.ptr);
        args.add(w.trellis.ptr);
        args.add(@as(i64, @intCast(w.strideK())));
        args.add(@as(i64, @intCast(w.strideN())));
        args.add(w.svh.ptr);
        args.add(w.bias.ptr);
        args.add(y.ptr);
        args.add(@as(c_int, 1)); // y bf16
        args.add(if (z) |zz| zz.ptr else @as(usize, 0));
        args.add(l.counters.ptr);
        args.add(@as(c_int, @intCast(m)));
        args.add(@as(c_int, @intCast(w.k)));
        args.add(@as(c_int, @intCast(w.n)));
        args.add(@as(c_int, @intCast(p.sk)));
        const grid_x: u32 = @intCast(w.n / 128);
        if (std.c.getenv("TF_TRACE_LAUNCH") != null) std.debug.print("[matmul] grid=({d},{d}) wk={d} m={d} k={d} n={d} shared={d}\n", .{ grid_x, p.sk, p.wk, m, w.k, w.n, shared });
        try launch.launch(f, .{
            .grid = .{ .x = grid_x, .y = @intCast(p.sk), .z = 1 },
            .block = .{ .x = @intCast(p.wk * 32), .y = 1, .z = 1 },
            .shared = shared,
        }, s, &args);
    }

    /// rotIn over raw device pointers: x (rows, k) bf16 -> xh fp16, scaled and rotated.
    pub fn rotInPtr(l: *const Linear, s: stream_.Stream, x: u64, suh: u64, xh: u64, m: usize, k: usize) !void {
        const blocks = (k / 128 + 3) / 4;
        var args = launch.Args{};
        args.add(x);
        args.add(@as(c_int, 1)); // BF16 inputs
        args.add(suh);
        args.add(xh);
        args.add(@as(c_int, @intCast(k)));
        if (std.c.getenv("TF_TRACE_LAUNCH") != null) std.debug.print("[rotIn] blocks={d} m={d} k={d}\n", .{ blocks, m, k });
        try launch.launch(l.rot_in, .{ .grid = .{ .x = @intCast(blocks), .y = @intCast(m), .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 } }, s, &args);
    }

    /// matmul over raw pointers, with the split-K partials buffer when the plan splits.
    pub fn matmulPtr(l: *const Linear, s: stream_.Stream, w: Weight, xh: u64, y: u64, z: u64, m: usize) !void {
        const p = w.split();
        const f = try l.function(p.wk, w.codebook, w.width.halves);
        const shared: u32 = @intCast(p.wk * @min(m, 8) * 128 * 4);
        if (std.c.getenv("TF_TRACE_LAUNCH") != null) std.debug.print("[matmul] entering wk={d} shared={d} k={d} n={d} m={d} sk={d}\n", .{ p.wk, shared, w.k, w.n, m, p.sk });
        if (shared > 48 * 1024) try f.allowDynamicShared(shared);
        var args = launch.Args{};
        args.add(xh);
        args.add(w.trellis.ptr);
        args.add(@as(i64, @intCast(w.strideK())));
        args.add(@as(i64, @intCast(w.strideN())));
        args.add(w.svh.ptr);
        args.add(w.bias.ptr);
        args.add(y);
        args.add(@as(c_int, 1)); // y bf16
        args.add(z); // 0 when unsplit
        args.add(l.counters.ptr);
        args.add(@as(c_int, @intCast(m)));
        args.add(@as(c_int, @intCast(w.k)));
        args.add(@as(c_int, @intCast(w.n)));
        args.add(@as(c_int, @intCast(p.sk)));
        const grid_x: u32 = @intCast(w.n / 128);
        if (std.c.getenv("TF_TRACE_LAUNCH") != null) std.debug.print("[matmul] grid=({d},{d}) wk={d} m={d} k={d} n={d} shared={d}\n", .{ grid_x, p.sk, p.wk, m, w.k, w.n, shared });
        try launch.launch(f, .{
            .grid = .{ .x = grid_x, .y = @intCast(p.sk), .z = 1 },
            .block = .{ .x = @intCast(p.wk * 32), .y = 1, .z = 1 },
            .shared = shared,
        }, s, &args);
    }

    /// W_q [k, n] fp16 unpacked from the trellis, once a prompt chunk: one warp a 16x16 tile over (N/16, K/16).
    pub fn unpackPtr(l: *const Linear, s: stream_.Stream, w: Weight, out: u64) !void {
        const f = try l.unpackFn(w.width.halves, w.codebook);
        var args = launch.Args{};
        args.add(w.trellis.ptr);
        args.add(out);
        args.add(@as(c_int, @intCast(w.n)));
        args.add(@as(i64, @intCast(w.strideK())));
        args.add(@as(i64, @intCast(w.strideN())));
        if (std.c.getenv("TF_TRACE_LAUNCH") != null) std.debug.print("[unpack] grid=({d},{d}) k={d} n={d} halves={d} cb={d}\n", .{ w.n / 16, w.k / 16, w.k, w.n, w.width.halves, @backingInt(w.codebook) });
        try launch.launch(f, .{ .grid = .{ .x = @intCast(w.n / 16), .y = @intCast(w.k / 16), .z = 1 }, .block = .{ .x = 32, .y = 1, .z = 1 } }, s, &args);
    }

    fn unpackFn(l: *const Linear, halves: u16, cb: Codebook) !@import("module.zig").Function {
        for (unp_syms, 0..) |sym, i| {
            if (sym.k2 == halves and sym.cb == @backingInt(cb)) return l.unp[i];
        }
        return error.UnsupportedWidth;
    }

    fn function(l: *const Linear, wk: u32, cb: Codebook, halves: u16) !@import("module.zig").Function {
        const k2: u16 = halves;
        for (lin_syms, 0..) |sym, i| {
            if (sym.k2 == k2 and sym.cb == @backingInt(cb) and sym.wk == wk) return l.lin[i];
        }
        return error.UnsupportedWidth;
    }
};

/// One uploaded EXL3 projection: the trellis as stored ([K/16, N/16, words] i16 = uint32 pairs), fp16 scales, bias.
pub const Weight = struct {
    trellis: memory.DeviceBuffer,
    suh: memory.DeviceBuffer,
    svh: memory.DeviceBuffer,
    bias: memory.DeviceBuffer,
    k: usize,
    n: usize,
    width: Width,
    codebook: Codebook,
    fixed_plan: ?Plan = null,

    /// The stored layout's uint32 strides: (words between k tiles, words between 128-column blocks).
    pub fn strideK(w: Weight) usize {
        return (w.n / 16) * w.width.tileWords();
    }

    pub fn strideN(w: Weight) usize {
        return 8 * w.width.tileWords();
    }

    pub fn split(w: Weight) Plan {
        return w.fixed_plan orelse plan(w.k, w.n, 192, 8);
    }
};

/// Upload one group as stored (no reorder): trellis, scales (packed signs expanded on the host), optional bias.
pub fn upload(d: *const Driver, a: std.mem.Allocator, g: Group) !Weight {
    if (g.k % 128 != 0 or g.n % 128 != 0) return error.NotMultipleOf128;
    var trellis = try memory.DeviceBuffer.alloc(d, g.trellis.bytes.len);
    errdefer trellis.free();
    try trellis.upload(0, g.trellis.bytes);
    var suh = try uploadScale(d, a, g.suh, g.k);
    errdefer suh.free();
    var svh = try uploadScale(d, a, g.svh, g.n);
    errdefer svh.free();
    var bias = memory.DeviceBuffer{ .d = d, .ptr = 0, .len = 0 };
    if (g.bias) |b| {
        bias = try memory.DeviceBuffer.alloc(d, b.bytes.len);
        try bias.upload(0, b.bytes);
    }
    return .{ .trellis = trellis, .suh = suh, .svh = svh, .bias = bias, .k = g.k, .n = g.n, .width = g.width, .codebook = g.codebook };
}

/// A scale tensor as fp16 bytes: fp16 as stored, or packed int16 sign words expanded by the format's rule.
fn uploadScale(d: *const Driver, a: std.mem.Allocator, t: format.Entry, size: usize) !memory.DeviceBuffer {
    if (t.dtype == .f16) {
        if (t.bytes.len != size * 2) return error.BadScale;
        var out = try memory.DeviceBuffer.alloc(d, t.bytes.len);
        errdefer out.free();
        try out.upload(0, t.bytes);
        return out;
    }
    if (t.dtype == .i16 and t.bytes.len * 8 == size * 2) {
        const words: []align(1) const i16 = @as([*]align(1) const i16, @ptrCast(t.bytes.ptr))[0 .. t.bytes.len / 2];
        const f16s = try a.alloc(f16, size);
        defer a.free(f16s);
        format.unpackSigns(words, f16s);
        var out = try memory.DeviceBuffer.alloc(d, size * 2);
        errdefer out.free();
        try out.upload(0, std.mem.sliceAsBytes(f16s));
        return out;
    }
    return error.BadScale;
}

test "plan matches linear.py at the 27B's shapes" {
    const t = std.testing;
    // q_proj 5120x5120: nb=40 < 64, kt=320; sk doubles until 40*sk >= 192, then K-divisibility holds
    var p = plan(5120, 5120, 192, 8);
    try t.expectEqual(@as(u32, 8), p.sk);
    try t.expectEqual(@as(u32, 4), p.wk);
    // lm_head 5120x151936: nb=1188 >= 64
    p = plan(5120, 151936, 192, 8);
    try t.expectEqual(@as(u32, 1), p.sk);
    try t.expectEqual(@as(u32, 8), p.wk);
    // down 17408x5120: nb=40, kt=1088; sk=8 needs kt%32==0 (1088%32=0) and 1088/32=34>=8
    p = plan(17408, 5120, 192, 8);
    try t.expectEqual(@as(u32, 8), p.sk);
    try t.expectEqual(@as(u32, 4), p.wk);
}

test "qwenPlan pins the 27B teacher's splits" {
    const t = std.testing;
    const rows = [_]struct { h: u16, k: usize, n: usize, sk: u32, wk: u32 }{
        .{ .h = 6, .k = 17408, .n = 5120, .sk = 4, .wk = 2 },
        .{ .h = 6, .k = 5120, .n = 10240, .sk = 2, .wk = 2 },
        .{ .h = 6, .k = 5120, .n = 6144, .sk = 1, .wk = 8 },
        .{ .h = 6, .k = 6144, .n = 5120, .sk = 4, .wk = 2 },
        .{ .h = 8, .k = 17408, .n = 5120, .sk = 1, .wk = 8 },
        .{ .h = 8, .k = 5120, .n = 10240, .sk = 5, .wk = 4 },
        .{ .h = 8, .k = 5120, .n = 6144, .sk = 5, .wk = 2 },
        .{ .h = 8, .k = 6144, .n = 5120, .sk = 16, .wk = 2 },
    };
    for (rows) |r| {
        const p = qwenPlan(r.h, r.k, r.n);
        try t.expectEqual(r.sk, p.sk);
        try t.expectEqual(r.wk, p.wk);
        const g = plan(r.k, r.n, 192, 8);
        try t.expect(g.sk != p.sk or g.wk != p.wk);
    }
    const head = qwenPlan(6, 5120, 248320);
    const generic = plan(5120, 248320, 192, 8);
    try t.expectEqual(generic.sk, head.sk);
    try t.expectEqual(generic.wk, head.wk);
}

test "every symbol table entry is unique" {
    var seen = std.StringHashMap(void).init(std.testing.allocator);
    defer seen.deinit();
    for (lin_syms) |s| try std.testing.expect((try seen.getOrPut(s.name)).found_existing == false);
    for (unp_syms) |s| try std.testing.expect((try seen.getOrPut(s.name)).found_existing == false);
}

test "width and strides" {
    const w = Weight{ .trellis = .{ .d = undefined, .ptr = 0, .len = 0 }, .suh = .{ .d = undefined, .ptr = 0, .len = 0 }, .svh = .{ .d = undefined, .ptr = 0, .len = 0 }, .bias = .{ .d = undefined, .ptr = 0, .len = 0 }, .k = 5120, .n = 5120, .width = .{ .halves = 6 }, .codebook = .mul1 };
    try std.testing.expectEqual(@as(usize, 320 * 24), w.strideK());
    try std.testing.expectEqual(@as(usize, 8 * 24), w.strideN());
}
