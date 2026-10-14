//! EXL3 dispatch and the b16 path for the 27B: dense projections by weight format, plus the GEMV scratch.

const std = @import("std");
const cuda = @import("cuda");

/// The trellis decode's scratch, grown to the largest call and reused (calls run in order on one stream).
pub const Trellis = struct {
    d: *const cuda.Driver,
    xh: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0 }, // fp16 rotated inputs
    z: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0 }, // fp32 split-K partials
    wq: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0 }, // fp16 unpacked W_q (prompt)
    hadamard: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0 }, // the 128x128 H, bf16

    pub fn init(d: *const cuda.Driver) Trellis {
        return .{ .d = d };
    }

    pub fn deinit(t: *Trellis) void {
        t.xh.free();
        t.z.free();
        t.wq.free();
        t.hadamard.free();
    }

    fn grow(t: *Trellis, buf: *cuda.DeviceBuffer, bytes: usize) !u64 {
        if (buf.len < bytes) {
            buf.free();
            buf.* = try cuda.DeviceBuffer.alloc(t.d, bytes);
        }
        return buf.ptr;
    }

    /// fp16 [rows, k] rotated inputs.
    pub fn xhAt(t: *Trellis, rows: usize, k: usize) !u64 {
        return t.grow(&t.xh, rows * k * 2);
    }

    /// fp32 [sk, rows, n] split-K partials.
    pub fn zAt(t: *Trellis, sk: usize, rows: usize, n: usize) !u64 {
        return t.grow(&t.z, sk * rows * n * 4);
    }

    /// fp16 [k, n] unpacked W_q.
    pub fn wqAt(t: *Trellis, k: usize, n: usize) !u64 {
        return t.grow(&t.wq, k * n * 2);
    }

    /// The 128x128 Hadamard matrix, bf16: +1/-1 by row&column index parity.
    pub fn hadamardAt(t: *Trellis) !u64 {
        if (t.hadamard.len == 0) {
            t.hadamard = try cuda.DeviceBuffer.alloc(t.d, 128 * 128 * 2);
            var h: [128 * 128]u16 = undefined;
            for (0..128) |i| for (0..128) |j| {
                const parity = @popCount(i & j) & 1;
                const v: f32 = if (parity == 1) -1.0 else 1.0;
                h[i * 128 + j] = @bitCast(bf16FromF32(v));
            };
            try t.hadamard.upload(0, std.mem.sliceAsBytes(&h));
        }
        return t.hadamard.ptr;
    }
};

fn bf16FromF32(v: f32) u16 {
    // exact for ±1: the only values the Hadamard holds
    return if (v < 0) 0xBF80 else 0x3F80;
}
