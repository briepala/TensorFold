//! Standalone: launch rot_in alone through cuda.exl3.Linear to isolate the INVALID_VALUE.

const std = @import("std");
const cuda = @import("cuda");

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var s = try cuda.Stream.init(ctx.d, true);
    defer s.deinit();

    const img = cuda.kernels.exl3_linear;
    std.debug.print("image bytes: {d}\n", .{img.len});
    var lin = try cuda.exl3.Linear.load(ctx.d);
    defer {
        lin.counters.free();
        lin.own_mod.?.unload();
    }

    // synthetic: x bf16 (4, 5120), suh fp16 5120, xh fp16 (4, 5120)
    const m = 4;
    const k = 5120;
    var x = try cuda.DeviceBuffer.alloc(ctx.d, m * k * 2);
    defer x.free();
    var suh = try cuda.DeviceBuffer.alloc(ctx.d, k * 2);
    defer suh.free();
    var xh = try cuda.DeviceBuffer.alloc(ctx.d, m * k * 2);
    defer xh.free();
    try x.fill16(0, null);
    try suh.fill16(0x3C00, null); // fp16 1.0
    try xh.fill16(0, null);

    try lin.rotInPtr(s, x.ptr, suh.ptr, xh.ptr, m, k);
    try s.synchronize();
    std.debug.print("rotIn alone: OK\n", .{});
}
