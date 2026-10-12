//! Host checks of the CUDA opener: chip classes, variables, context windows. No device.
const std = @import("std");
const cuda_open = @import("cuda.zig");
const budget = @import("cuda_memory.zig");

test "this host's /proc/meminfo reads whole and parses (Linux)" {
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest;
    const a = std.testing.allocator;
    const text = budget.procText(a, std.testing.io, "/proc/meminfo") orelse return error.Unreadable;
    defer a.free(text);
    try std.testing.expect(budget.meminfo(text) != null);
}

test "chip classes name the compute capability as gate entries do" {
    const a = std.testing.allocator;
    const name = cuda_open.chipClass(a, 121).?;
    defer a.free(name);
    try std.testing.expectEqualStrings("nvidia-sm121", name);
}

test "every registered family is listed for capabilities" {
    try std.testing.expectEqual(@as(usize, cuda_open.registry.len), cuda_open.families.len);
    try std.testing.expectEqualStrings("nemotron_h", cuda_open.families[0].model_type);
}

test "variables: unset or empty means the default, anything but a whole number is refused" {
    try std.testing.expectEqual(@as(?u32, null), try cuda_open.envNumber(null));
    try std.testing.expectEqual(@as(?u32, null), try cuda_open.envNumber(""));
    try std.testing.expectEqual(@as(?u32, 3), try cuda_open.envNumber(" 3"));
    try std.testing.expectError(error.Invalid, cuda_open.envNumber("one"));
    try std.testing.expectError(error.Invalid, cuda_open.envNumber("-1"));
}

test "context windows: the family default inside the model's, 0 for the model's, never past it" {
    try std.testing.expectEqual(@as(i64, 16384), try cuda_open.contextWindow(null, 262144, 16384));
    try std.testing.expectEqual(@as(i64, 4096), try cuda_open.contextWindow(null, 4096, 16384));
    try std.testing.expectEqual(@as(i64, 262144), try cuda_open.contextWindow(0, 262144, 16384));
    try std.testing.expectEqual(@as(i64, 65536), try cuda_open.contextWindow(65536, 262144, 16384));
    try std.testing.expectError(error.PastNative, cuda_open.contextWindow(300000, 262144, 16384));
    try std.testing.expectError(error.Negative, cuda_open.contextWindow(-5, 262144, 16384));
    try std.testing.expectError(error.NoNative, cuda_open.contextWindow(0, 0, 16384));
}
