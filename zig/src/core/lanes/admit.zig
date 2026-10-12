//! Admitting streams that arrive together: shared prompt passes (multi.py's _fill_batch), then open.zig's first tokens.
const std = @import("std");
const be = @import("backend.zig");
const ev = @import("events.zig");
const trail = @import("trail.zig");
const open = @import("open.zig");
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;
const f = ev.f;
const str = trail.str;

/// Whether the backend prefills several streams in shared passes (`prefill_many`).
pub fn together(e: *const Engine) bool {
    return e.backend.vtable.prefill_many != null;
}

/// open.add for several streams: shared prompt passes when the backend has them, every first draft in one request.
pub fn streams(e: *Engine, list: []const *Stream, errs: []?anyerror) !void {
    _ = e.arena.reset(.retain_capacity);
    const a = e.arena.allocator();
    @memset(errs, null);
    for (list) |s| try trail.event(e, &.{ f("ev", str("add")), f("stream", str(s.id)) });
    if (list.len > 1 and e.backend.vtable.prefill_many != null) {
        e.backend.vtable.prefill_many.?(e.backend.ptr, list, errs) catch |x| for (errs) |*err| {
            if (err.* == null) err.* = x;
        };
    } else for (list, errs) |s, *err| e.backend.prefill(s) catch |x| {
        err.* = x;
    };
    const openings = try a.alloc(open.Opening, list.len);
    var requests: std.ArrayList(be.DraftRequest) = .empty;
    for (list, errs, openings) |s, *err, *o| {
        if (err.*) |x| {
            if (x == error.Cancelled) e.backend.release(s); // the host finishes a cancelled stream without the core
            continue;
        }
        o.* = open.opening(e, s) catch |x| { // open.opening releases a stream cancelled in its last chunk
            err.* = x;
            continue;
        };
        if (o.asked) |d| try requests.append(a, open.draftRequest(s, o.feed, d));
    }
    if (requests.items.len > 0) try e.backend.draft(requests.items);
    for (list, errs, openings) |s, err, o| {
        if (err == null) try open.joined(e, s, o);
    }
}
