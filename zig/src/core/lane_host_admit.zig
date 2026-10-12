//! Which queued request a lane host admits next, and a batch whose prompt passes the backend shares.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Job = LaneHost.Job;

/// Every queued request the lanes have room for. One at a time, unless the backend shares prompt passes.
pub fn ready(h: *LaneHost) void {
    if (!h.core.together() or h.info_.admit != null) {
        while (h.admitOne()) {}
        return;
    }
    var batch: std.ArrayList(*Job) = .empty;
    defer batch.deinit(h.gpa);
    while (true) {
        if (batch.items.len > 0 and warmNext(h)) {
            if (!addBatch(h, batch.items)) return;
            batch.clearRetainingCapacity();
        }
        const job = switch (h.prepare()) {
            .none => break,
            .dropped => continue,
            .job => |j| j,
        };
        if (h.opensAlone(job)) {
            if (batch.items.len > 0 and !addBatch(h, batch.items)) return;
            batch.clearRetainingCapacity();
            if (!h.openJob(job)) return;
            continue;
        }
        batch.append(h.gpa, job) catch {
            _ = h.drop(job, "out of memory");
        };
    }
    _ = addBatch(h, batch.items);
}

/// The next queued request is a warm pass: streams already taken go in before it.
fn warmNext(h: *LaneHost) bool {
    if (!h.info_.warm_turns) return false;
    h.lock();
    defer h.unlock();
    return h.queued.items.len > 0 and h.queued.items[0].request.max_tokens == 0;
}

/// One job through addStream, several with their prompt passes shared. False when one waits for memory.
fn addBatch(h: *LaneHost, jobs: []const *Job) bool {
    if (jobs.len == 0) return true;
    if (jobs.len == 1) {
        return h.openJob(jobs[0]);
    }
    const streams = h.gpa.alloc(*lanes.Stream, jobs.len) catch return dropAll(h, jobs, "out of memory");
    defer h.gpa.free(streams);
    const errs = h.gpa.alloc(?anyerror, jobs.len) catch return dropAll(h, jobs, "out of memory");
    defer h.gpa.free(errs);
    for (streams, jobs) |*s, job| s.* = &job.stream;
    @memset(errs, null);
    var failed: ?anyerror = null; // the call's own failure: a stream it left without its own error ends with it
    h.core.addStreams(streams, errs) catch |e| {
        failed = e;
    };
    for (jobs, errs) |job, err| {
        if (!waits(err orelse failed)) _ = settle(h, job, err orelse failed);
    }
    var more = true; // waiting ones go back to the queue's head in order: the last one first
    var i = jobs.len;
    while (i > 0) {
        i -= 1;
        if (waits(errs[i] orelse failed) and !settle(h, jobs[i], error.LaneMemory)) more = false;
    }
    return more;
}

fn waits(err: ?anyerror) bool {
    return if (err) |e| e == error.LaneMemory else false;
}

/// A job after its prompt pass: delivered when it went in, else cancelled, dropped or requeued (false: holds).
fn settle(h: *LaneHost, job: *Job, err: ?anyerror) bool {
    if (err) |e| return switch (e) {
        error.Cancelled => h.cancel(job),
        error.LaneMemory => h.requeue(job),
        else => h.drop(job, h.words(e)),
    };
    h.prefilled(job, job.began);
    if (h.deliver(job)) h.remove(job);
    return true;
}

fn dropAll(h: *LaneHost, jobs: []const *Job, message: []const u8) bool {
    for (jobs) |job| _ = h.drop(job, message);
    return false;
}

/// The queued job to admit (its index), or null to hold; the caller holds the lock and a lane is free.
pub fn pick(h: *LaneHost) ?usize {
    const a = h.info_.admit orelse return 0;
    if (h.core.activeCount() == 0) return 0; // alone: the whole pool is free (kept prefixes give their blocks back)
    const head = h.queued.items[0];
    if (fits(a, head)) return 0;
    if (head.waits_since == null) head.waits_since = h.admissions;
    // it waits on the streams admitted before it waited: while one runs, later requests that fit go first; then none do
    for (h.admitted.items) |j| if (j.seq < head.waits_since.?) {
        for (h.queued.items[1..], 1..) |j2, i| if (fits(a, j2)) return i;
        return null;
    };
    return null;
}

fn fits(a: api.Admit, job: *const Job) bool {
    return a.fits(a.ctx, @intCast(job.request.prompt.len), job.request.max_tokens);
}
