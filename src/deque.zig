const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const Buffer = @import("buffer.zig").Buffer;

/// Initial Deque buffer size exponent: 2^8 = 256 slots.
const initial_size_exp: u5 = 8;

/// Lock-free Chase-Lev work-stealing deque.
///
/// Single-producer (owner) pushes and pops from the bottom (LIFO).
/// Multiple consumers (thieves) steal from the top (FIFO).
///
/// Based on "Dynamic Circular Work-Stealing Deque" (Chase & Lev, 2005),
/// with memory ordering corrections from Lê et al., 2013.
pub fn Deque(comptime T: type) type {
    return struct {
        // top, bottom, and buffer are placed on separate cachelines to avoid
        // false sharing: stealers hammer top with CAS, owner hammers bottom on
        // every push/pop, and buffer is read by both. Adjacency would let one
        // thread's writes evict another's hot line.
        top: std.atomic.Value(usize) align(std.atomic.cache_line),
        bottom: std.atomic.Value(usize) align(std.atomic.cache_line),
        buffer: std.atomic.Value(*Buffer(T)) align(std.atomic.cache_line),
        allocator: Allocator,
        // Keep old buffers alive until deinit to avoid use-after-free by stealers
        old_buffers: std.ArrayList(*Buffer(T)),

        const Self = @This();

        pub fn init(allocator: Allocator) !Self {
            const buf = try allocator.create(Buffer(T));
            errdefer allocator.destroy(buf);
            buf.* = try Buffer(T).init(allocator, initial_size_exp);
            errdefer buf.deinit(allocator);

            // Preallocate old_buffers to its maximum possible length so that
            // `push`'s grow path cannot fail for bookkeeping reasons alone.
            // Each grow doubles the buffer, so at most (max_size_exp -
            // initial_size_exp) entries are ever appended.
            var old_buffers: std.ArrayList(*Buffer(T)) = .empty;
            errdefer old_buffers.deinit(allocator);
            try old_buffers.ensureTotalCapacityPrecise(
                allocator,
                @as(usize, Buffer(T).max_size_exp) - initial_size_exp,
            );

            return .{
                .top = std.atomic.Value(usize).init(0),
                .bottom = std.atomic.Value(usize).init(0),
                .buffer = std.atomic.Value(*Buffer(T)).init(buf),
                .allocator = allocator,
                .old_buffers = old_buffers,
            };
        }

        pub fn deinit(self: *Self) void {
            // Free all old buffers
            for (self.old_buffers.items) |buf| {
                buf.deinit(self.allocator);
                self.allocator.destroy(buf);
            }
            self.old_buffers.deinit(self.allocator);
            // Free current buffer
            const buf = self.buffer.load(.unordered);
            buf.deinit(self.allocator);
            self.allocator.destroy(buf);
        }

        /// Push an item to the bottom of the deque. Owner thread only.
        pub fn push(self: *Self, value: T) !void {
            const b = self.bottom.load(.unordered);
            const t = self.top.load(.acquire);
            var buf = self.buffer.load(.acquire);

            const size = b -% t;
            if (size >= buf.mask + 1) {
                // Buffer full, grow. old_buffers was preallocated in init() to
                // its maximum possible length, so appendAssumeCapacity cannot
                // fail here — the only fallible ops are the two allocations
                // below, both cleaned up via errdefer.
                const new_buf = try self.allocator.create(Buffer(T));
                errdefer self.allocator.destroy(new_buf);
                new_buf.* = try buf.grow(self.allocator, t, b);
                errdefer new_buf.deinit(self.allocator);
                self.old_buffers.appendAssumeCapacity(buf);
                self.buffer.store(new_buf, .release);
                buf = new_buf;
            }

            buf.put(b, value);
            // Release fence equivalent: ensure the put is visible before bottom update
            self.bottom.store(b +% 1, .seq_cst);
        }

        /// Pop an item from the bottom of the deque. Owner thread only.
        /// Returns null if the deque is empty.
        pub fn pop(self: *Self) ?T {
            const b = self.bottom.load(.seq_cst) -% 1;
            self.bottom.store(b, .seq_cst);
            const t = self.top.load(.seq_cst);
            const buf = self.buffer.load(.acquire);
            const capacity = buf.mask + 1;
            const size = (b +% 1) -% t;

            if (size != 0 and size <= capacity) {
                // Non-empty
                const value = buf.get(b);
                if (size == 1) {
                    // Last element — race with stealers. The top CAS is the
                    // synchronization point; the bottom store below only needs
                    // monotonic/relaxed semantics (single-writer owner).
                    if (self.top.cmpxchgStrong(t, t +% 1, .seq_cst, .seq_cst) != null) {
                        // Lost the race
                        self.bottom.store(t +% 1, .unordered);
                        return null;
                    }
                    self.bottom.store(t +% 1, .unordered);
                }
                return value;
            } else {
                // Empty. No synchronization needed on this store: the owner is
                // the sole writer of bottom, and stealers that observe a stale
                // value still fail their CAS on top.
                self.bottom.store(t, .unordered);
                return null;
            }
        }

        pub const StealResult = union(enum) {
            success: T,
            empty,
            retry,
        };

        /// Steal an item from the top of the deque. Any thread.
        pub fn steal(self: *Self) StealResult {
            const t = self.top.load(.seq_cst);
            const b = self.bottom.load(.seq_cst);
            const buf = self.buffer.load(.acquire);
            const capacity = buf.mask + 1;
            const size = b -% t;

            if (size == 0 or size > capacity) {
                return .empty;
            }

            const value = buf.get(t);

            if (self.top.cmpxchgStrong(t, t +% 1, .seq_cst, .seq_cst) != null) {
                return .retry;
            }

            return .{ .success = value };
        }

        /// Returns the number of items in the deque (approximate, for diagnostics only).
        pub fn count(self: *const Self) usize {
            const b = self.bottom.load(.acquire);
            const t = self.top.load(.acquire);
            const size = b -% t;
            const buf = self.buffer.load(.acquire);
            const capacity = buf.mask + 1;
            return if (size > capacity) 0 else size;
        }
    };
}

// =============================================================================
// Tests
// =============================================================================

test "push and pop single item" {
    var d = try Deque(u64).init(std.testing.allocator);
    defer d.deinit();

    try d.push(42);
    try std.testing.expectEqual(@as(u64, 42), d.pop().?);
    try std.testing.expectEqual(@as(?u64, null), d.pop());
}

test "push and pop LIFO order" {
    var d = try Deque(u64).init(std.testing.allocator);
    defer d.deinit();

    for (0..10) |i| {
        try d.push(@intCast(i));
    }

    // Pop should return in reverse order (LIFO)
    var i: u64 = 10;
    while (i > 0) {
        i -= 1;
        try std.testing.expectEqual(i, d.pop().?);
    }
    try std.testing.expectEqual(@as(?u64, null), d.pop());
}

test "steal returns FIFO order" {
    var d = try Deque(u64).init(std.testing.allocator);
    defer d.deinit();

    for (0..10) |i| {
        try d.push(@intCast(i));
    }

    // Steal should return in insertion order (FIFO)
    for (0..10) |i| {
        const result = d.steal();
        switch (result) {
            .success => |val| try std.testing.expectEqual(@as(u64, @intCast(i)), val),
            else => return error.UnexpectedResult,
        }
    }
    try std.testing.expectEqual(Deque(u64).StealResult.empty, d.steal());
}

test "empty deque" {
    var d = try Deque(u64).init(std.testing.allocator);
    defer d.deinit();

    try std.testing.expectEqual(@as(?u64, null), d.pop());
    try std.testing.expectEqual(Deque(u64).StealResult.empty, d.steal());
}

test "grow buffer" {
    var d = try Deque(u64).init(std.testing.allocator);
    defer d.deinit();

    // Push more than initial capacity (256)
    for (0..512) |i| {
        try d.push(@intCast(i));
    }

    // Pop all and verify correctness
    var i: u64 = 512;
    while (i > 0) {
        i -= 1;
        try std.testing.expectEqual(i, d.pop().?);
    }
}

test "grow keeps old buffers alive for in-flight stealers" {
    var d = try Deque(u64).init(std.testing.allocator);
    defer d.deinit();

    const old_buf = d.buffer.load(.acquire);
    const old_capacity = old_buf.mask + 1;

    // Push one past capacity to force a grow.
    for (0..old_capacity + 1) |i| {
        try d.push(@intCast(i));
    }

    // The pre-grow buffer must be retained until deinit.
    try std.testing.expectEqual(@as(usize, 1), d.old_buffers.items.len);
    try std.testing.expectEqual(old_buf, d.old_buffers.items[0]);

    // Simulate a thief that observed old_buf before grow and is still reading it.
    try std.testing.expectEqual(@as(u64, 0), old_buf.get(0));
}

test "concurrent push and steal" {
    const num_items: u64 = 10_000;
    var d = try Deque(u64).init(std.testing.allocator);
    defer d.deinit();

    // Shared counter for stolen items
    var stolen_count = std.atomic.Value(u64).init(0);
    var stolen_sum = std.atomic.Value(u64).init(0);

    const StealerContext = struct {
        deque: *Deque(u64),
        stolen_count: *std.atomic.Value(u64),
        stolen_sum: *std.atomic.Value(u64),
        done: *std.atomic.Value(bool),
    };

    var done = std.atomic.Value(bool).init(false);

    const stealer_fn = struct {
        fn run(ctx: StealerContext) void {
            var local_count: u64 = 0;
            var local_sum: u64 = 0;
            while (true) {
                switch (ctx.deque.steal()) {
                    .success => |val| {
                        local_count += 1;
                        local_sum += val;
                    },
                    .empty, .retry => {
                        if (ctx.done.load(.acquire)) break;
                        std.atomic.spinLoopHint();
                    },
                }
            }
            _ = ctx.stolen_count.fetchAdd(local_count, .acq_rel);
            _ = ctx.stolen_sum.fetchAdd(local_sum, .acq_rel);
        }
    }.run;

    // Spawn 4 stealer threads
    var stealers: [4]std.Thread = undefined;
    const ctx = StealerContext{
        .deque = &d,
        .stolen_count = &stolen_count,
        .stolen_sum = &stolen_sum,
        .done = &done,
    };
    for (&stealers) |*t| {
        t.* = try std.Thread.spawn(.{}, stealer_fn, .{ctx});
    }

    // Owner pushes items and pops some
    var owner_count: u64 = 0;
    var owner_sum: u64 = 0;
    for (0..num_items) |i| {
        try d.push(@intCast(i));
        // Occasionally pop from owner side too
        if (i % 3 == 0) {
            if (d.pop()) |val| {
                owner_count += 1;
                owner_sum += val;
            }
        }
    }

    // Signal stealers to stop and drain remaining
    done.store(true, .release);

    // Owner drains remaining
    while (d.pop()) |val| {
        owner_count += 1;
        owner_sum += val;
    }

    for (&stealers) |*t| {
        t.join();
    }

    const total_count = owner_count + stolen_count.load(.acquire);
    const total_sum = owner_sum + stolen_sum.load(.acquire);
    const expected_sum = num_items * (num_items - 1) / 2;

    try std.testing.expectEqual(num_items, total_count);
    try std.testing.expectEqual(expected_sum, total_sum);
}
