const std = @import("std");
const Allocator = std.mem.Allocator;

/// Circular buffer that can be grown but never shrunk.
pub fn Buffer(comptime T: type) type {
    return struct {
        data: []std.atomic.Value(T),
        mask: usize, // len - 1, since len is always power of 2

        const Self = @This();

        /// Hard cap on buffer size: size_exp is u5 (max 31), so 2^31 slots.
        pub const max_size_exp: u5 = 31;

        pub fn init(allocator: Allocator, size_exp: u5) !Self {
            const len: usize = @as(usize, 1) << size_exp;
            const data = try allocator.alloc(std.atomic.Value(T), len);
            // Slots are left uninitialized: Chase-Lev only ever reads indices
            // in [top, bottom), and push() writes before advancing bottom, so
            // no load ever observes these undefined bytes under the protocol.
            for (data) |*slot| {
                slot.* = std.atomic.Value(T).init(undefined);
            }
            return .{
                .data = data,
                .mask = len - 1,
            };
        }

        pub fn deinit(self: *Self, allocator: Allocator) void {
            allocator.free(self.data);
        }

        pub fn get(self: *const Self, index: usize) T {
            return self.data[index & self.mask].load(.unordered);
        }

        pub fn put(self: *Self, index: usize, value: T) void {
            self.data[index & self.mask].store(value, .unordered);
        }

        pub fn grow(self: *const Self, allocator: Allocator, top: usize, bottom: usize) !Self {
            return self.growWithMax(allocator, top, bottom, max_size_exp);
        }

        /// Internal: parameterized for unit tests so we can exercise the
        /// BufferAtMaxSize path without allocating 2^31 slots.
        fn growWithMax(self: *const Self, allocator: Allocator, top: usize, bottom: usize, max_exp: u5) !Self {
            const old_len = self.mask + 1;
            const old_exp = std.math.log2(old_len);
            if (old_exp >= max_exp) return error.BufferAtMaxSize;
            const new_exp: u5 = @intCast(old_exp + 1);
            var new_buf = try Self.init(allocator, new_exp);
            const size = bottom -% top;
            for (0..size) |offset| {
                const i = top +% offset;
                new_buf.put(i, self.get(i));
            }
            return new_buf;
        }
    };
}

test "Buffer.grow returns BufferAtMaxSize at the limit" {
    const Buf = Buffer(u8);
    var buf = try Buf.init(std.testing.allocator, 4);
    defer buf.deinit(std.testing.allocator);
    // Cap max_exp at the current size_exp; grow must refuse.
    try std.testing.expectError(error.BufferAtMaxSize, buf.growWithMax(std.testing.allocator, 0, 0, 4));
}

test "Buffer.grow succeeds below the limit" {
    const Buf = Buffer(u8);
    var buf = try Buf.init(std.testing.allocator, 3);
    defer buf.deinit(std.testing.allocator);
    var grown = try buf.growWithMax(std.testing.allocator, 0, 0, 4);
    defer grown.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 16), grown.mask + 1);
}
