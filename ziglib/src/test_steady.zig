const std = @import("std");
const expectEqual = std.testing.expectEqual;
const expect = std.testing.expect;
const steady = @import("steady.zig");

test "test_get_num_bins" {
    try expectEqual(steady.get_num_bins(1), 0);
    try expectEqual(steady.get_num_bins(2), 1);
    try expectEqual(steady.get_num_bins(4), 2);
    try expectEqual(steady.get_num_bins(8), 4);
    try expectEqual(steady.get_num_bins(16), 8);
}

test "test_get_num_segments" {
    try expectEqual(steady.get_num_segments(1), 0);
    try expectEqual(steady.get_num_segments(2), 1);
    try expectEqual(steady.get_num_segments(4), 2);
    try expectEqual(steady.get_num_segments(8), 3);
    try expectEqual(steady.get_num_segments(16), 4);
    try expectEqual(steady.get_num_segments(32), 5);
    try expectEqual(steady.get_num_segments(64), 6);
    try expectEqual(steady.get_num_segments(128), 7);
    try expectEqual(steady.get_num_segments(256), 8);
    try expectEqual(steady.get_num_segments(512), 9);
    try expectEqual(steady.get_num_segments(1024), 10);
    try expectEqual(steady.get_num_segments(2048), 11);
}

test "test_get_nth_bin_width" {
    var surface_size: u32 = 1;
    while (surface_size <= 1 << 19) : (surface_size *= 2) {
        const num_bins = steady.get_num_bins(surface_size);
        var bins = std.ArrayList(u32).init(std.heap.page_allocator);
        defer bins.deinit();

        // Populate bins based on get_nth_bin_width.
        var n: u32 = 0;
        while (n < num_bins) : (n += 1) {
            bins.append(steady.get_nth_bin_width(n, surface_size)) catch unreachable;
        }

        if (surface_size == 1) {
            // Special case the trivial case.
            try expect(bins.items.len == 0);
            continue;
        }

        // Bin widths should be nonincreasing.
        var i: usize = 1;
        while (i < bins.items.len) : (i += 1) {
            try expect(bins.items[i - 1] >= bins.items[i]);
        }

        // Additional checks converted from Python to Zig.
        // Note: Zig does not have a direct equivalent for Python's Counter,
        // so you would need to implement counting logic manually.

        // More logic here...
    }
}
