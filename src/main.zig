// SPDX-License-Identifier: MIT
// Copyright (c) 2023 Keith Chambers

const std = @import("std");
const assert = std.debug.assert;
const win = std.os.windows;
const user32 = win.user32;
const window_client = @import("window_client.zig");
const game = @import("game.zig");
const renderer = @import("renderer.zig");
const util = @import("util.zig");
const event_system = @import("event_system.zig");
const EventBuffer = event_system.EventBuffer;

var default_allocator: std.mem.Allocator = undefined;

pub fn wWinMain(hinstance: win.HINSTANCE, prev_instance: ?win.HINSTANCE, cmd_line: win.PWSTR, cmd_show_count: win.INT) win.INT {
    _ = cmd_line;
    _ = prev_instance;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var allocator = gpa.allocator();
    default_allocator = allocator;

    window_client.init(hinstance, cmd_show_count) catch return 1;
    defer window_client.deinit();

    renderer.init(default_allocator) catch return 1;
    defer renderer.deinit(default_allocator);

    game.init() catch return 1;
    defer game.deinit();

    doAppLoop() catch return 1;

    std.log.info("Terminated cleanly", .{});

    return 0;
}

fn doAppLoop() !void {
    var event_buffer = EventBuffer.create();
    while (!window_client.is_shutdown_requested) {
        try window_client.processEvents(&event_buffer);

        for (event_buffer.buffer[0..event_buffer.count]) |event| {
            if (event == .shutdown_requested) {
                std.log.info("Shutting down application", .{});
            }
        }

        try game.processEvents(&event_buffer);
        try renderer.processEvents(&event_buffer);

        event_buffer.clear();
    }
}
