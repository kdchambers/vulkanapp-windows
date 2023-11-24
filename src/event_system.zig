const std = @import("std");
const util = @import("util.zig");
const EventBufferTempl = util.EventBuffer;

pub const RuntimeEvent = enum(u8) {
    //
    // Window Events
    //
    screen_dimensions_changed,
    mouse_position_changed,
    key_pressed,
    shutdown_requested,
};

pub const EventBuffer = EventBufferTempl(RuntimeEvent, 32);
