const std = @import("std");
const win = std.os.windows;
const user32 = win.user32;
const util = @import("util.zig");
const geometry = util.geometry;
const event_system = @import("event_system.zig");
const EventBuffer = event_system.EventBuffer;

/// Screen dimensions of the application, as reported by wayland
/// Initial values are arbirary and will be updated once the wayland
/// server reports a change
pub var screen_dimensions = geometry.Dimensions2D(u32){
    .width = 1040,
    .height = 640,
};

var mouse_coordinates = geometry.Coordinates2D(f64){ .x = 0.0, .y = 0.0 };
var is_mouse_in_screen = false;

pub var app_hinstance: win.HINSTANCE = undefined;
pub var window_handle: win.HWND = undefined;
var default_cursor: win.HCURSOR = undefined;

pub var is_shutdown_requested: bool = false;

const IDC_ARROW: [*:0]const u16 = makeIntreSourceW(32512);

fn makeIntreSourceW(comptime value: u64) [*:0]const u16 {
    return @as([*:0]const u16, @ptrFromInt(value));
}

pub var controls: packed struct(u32) {
    up: bool = false,
    left: bool = false,
    right: bool = false,
    down: bool = false,
    reserved: u28 = 0,
} = .{};

//
// This is a hack, I should just load windows headers
//
extern fn LoadCursorW(hwnd: ?win.HINSTANCE, lpCursorName: [*:0]const u16) callconv(.C) win.HCURSOR;
extern fn SetCursor(cursor: ?win.HCURSOR) callconv(.C) win.HCURSOR;

pub fn init(hinstance: win.HINSTANCE, cmd_show_count: win.INT) !void {
    app_hinstance = hinstance;

    const class_name: [:0]const u8 = "Sample Window Class";

    const wc = user32.WNDCLASSEXW{
        .lpfnWndProc = windowCallback,
        .hInstance = hinstance,
        .lpszClassName = std.unicode.utf8ToUtf16LeStringLiteral(class_name),
        .hIcon = null,
        .hCursor = null,
        .hbrBackground = null,
        .lpszMenuName = null,
        .hIconSm = null,
        .style = 0,
    };

    const class_atom: win.ATOM = user32.RegisterClassExW(&wc);
    _ = class_atom;

    window_handle = user32.createWindowExW(
        0,
        std.unicode.utf8ToUtf16LeStringLiteral(class_name),
        std.unicode.utf8ToUtf16LeStringLiteral("Vulkan App"),
        user32.WS_OVERLAPPEDWINDOW,
        user32.CW_USEDEFAULT,
        user32.CW_USEDEFAULT,
        user32.CW_USEDEFAULT,
        user32.CW_USEDEFAULT,
        null,
        null,
        hinstance,
        null,
    ) catch |err| {
        std.log.err("Failed to create window. Error: {}", .{err});
        return error.CreateWindowFail;
    };

    _ = user32.showWindow(window_handle, cmd_show_count);

    default_cursor = LoadCursorW(null, IDC_ARROW);
}

pub fn deinit() void {}

var key_pressed: bool = false;

pub fn processEvents(event_buffer: *EventBuffer) !void {
    screen_dimensions_changed = false;

    var message: user32.MSG = undefined;
    if (try user32.peekMessageW(&message, null, 0, 0, user32.PM_REMOVE)) {
        if (message.message == user32.WM_QUIT) {
            try event_buffer.write(.shutdown_requested);
            is_shutdown_requested = true;

            std.log.info("App terminate requested", .{});
        } else {
            _ = user32.translateMessage(&message);
            _ = user32.dispatchMessageW(&message);
        }
    }

    if (key_pressed) {
        try event_buffer.write(.key_pressed);
    }

    if (screen_dimensions_changed) {
        try event_buffer.write(.screen_dimensions_changed);
    }

    try event_buffer.write(.key_pressed);
}

pub var screen_dimensions_changed: bool = false;

fn windowCallback(hwnd: win.HWND, umessage: win.UINT, wparam: win.WPARAM, lparam: win.LPARAM) callconv(.C) win.LRESULT {
    switch (umessage) {
        // user32.WM_MOUSEACTIVATE => std.log.info("WM_MOUSEACTIVATE", .{}),
        user32.WM_CAPTURECHANGED => {
            // is_mouse_in_screen = false;
            std.log.info("WM_CAPTURECHANGED", .{});
        },
        user32.WM_TIMER => {
            std.log.info("Timer tick", .{});
            std.debug.assert(false);
            // return 0;
        },
        user32.WM_SETFOCUS => std.log.info("WM_SETFOCUS", .{}),
        user32.WM_KILLFOCUS => std.log.info("WM_KILLFOCUS", .{}),
        // user32.WM_SYSCOMMAND => std.log.info("WM_SYSCOMMAND", .{}),
        user32.WM_CLOSE => std.log.info("WM_CLOSE", .{}),
        user32.WM_INPUTLANGCHANGE => std.log.info("WM_INPUTLANGCHANGE", .{}),
        user32.WM_CHAR => {
            std.log.info("WM_CHAR", .{});
            var utf8_buffer: [4]u8 = undefined;
            const len: u32 = std.unicode.utf8Encode(@intCast(wparam), &utf8_buffer) catch unreachable;
            std.debug.assert(len == 1);
            std.log.info("Char: {c}", .{utf8_buffer[0]});
        },
        user32.WM_SYSCHAR => std.log.info("WM_CHAR, WM_SYSCHAR", .{}),
        user32.WM_UNICHAR => std.log.info("WM_UNICHAR", .{}),
        user32.WM_KEYDOWN => {
            std.log.info("Keydown!!. wparam: {d}", .{wparam});

            switch (wparam) {
                0x57 => controls.up = true,
                0x53 => controls.down = true,
                0x41 => controls.left = true,
                0x44 => controls.right = true,
                else => {},
            }
        },
        user32.WM_KEYUP => {
            switch (wparam) {
                0x57 => controls.up = false,
                0x53 => controls.down = false,
                0x41 => controls.left = false,
                0x44 => controls.right = false,
                else => {},
            }
        },
        user32.WM_SYSKEYDOWN, user32.WM_SYSKEYUP => std.log.info("user32.WM_SYSKEYDOWN, user32.WM_SYSKEYUP", .{}),
        user32.WM_INPUT => std.log.info("WM_INPUT", .{}),
        user32.WM_MOUSELEAVE => std.log.info("WM_MOUSELEAVE", .{}),
        user32.WM_MOUSEWHEEL => std.log.info("WM_MOUSEWHEEL", .{}),
        user32.WM_MOUSEHWHEEL => std.log.info("WM_MOUSEHWHEEL", .{}),
        user32.WM_ENTERSIZEMOVE => {
            std.log.info("WM_ENTERSIZEMOVE", .{});
            //
            // Drop back to about 30fps
            //
            const time_interval_ms: win.UINT = 6;
            wm_timer_id = user32.setTimer(null, 0, time_interval_ms, &timerCallback) catch |err| {
                std.log.err("Failed to set timer. Err: {}", .{err});
                return 0;
            };
            return 0;
        },
        user32.WM_ENTERMENULOOP => std.log.info("WM_ENTERMENULOOP", .{}),
        user32.WM_EXITSIZEMOVE => {
            std.log.info("WM_EXITSIZEMOVE", .{});
            user32.killTimer(null, wm_timer_id) catch |err| {
                std.log.info("Failed to fill wm timer. Error: {}", .{err});
            };
            return 0;
        },
        user32.WM_EXITMENULOOP => std.log.info("WM_EXITMENULOOP", .{}),
        user32.WM_MOVE => std.log.info("WM_MOVE", .{}),
        user32.WM_SIZING => {
            if (wparam >= 1 and wparam <= 8) {
                const window_edge: WindowEdge = @enumFromInt(wparam);
                _ = window_edge;
                // std.log.info("WM_SIZING edge: {}", .{window_edge});
            } else {
                std.log.info("Invalid window edge: {d}", .{wparam});
            }
            return 1;
        },
        user32.WM_GETMINMAXINFO => {
            const info: *MinMaxInfo = @ptrFromInt(@as(usize, @intCast(lparam)));
            _ = info;
            // std.log.info("WM_GETMINMAXINFO", .{});
            // std.log.info("min_track_size: ({d}, {d})", .{ info.min_track_size.x, info.min_track_size.y});
        },
        // user32.WM_PAINT => std.log.info("WM_PAINT", .{}),
        user32.WM_ERASEBKGND => {
            std.log.info("WM_ERASEBKGND", .{});
            // processFrame(default_allocator, &graphics_context) catch return 1;
            return 1;
        },
        // user32.WM_NCACTIVATE => std.log.info("WM_NCACTIVATE", .{}),
        // user32.WM_NCPAINT => std.log.info("WM_NCPAINT", .{}),
        // user32.WM_DWMCOMPOSITIONCHANGED => std.log.info("WM_DWMCOMPOSITIONCHANGED", .{}),
        // user32.WM_DWMCOLORIZATIONCOLORCHANGED => std.log.info("WM_DWMCOLORIZATIONCOLORCHANGED", .{}),
        // user32.WM_GETDPISCALEDSIZE => std.log.info("WM_GETDPISCALEDSIZE", .{}),
        // user32.WM_DPICHANGED => std.log.info("WM_DPICHANGED", .{}),
        user32.WM_SETCURSOR => {
            const HTCLIENT: u64 = 1;
            if (lparam & @as(u64, 0xffff) == HTCLIENT) {
                // std.log.info("WM_SETCURSOR", .{});
                _ = SetCursor(default_cursor);
            } else {
                // std.log.info("Cursor outside of client area", .{});
            }
            // return win.TRUE;
        },
        user32.WM_DROPFILES => std.log.info("WM_DROPFILES", .{}),
        user32.WM_DESTROY => {
            user32.postQuitMessage(0);
            return 0;
        },
        // user32.WM_CHAR => {
        //     std.log.info("Char pressed: {d}", .{wparam});
        // },
        user32.WM_MOUSEACTIVATE => {
            std.log.info("Window activated", .{});
            return 0;
        },
        user32.WM_LBUTTONUP => {
            std.log.info("Mouse button released", .{});

            if (!is_mouse_button_down) {
                return 0;
            }
            std.log.info("Mouse clicked!", .{});
            is_mouse_button_down = false;
            return 0;
        },
        user32.WM_LBUTTONDOWN => {
            is_mouse_button_down = true;
            return 0;
        },
        user32.WM_MOUSEMOVE => {
            const x: i32 = @intCast(lparam & @as(isize, 0xfff));
            const y: i32 = @intCast((lparam >> 16) & @as(isize, 0xffff));
            mouse_coordinates = .{
                .x = @floatFromInt(x),
                .y = @floatFromInt(y),
            };
            // std.log.info("Mouse: {d} x {d}", .{ x, y });
            return 0;
        },
        user32.WM_SIZE => {
            std.log.info("Window resizing", .{});
            screen_dimensions_changed = true;
            const width: i32 = @intCast(lparam & @as(isize, 0xfff));
            const height: i32 = @intCast((lparam >> 16) & @as(isize, 0xffff));
            screen_dimensions = .{
                .width = @intCast(width),
                .height = @intCast(height),
            };
            return 0;
        },
        else => {
            // std.log.info("Unknown event..", .{});
            // return user32.defWindowProcW(hwnd, umessage, wparam, lparam);
        },
    }
    return user32.defWindowProcW(hwnd, umessage, wparam, lparam);
    // return 0;
}

var is_mouse_button_down: bool = false;

const WindowEdge = enum(win.WPARAM) {
    left = 1,
    right = 2,
    top = 3,
    top_left = 4,
    top_right = 5,
    bottom = 6,
    bottom_left = 7,
    bottom_right = 8,
};

const Point = extern struct {
    x: i32,
    y: i32,
};

const MinMaxInfo = extern struct {
    reserved: Point,
    max_size: Point,
    max_position: Point,
    min_track_size: Point,
    max_track_size: Point,
};

var wm_timer_id: win.UINT_PTR = undefined;

fn timerCallback(hWnd: win.HWND, message: win.UINT, idTimer: win.UINT_PTR, dwTime: win.DWORD) callconv(win.WINAPI) void {
    _ = hWnd;
    _ = message;
    _ = idTimer;
    _ = dwTime;

    // const current_ts: i128 = std.time.nanoTimestamp();
    // const delta_ts = current_ts - game_current_ts;
    // const delta_s: f32 = @floatFromInt(@divTrunc(delta_ts, std.time.ns_per_s));
    // game_current_ts = current_ts;

    // processFrame(default_allocator, &graphics_context, delta_s) catch |err| {
    //     std.log.warn("Failed to process frame. Error: {}", .{err});
    //     return;
    // };
}
