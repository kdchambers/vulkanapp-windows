const std = @import("std");
const assert = std.debug.assert;
const renderer = @import("renderer.zig");
const event_system = @import("event_system.zig");
const EventBuffer = event_system.EventBuffer;
const util = @import("util.zig");
const geometry = util.geometry;
const graphics = util.graphics;
const normalizePoint = util.normalizePoint;
const normalizeDist = util.normalizeDist;
const window_client = @import("window_client.zig");

const pixels_per_s: f32 = 100;

/// The x offset of the game when the current frame was rendered
var frame_offset_px: f32 = 0;

/// The distance from the beginning of the frame to the end of the last obstacle
var frame_width_px: f32 = 0;

const ground_height_px: f32 = 40;

pub var game_lost: bool = false;

const velocity_pixels: f32 = 5.0;

var player_position = geometry.Coordinates2D(f32){
    .x = 20.0,
    .y = ground_height_px,
};

var controls: packed struct {
    up: bool = false,
    left: bool = false,
    right: bool = false,
    down: bool = false,
} = .{};

pub var screen_offset_px: f32 = 0;

var prng: std.rand.Xoshiro256 = undefined;
var random: std.rand.Random = undefined;

var game_start_ts: i128 = 0;
var game_current_ts: i128 = 0;

fn drawEndScreen() !void {
    renderer.clearVertices();
    const screen_background_color = graphics.RGBA(f32).fromInt(u8, 240, 20, 20, 255);
    const screen_extent = geometry.Extent2D(f32){
        .x = -1.0,
        .y = -1.0,
        .width = 2.0,
        .height = 2.0,
    };
    try renderer.addQuadColored(screen_extent, screen_background_color, .top_left);
}

fn drawPausedScreen() !void {
    renderer.clearVertices();
    const screen_background_color = graphics.RGBA(f32).fromInt(u8, 240, 20, 200, 255);
    const screen_extent = geometry.Extent2D(f32){
        .x = -1.0,
        .y = -1.0,
        .width = 2.0,
        .height = 2.0,
    };
    try renderer.addQuadColored(screen_extent, screen_background_color, .top_left);
}

fn needExtentGeometry(screen_width: f32, screen_offset: f32) bool {
    const visible_x_offset: f32 = screen_offset + screen_width;
    const frame_visible_until: f32 = frame_offset_px + frame_width_px;
    const needs_extending: bool = visible_x_offset >= frame_visible_until;
    return needs_extending;
}

var is_meshdraw_required: bool = true;

const player_dimensions: geometry.Dimensions2D(f32) = .{
    .width = 30,
    .height = 40,
};

pub fn init() !void {
    prng = std.rand.DefaultPrng.init(0);
    random = prng.random();

    game_start_ts = std.time.nanoTimestamp();
    game_current_ts = game_start_ts;

    obstacle_buffer.init();
}

pub fn deinit() void {}

pub fn processEvents(event_buffer: *EventBuffer) !void {
    switch (game_state) {
        .playing => try updateGame(event_buffer),
        .paused => try drawPausedScreen(),
        .finished => try drawEndScreen(),
    }
}

fn updateGame(event_buffer: *EventBuffer) !void {
    game_current_ts = std.time.nanoTimestamp();

    const delta_ts: f32 = @floatFromInt(game_current_ts - game_start_ts);
    const delta_s: f32 = delta_ts / std.time.ns_per_s;

    const screen_height: f32 = @floatFromInt(window_client.screen_dimensions.height);
    const screen_width: f32 = @floatFromInt(window_client.screen_dimensions.width);

    for (event_buffer.buffer[0..event_buffer.count]) |event| {
        switch (event) {
            .key_pressed => {
                if (window_client.controls.up)
                    player_position.y -= velocity_pixels;
                if (window_client.controls.down)
                    player_position.y += velocity_pixels;
                if (window_client.controls.left)
                    player_position.x -= velocity_pixels;
                if (window_client.controls.right)
                    player_position.x += velocity_pixels;
            },
            else => {},
        }
    }

    const obs_color = graphics.RGBA(f32){ .r = 0.7, .g = 0.4, .b = 0.3, .a = 1.0 };
    const obstacle_baseline_px: f32 = ground_height_px;

    screen_offset_px = delta_s * pixels_per_s;

    const player_extent = geometry.Extent2D(f32){
        .x = normalizePoint(player_position.x, screen_width),
        .y = normalizePoint(player_position.y, screen_height),
        .width = normalizeDist(player_dimensions.width, screen_width),
        .height = normalizeDist(player_dimensions.height, screen_height),
    };
    const player_color = graphics.RGBA(f32){ .r = 0.1, .g = 0.8, .b = 0.1, .a = 1.0 };

    const player_end_x: f32 = normalizeDist(screen_offset_px, screen_width) + player_extent.x + player_extent.width;
    const player_start_x: f32 = normalizeDist(screen_offset_px, screen_width) + player_extent.x;

    const player_bottom_y: f32 = player_extent.y;

    for (obstacle_buffer.buffer) |obstacle| {
        const obs_extent = geometry.Extent2D(f32){
            .x = normalizePoint(obstacle.x, screen_width),
            .y = normalizePoint(screen_height - obstacle_baseline_px, screen_height),
            .width = normalizeDist(obstacle.width, screen_width),
            .height = normalizeDist(obstacle.height, screen_height),
        };
        const obs_end: f32 = obs_extent.x + obs_extent.width;
        const obs_start: f32 = obs_extent.x;

        const end_intercepts: bool = (player_end_x > obs_start and player_end_x < obs_end);
        const start_intercepts: bool = player_start_x > obs_start and player_start_x < obs_end;

        if (start_intercepts or end_intercepts) {
            if (player_bottom_y > (obs_extent.y - obs_extent.height)) {
                std.log.info("Game over!", .{});
                game_state = .finished;
                game_lost = true;
            }
        }
    }

    if (needExtentGeometry(screen_width, screen_offset_px)) {
        std.log.warn("Geometry needs to be extended", .{});
        obstacle_buffer.extentGeometry(screen_offset_px);

        frame_width_px = obstacle_buffer.frameWidth();
        frame_offset_px = obstacle_buffer.frameStart();

        is_meshdraw_required = true;
    }

    if (!is_meshdraw_required) {
        try renderer.overwriteQuadColored(0, player_extent, player_color, .bottom_left);
        return;
    }

    renderer.clearVertices();

    try renderer.addQuadColored(player_extent, player_color, .bottom_left);

    is_meshdraw_required = false;

    const ground_color = graphics.RGBA(f32).fromInt(u8, 10, 200, 30, 255);
    const ground_extent = geometry.Extent2D(f32){
        .x = -1.0,
        .y = 1.0,
        //
        // TODO: needExtentGeometry is wrong, this 200 buffer shouldn't be needed
        //
        .width = normalizeDist(frame_width_px, screen_width + 200),
        .height = normalizeDist(ground_height_px, screen_height),
    };

    try renderer.addQuadColored(ground_extent, ground_color, .bottom_left);

    for (obstacle_buffer.buffer) |obstacle| {
        const obs_extent = geometry.Extent2D(f32){
            .x = normalizePoint(obstacle.x, screen_width),
            .y = normalizePoint(screen_height - obstacle_baseline_px, screen_height),
            .width = normalizeDist(obstacle.width, screen_width),
            .height = normalizeDist(obstacle.height, screen_height),
        };
        try renderer.addQuadColored(obs_extent, obs_color, .bottom_left);
    }
}

const GameState = enum {
    playing,
    paused,
    finished,
};
pub var game_state: GameState = .playing;

const Obstacles = struct {
    const Obstacle = struct {
        x: f32,
        width: f32,
        height: f32,
    };

    const buffer_size = 12;
    const variance: f32 = 300.0;

    buffer: [buffer_size]Obstacle,

    pub fn init(self: *@This()) void {
        var current_x: f32 = 340;

        const obstacle_interval: f32 = 220;

        for (&self.buffer) |*obstacle| {
            obstacle.* = .{ .x = current_x, .height = 40, .width = 20 };
            current_x += (obstacle_interval + (random.float(f32) * variance));
        }
    }

    pub fn frameWidth(self: @This()) f32 {
        return (self.buffer[buffer_size - 1].x + self.buffer[buffer_size - 1].width) - self.buffer[0].x;
    }

    pub fn frameStart(self: @This()) f32 {
        return self.buffer[0].x;
    }

    pub fn visibleObstacles(self: *@This(), x_offset: f32, width: f32) []const Obstacle {
        const start_index: usize = blk: {
            for (self.buffer, 0..) |obstacle, obstacle_i| {
                if ((obstacle.x + obstacle.width) > x_offset) {
                    break :blk obstacle_i;
                }
            }
            unreachable;
        };
        const end_index: usize = blk: {
            const x_max: f32 = x_offset + width;
            for (self.buffer[start_index..], 0..) |obstacle, obstacle_i| {
                if (obstacle.x > x_max) {
                    break :blk obstacle_i + start_index;
                }
            }
            break :blk buffer_size - 1;
        };

        assert(start_index < end_index);
        const visible_ostacles = self.buffer[start_index..end_index];

        return visible_ostacles;
    }

    pub fn extentGeometry(self: *@This(), min_x_threshold: f32) void {
        const keep_index: usize = blk: {
            for (self.buffer, 0..) |obstacle, obstacle_i| {
                if ((obstacle.x + obstacle.width) >= min_x_threshold) {
                    break :blk obstacle_i;
                }
            }
            unreachable;
        };

        const obstacle_interval: f32 = 220;
        var current_x: f32 = self.buffer[buffer_size - 1].x + obstacle_interval;

        const keep_count: usize = buffer_size - keep_index;
        for (0..keep_count, keep_index..buffer_size) |dst_i, src_i| {
            self.buffer[dst_i] = self.buffer[src_i];
        }

        for (self.buffer[keep_count..]) |*obstacle| {
            obstacle.* = .{ .x = current_x, .height = 40, .width = 20 };
            current_x += (obstacle_interval + (random.float(f32) * variance));
        }
    }
};

var obstacle_buffer: Obstacles = undefined;
