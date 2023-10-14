const std = @import("std");

const Builder = std.build.Builder;
const Build = std.build;
const Pkg = Build.Pkg;

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "vulkanapp-windows",
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const unit_tests = b.addTest(.{
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    const vkzig_dep = b.dependency("vulkan_zig", .{
        .registry = @as([]const u8, b.pathFromRoot("vendor/vk.xml")),
    });
    const vkzig_bindings = vkzig_dep.module("vulkan-zig");
    exe.addModule("vulkan-zig", vkzig_bindings);

    const zigimg_dep = b.dependency("zigimg", .{});
    const zigimg_bindings = zigimg_dep.module("zigimg");
    exe.addModule("zigimg", zigimg_bindings);

    const shaders_module = b.createModule(.{
        .source_file = .{ .path = "shaders/shaders.zig" },
        .dependencies = &.{},
    });
    exe.addModule("shaders", shaders_module);

    const run_unit_tests = b.addRunArtifact(unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);
}
