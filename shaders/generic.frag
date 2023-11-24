#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 inTexCoord;
layout(location = 1) in vec4 inColor;

layout(location = 0) out vec4 outColor;

layout( push_constant ) uniform Block {
    vec2 player_offset;
} PushConstant;

void main() {
    outColor = inColor;
}
