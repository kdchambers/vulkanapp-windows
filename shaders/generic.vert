#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec2 inTexCoord;
layout(location = 2) in vec4 inColor;

layout(location = 0) out vec2 outTexCoord;
layout(location = 1) out vec4 outColor;

layout( push_constant ) uniform Block {
    vec2 player_offset;
} PushConstant;

void main() {
    gl_Position = vec4(inPosition.x - PushConstant.player_offset.x, inPosition.y - PushConstant.player_offset.y, 0.0, 1.0);
    outColor = inColor;
    outTexCoord = inTexCoord;
}