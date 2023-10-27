@echo off

dxc.exe -spirv -T vs_6_0 -E main VertexShader.hlsl -Fo .\VertexShader.spv
dxc.exe -spirv -T ps_6_0 -E main PixelShader.hlsl -Fo .\PixelShader.spv

glslang -V -D --target-env vulkan1.2 -e main -S vert VertexShader.hlsl -o VertexShader.spv
glslang -V -D --target-env vulkan1.2 -e main -S frag PixelShader.hlsl -o PixelShader.spv

echo 'Done'
@echo on