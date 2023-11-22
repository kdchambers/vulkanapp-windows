@echo off

@REM dxc.exe -spirv -T vs_6_0 -E main VertexShader.hlsl -Fo .\VertexShader.spv
@REM dxc.exe -spirv -T ps_6_0 -E main PixelShader.hlsl -Fo .\PixelShader.spv

@REM glslang -V -D --target-env vulkan1.2 -e main -S vert VertexShader.hlsl -o VertexShader.spv
@REM glslang -V -D --target-env vulkan1.2 -e main -S frag PixelShader.hlsl -o PixelShader.spv

glslang -V generic.vert -o generic.vert.spv
glslang -V generic.frag -o generic.frag.spv

echo 'Done'
@echo on