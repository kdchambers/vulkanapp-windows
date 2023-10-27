@echo off
dxc.exe -spirv -T vs_6_0 -E main VertexShader.hlsl -Fo .\VertexShader.spv
dxc.exe -spirv -T ps_6_0 -E main PixelShader.hlsl -Fo .\PixelShader.spv
echo 'Done'
@echo on