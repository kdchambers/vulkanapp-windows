struct VSInput
{
[[vk::location(0)]] float2 Position : POSITION0;
[[vk::location(1)]] float2 TexUV : TEXCOORD0;
[[vk::location(2)]] float4 Color : COLOR0;
};

struct VSOutput
{
	float4 Pos : SV_POSITION;
[[vk::location(0)]] float2 TexUV : TEXCOORD0;
[[vk::location(1)]] float4 Color : COLOR0;
};

VSOutput main(VSInput input, uint VertexIndex : SV_VertexID)
{
	VSOutput output = (VSOutput)0;
	output.Color = input.Color;
	output.Pos = float4(input.Position, 0.0f, 1.0f);
    output.TexUV = input.TexUV;
	return output;
}