
[[vk::binding(0)]] Texture2DArray<float4> texture : register(t0);
[[vk::binding(0)]] SamplerState sampler1 : register(s0);

struct VSInput
{
[[vk::location(0)]] float2 TexUV : TEXCOORD0;
[[vk::location(1)]] float4 Color : COLOR0;
};

struct VSOutput
{
	float4 Color : SV_Target;
};

VSOutput main(VSInput input)
{
    VSOutput output = (VSOutput)0;
    output.Color = texture.Sample(sampler1, input.TexUV.x, input.TexUV.y, 0) * input.Color;
    return output;
}
