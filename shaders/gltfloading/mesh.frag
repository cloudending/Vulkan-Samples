#version 450

layout(set = 0, binding = 0) uniform sampler2D base_color_texture;

layout(location = 1) in vec2 in_uv;

layout (location = 0) out vec4 outFragColor;
void main() 
{
	vec4 color = texture(base_color_texture, in_uv);
	outFragColor = color;		
}