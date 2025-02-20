#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texcoord_0;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec3 tangent;


layout(set = 0, binding = 1) uniform UBOScene
{
	mat4 projection;
	mat4 view;
	vec4 lightPos;
	vec4 viewPos;
} ubo_scene;

layout(push_constant) uniform PushConsts {
	mat4 model;
} mesh_info;

layout (location = 1) out vec2 o_uv;

void main() 
{
	o_uv = texcoord_0;
	gl_Position = ubo_scene.projection * ubo_scene.view * mesh_info.model * vec4(position.xyz, 1.0);	
}