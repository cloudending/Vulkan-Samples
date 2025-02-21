#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texcoord_0;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec4 joints_0;
layout(location = 4) in vec4 weights_0;

layout(set = 0, binding = 1) uniform UBOScene
{
	mat4 projection;
	mat4 view;
	vec4 lightPos;
	vec4 viewPos;
} ubo_scene;

layout(std430, set = 0, binding = 2) readonly buffer JointMatrices {
	mat4 matrix[];
} joint_matrices;

layout(push_constant) uniform PushConsts {
	mat4 model;
} mesh_info;

layout (location = 1) out vec2 o_uv;

void main() 
{
	o_uv = texcoord_0;

	mat4 skin_mat = 
		weights_0.x * joint_matrices.matrix[int(joints_0.x)] +
		weights_0.y * joint_matrices.matrix[int(joints_0.y)] +
		weights_0.z * joint_matrices.matrix[int(joints_0.z)] +
		weights_0.w * joint_matrices.matrix[int(joints_0.w)];

	gl_Position = ubo_scene.projection * ubo_scene.view * mesh_info.model * skin_mat * vec4(position.xyz, 1.0);	
}