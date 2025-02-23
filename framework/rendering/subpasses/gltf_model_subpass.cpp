/* Copyright (c) 2019-2024, Arm Limited and Contributors
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 the "License";
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "rendering/subpasses/gltf_model_subpass.h"
#include "common/utils.h"
#include "common/vk_common.h"
#include "rendering/render_context.h"
#include "scene_graph/components/camera.h"
#include "scene_graph/components/image.h"
#include "scene_graph/components/material.h"
#include "scene_graph/components/mesh.h"
#include "scene_graph/components/pbr_material.h"
#include "scene_graph/components/texture.h"
#include "scene_graph/node.h"
#include "scene_graph/scene.h"

#define MAX_FORWARD_LIGHT_COUNT 2


static int render_count = 0;


struct alignas(16) ForwardLights1
{
	vkb::rendering::Light directional_lights[MAX_FORWARD_LIGHT_COUNT];
	vkb::rendering::Light point_lights[MAX_FORWARD_LIGHT_COUNT];
	vkb::rendering::Light spot_lights[MAX_FORWARD_LIGHT_COUNT];
};

namespace vkb
{
GLTFModelSubpass::GLTFModelSubpass(RenderContext &render_context, ShaderSource &&vertex_source, ShaderSource &&fragment_source, sg::Scene &scene_, sg::Camera &camera) :
    Subpass{render_context, std::move(vertex_source), std::move(fragment_source)},
    meshes{scene_.get_components<sg::Mesh>()},
    camera{camera},
    scene{scene_}
{
}

void GLTFModelSubpass::prepare()
{

	auto &device = get_render_context().get_device();
	for (auto &mesh : meshes)
	{
		for (auto &sub_mesh : mesh->get_submeshes())
		{
			auto &variant = sub_mesh->get_mut_shader_variant();

			// Same as Geometry except adds lighting definitions to sub mesh variants.
			//variant.add_definitions({"MAX_LIGHT_COUNT " + std::to_string(MAX_FORWARD_LIGHT_COUNT)});

			//variant.add_definitions(vkb::rendering::light_type_definitions);

			auto &vert_module = device.get_resource_cache().request_shader_module(VK_SHADER_STAGE_VERTEX_BIT, get_vertex_shader(), variant);
			auto &frag_module = device.get_resource_cache().request_shader_module(VK_SHADER_STAGE_FRAGMENT_BIT, get_fragment_shader(), variant);
		}
	}

	
	
	// Build all shader variance upfront
	//auto &device = get_render_context().get_device();
	//for (auto &mesh : meshes)
	//{
	//	for (auto &sub_mesh : mesh->get_submeshes())
	//	{
	//		auto &variant     = sub_mesh->get_shader_variant();
	//		auto &vert_module = device.get_resource_cache().request_shader_module(VK_SHADER_STAGE_VERTEX_BIT, get_vertex_shader(), variant);
	//		auto &frag_module = device.get_resource_cache().request_shader_module(VK_SHADER_STAGE_FRAGMENT_BIT, get_fragment_shader(), variant);
	//	}
	//}
}

void GLTFModelSubpass::get_sorted_nodes(std::multimap<float, std::pair<sg::Node *, sg::SubMesh *>> &opaque_nodes, std::multimap<float, std::pair<sg::Node *, sg::SubMesh *>> &transparent_nodes)
{
	auto camera_transform = camera.get_node()->get_transform().get_world_matrix();

	for (auto &mesh : meshes)
	{
		for (auto &node : mesh->get_nodes())
		{
			auto node_transform = node->get_transform().get_world_matrix();

			const sg::AABB &mesh_bounds = mesh->get_bounds();

			sg::AABB world_bounds{mesh_bounds.get_min(), mesh_bounds.get_max()};
			world_bounds.transform(node_transform);

			float distance = glm::length(glm::vec3(camera_transform[3]) - world_bounds.get_center());

			for (auto &sub_mesh : mesh->get_submeshes())
			{
				if (sub_mesh->get_material()->alpha_mode == sg::AlphaMode::Blend)
				{
					transparent_nodes.emplace(distance, std::make_pair(node, sub_mesh));
				}
				else
				{
					opaque_nodes.emplace(distance, std::make_pair(node, sub_mesh));
				}
			}
		}
	}
}

/// <summary>
/// Draw every Node of gltf
/// </summary>
/// <param name="command_buffer"></param>
void GLTFModelSubpass::draw(CommandBuffer &command_buffer)
{
	render_count = 0;
	allocate_lights<ForwardLights1>(scene.get_components<sg::Light>(), MAX_FORWARD_LIGHT_COUNT);
	command_buffer.bind_lighting(get_lighting_state(), 0, 4);

	std::multimap<float, std::pair<sg::Node *, sg::SubMesh *>> opaque_nodes;
	std::multimap<float, std::pair<sg::Node *, sg::SubMesh *>> transparent_nodes;

	get_sorted_nodes(opaque_nodes, transparent_nodes);

	// Draw opaque objects in front-to-back order
	{
		ScopedDebugLabel opaque_debug_label{command_buffer, "Opaque objects"};

		for (auto node_it = opaque_nodes.begin(); node_it != opaque_nodes.end(); node_it++)
		{
			update_uniform(command_buffer, *node_it->second.first, thread_index);

			update_joint_matrix(command_buffer, *node_it->second.first, thread_index);

			// Invert the front face if the mesh was flipped
			const auto &scale      = node_it->second.first->get_transform().get_scale();
			bool        flipped    = scale.x * scale.y * scale.z < 0;
			VkFrontFace front_face = flipped ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;

			draw_submesh(command_buffer, *node_it->second.second, *node_it->second.first, front_face);
		}
	}

	// Enable alpha blending
	ColorBlendAttachmentState color_blend_attachment{};
	color_blend_attachment.blend_enable           = VK_TRUE;
	color_blend_attachment.src_color_blend_factor = VK_BLEND_FACTOR_SRC_ALPHA;
	color_blend_attachment.dst_color_blend_factor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	color_blend_attachment.src_alpha_blend_factor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;

	ColorBlendState color_blend_state{};
	color_blend_state.attachments.resize(get_output_attachments().size());
	for (auto &it : color_blend_state.attachments)
	{
		it = color_blend_attachment;
	}
	command_buffer.set_color_blend_state(color_blend_state);

	command_buffer.set_depth_stencil_state(get_depth_stencil_state());

	// Draw transparent objects in back-to-front order
	{
		ScopedDebugLabel transparent_debug_label{command_buffer, "Transparent objects"};

		for (auto node_it = transparent_nodes.rbegin(); node_it != transparent_nodes.rend(); node_it++)
		{
			update_uniform(command_buffer, *node_it->second.first, thread_index);
			update_joint_matrix(command_buffer, *node_it->second.first, thread_index);
			draw_submesh(command_buffer, *node_it->second.second, *node_it->second.first);
		}
	}

	//LOGI("render mesh count:{}", render_count);
}

void GLTFModelSubpass::update_uniform(CommandBuffer &command_buffer, sg::Node &node, size_t thread_index)
{
	UBOScene ubo_scene;
	ubo_scene.view = camera.get_view();
	ubo_scene.projection = camera.get_pre_rotation() * vkb::rendering::vulkan_style_projection(camera.get_projection());
	ubo_scene.viewPos  = glm::vec4(glm::inverse(camera.get_view())[3]);
	ubo_scene.lightPos = glm::vec4(5.0f, 5.0f, -5.0f, 1.0f);
	auto &render_frame = get_render_context().get_active_frame();
	auto allocation = render_frame.allocate_buffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sizeof(UBOScene), thread_index);
	allocation.update(ubo_scene);

	command_buffer.bind_buffer(allocation.get_buffer(), allocation.get_offset(), allocation.get_size(), 0, 1, 0);
}

void GLTFModelSubpass::update_joint_matrix(CommandBuffer &command_buffer, sg::Node &node, size_t thread_index)
{
	sg::Skin *skin = node.get_skin();
	if (skin) {
		node.update_joint_matrix();
		command_buffer.bind_buffer(*skin->inverse_bind_matrices_ssbo, 0, skin->inverse_bind_matrices.size() * sizeof(glm::mat4), 0, 2, 0);
		
	}
}

void GLTFModelSubpass::draw_submesh(CommandBuffer &command_buffer, sg::SubMesh &sub_mesh, sg::Node& node, VkFrontFace front_face)
{
	auto &device = command_buffer.get_device();

	ScopedDebugLabel submesh_debug_label{command_buffer, sub_mesh.get_name().c_str()};

	prepare_pipeline_state(command_buffer, front_face, sub_mesh.get_material()->double_sided);

	MultisampleState multisample_state{};
	multisample_state.rasterization_samples = get_sample_count();
	command_buffer.set_multisample_state(multisample_state);

	auto &vert_shader_module = device.get_resource_cache().request_shader_module(VK_SHADER_STAGE_VERTEX_BIT, get_vertex_shader(), sub_mesh.get_shader_variant());
	auto &frag_shader_module = device.get_resource_cache().request_shader_module(VK_SHADER_STAGE_FRAGMENT_BIT, get_fragment_shader(), sub_mesh.get_shader_variant());

	std::vector<ShaderModule *> shader_modules{&vert_shader_module, &frag_shader_module};

	auto &pipeline_layout = prepare_pipeline_layout(command_buffer, shader_modules);

	command_buffer.bind_pipeline_layout(pipeline_layout);

	if (pipeline_layout.get_push_constant_range_stage(sizeof(MeshInfo)) != 0)
	{
		prepare_push_constants(command_buffer, node);
	}

	DescriptorSetLayout &descriptor_set_layout = pipeline_layout.get_descriptor_set_layout(0);

	for (auto &texture : sub_mesh.get_material()->textures)
	{
		if (auto layout_binding = descriptor_set_layout.get_layout_binding(texture.first))
		{
			command_buffer.bind_image(texture.second->get_image()->get_vk_image_view(),
			                          texture.second->get_sampler()->vk_sampler,
			                          0, layout_binding->binding, 0);
		}
	}

	auto vertex_input_resources = pipeline_layout.get_resources(ShaderResourceType::Input, VK_SHADER_STAGE_VERTEX_BIT);

	VertexInputState vertex_input_state;

	for (auto &input_resource : vertex_input_resources)
	{
		sg::VertexAttribute attribute;

		if (!sub_mesh.get_attribute(input_resource.name, attribute))
		{
			continue;
		}

		VkVertexInputAttributeDescription vertex_attribute{};
		vertex_attribute.binding  = input_resource.location;
		vertex_attribute.format   = attribute.format;
		vertex_attribute.location = input_resource.location;
		vertex_attribute.offset   = attribute.offset;

		vertex_input_state.attributes.push_back(vertex_attribute);

		VkVertexInputBindingDescription vertex_binding{};
		vertex_binding.binding = input_resource.location;
		vertex_binding.stride  = attribute.stride;

		vertex_input_state.bindings.push_back(vertex_binding);
	}

	command_buffer.set_vertex_input_state(vertex_input_state);

	// Find submesh vertex buffers matching the shader input attribute names
	for (auto &input_resource : vertex_input_resources)
	{
		const auto &buffer_iter = sub_mesh.vertex_buffers.find(input_resource.name);

		if (buffer_iter != sub_mesh.vertex_buffers.end())
		{
			std::vector<std::reference_wrapper<const vkb::core::BufferC>> buffers;
			buffers.emplace_back(std::ref(buffer_iter->second));

			// Bind vertex buffers only for the attribute locations defined
			command_buffer.bind_vertex_buffers(input_resource.location, std::move(buffers), {0});
		}
	}
	render_count++;
	draw_submesh_command(command_buffer, sub_mesh);
}

void GLTFModelSubpass::prepare_pipeline_state(CommandBuffer &command_buffer, VkFrontFace front_face, bool double_sided_material)
{
	RasterizationState rasterization_state = base_rasterization_state;
	rasterization_state.front_face         = front_face;

	if (double_sided_material)
	{
		rasterization_state.cull_mode = VK_CULL_MODE_NONE;
	}

	command_buffer.set_rasterization_state(rasterization_state);

	MultisampleState multisample_state{};
	multisample_state.rasterization_samples = get_sample_count();
	command_buffer.set_multisample_state(multisample_state);
}

PipelineLayout &GLTFModelSubpass::prepare_pipeline_layout(CommandBuffer &command_buffer, const std::vector<ShaderModule *> &shader_modules)
{
	// Sets any specified resource modes
	for (auto &shader_module : shader_modules)
	{
		for (auto &resource_mode : get_resource_mode_map())
		{
			shader_module->set_resource_mode(resource_mode.first, resource_mode.second);
		}
	}

	return command_buffer.get_device().get_resource_cache().request_pipeline_layout(shader_modules);
}

void GLTFModelSubpass::prepare_push_constants(CommandBuffer &command_buffer, sg::Node &node)
{
	auto &transform = node.get_transform();
	MeshInfo model_info{};
	model_info.model = transform.get_world_matrix();

	auto data = to_bytes(model_info);

	if (!data.empty())
	{
		command_buffer.push_constants(data);
	}
}

void GLTFModelSubpass::draw_submesh_command(CommandBuffer &command_buffer, sg::SubMesh &sub_mesh)
{
	// Draw submesh indexed if indices exists
	if (sub_mesh.vertex_indices != 0)
	{
		// Bind index buffer of submesh
		command_buffer.bind_index_buffer(*sub_mesh.index_buffer, sub_mesh.index_offset, sub_mesh.index_type);

		// Draw submesh using indexed data
		command_buffer.draw_indexed(sub_mesh.vertex_indices, 1, 0, 0, 0);
	}
	else
	{
		// Draw submesh using vertices only
		command_buffer.draw(sub_mesh.vertices_count, 1, 0, 0);
	}
}

void GLTFModelSubpass::set_thread_index(uint32_t index)
{
	thread_index = index;
}
}        // namespace vkb
