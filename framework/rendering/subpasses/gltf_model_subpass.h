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

#pragma once

#include "common/error.h"

#include "common/glm_common.h"

#include "rendering/subpass.h"

namespace vkb
{
namespace sg
{
class Scene;
class Node;
class Mesh;
class SubMesh;
class Camera;
}        // namespace sg

/**
 * @brief Global uniform structure for base shader
 */
struct alignas(16) UBOScene
{
	glm::mat4 projection;
	glm::mat4 view;
	glm::vec4 lightPos;
	glm::vec4 viewPos;
};

/**
 * @brief PBR material uniform for base shader
 */
struct MeshInfo
{
	glm::mat4 model;
};

/**
 * @brief This subpass is responsible for rendering a Scene
 */
class GLTFModelSubpass : public vkb::rendering::SubpassC
{
  public:
	/**
	 * @brief Constructs a subpass for the geometry pass of Deferred rendering
	 * @param render_context Render context
	 * @param vertex_shader Vertex shader source
	 * @param fragment_shader Fragment shader source
	 * @param scene Scene to render on this subpass
	 * @param camera Camera used to look at the scene
	 */
	GLTFModelSubpass(RenderContext &render_context, ShaderSource &&vertex_shader, ShaderSource &&fragment_shader, sg::Scene &scene, sg::Camera &camera);

	virtual ~GLTFModelSubpass() = default;

	virtual void prepare() override;

	/**
	 * @brief Record draw commands
	 */
	virtual void draw(CommandBuffer &command_buffer) override;

	/**
	 * @brief Thread index to use for allocating resources
	 */
	void set_thread_index(uint32_t index);

  protected:
	virtual void update_uniform(CommandBuffer &command_buffer, sg::Node &node, size_t thread_index);

	virtual void update_joint_matrix(CommandBuffer &command_buffer, sg::Node &nodem, size_t thread_index);

	void draw_submesh(CommandBuffer &command_buffer, sg::SubMesh &sub_mesh, sg::Node &node, VkFrontFace front_face = VK_FRONT_FACE_COUNTER_CLOCKWISE);

	virtual void prepare_pipeline_state(CommandBuffer &command_buffer, VkFrontFace front_face, bool double_sided_material);

	virtual PipelineLayout &prepare_pipeline_layout(CommandBuffer &command_buffer, const std::vector<ShaderModule *> &shader_modules);

	virtual void prepare_push_constants(CommandBuffer &command_buffer, sg::Node &node);

	virtual void draw_submesh_command(CommandBuffer &command_buffer, sg::SubMesh &sub_mesh);

	/**
	 * @brief Sorts objects based on distance from camera and classifies them
	 *        into opaque and transparent in the arrays provided
	 */
	void get_sorted_nodes(std::multimap<float, std::pair<sg::Node *, sg::SubMesh *>> &opaque_nodes,
	                      std::multimap<float, std::pair<sg::Node *, sg::SubMesh *>> &transparent_nodes);

	sg::Camera &camera;

	std::vector<sg::Mesh *> meshes;

	sg::Scene &scene;

	uint32_t thread_index{0};

	vkb::RasterizationState base_rasterization_state{};
};

}        // namespace vkb
