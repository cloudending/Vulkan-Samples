/* Copyright (c) 2018-2019, Arm Limited and Contributors
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

#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

#include "common/glm_common.h"
#include <glm/gtc/type_ptr.hpp>
#include "core/buffer.h"
#include "scene_graph/component.h"

namespace vkb
{
namespace sg
{


class Skin : public Component
{
  public:

	Skin() = delete;

	Skin(const std::string &name);

	Skin(Skin &&other) = default;

	virtual ~Skin() = default;

	virtual std::type_index get_type() override;

	void set_skeleton_node(Node &node);

	void add_inverse_bind_matrice(glm::mat4 &mat);

	std::unique_ptr<vkb::core::BufferC> inverse_bind_matrices_ssbo;

	std::vector<glm::mat4> inverse_bind_matrices;

	std::vector<Node *>    joints;
  private:
	Node *skeleton_root;


};
}        // namespace sg
}        // namespace vkb
