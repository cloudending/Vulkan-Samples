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

#include "skin.h"

#include "scene_graph/components/image.h"
#include "scene_graph/components/sampler.h"

namespace vkb
{
namespace sg
{
Skin::Skin(const std::string &name) :
    Component{name}
{}

std::type_index Skin::get_type()
{
	return typeid(Skin);
}

void Skin::set_skeleton_node(Node &node)
{
	skeleton_root = &node;
}

void Skin::add_inverse_bind_matrice(glm::mat4 &mat)
{
	inverse_bind_matrices.push_back(mat);
}



}        // namespace sg
}        // namespace vkb
