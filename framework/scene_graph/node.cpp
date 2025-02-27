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

#include "node.h"

#include "component.h"
#include "components/transform.h"
#include "components/skin.h"

namespace vkb
{
namespace sg
{
Node::Node(const size_t id, const std::string &name) :
    id{id},
    name{name},
    transform{*this}
{
	set_component(transform);
}

const size_t Node::get_id() const
{
	return id;
}

const std::string &Node::get_name() const
{
	return name;
}

void Node::set_parent(Node &p)
{
	parent = &p;

	transform.invalidate_world_matrix();
}

Node *Node::get_parent() const
{
	return parent;
}

void Node::add_child(Node &child)
{
	children.push_back(&child);
}

const std::vector<Node *> &Node::get_children() const
{
	return children;
}

void Node::set_component(Component &component)
{
	auto it = components.find(component.get_type());

	if (it != components.end())
	{
		it->second = &component;
	}
	else
	{
		components.insert(std::make_pair(component.get_type(), &component));
	}
}

Component &Node::get_component(const std::type_index index)
{
	return *components.at(index);
}

bool Node::has_component(const std::type_index index)
{
	return components.count(index) > 0;
}

void Node::set_skin_index(int index)
{
	skin_index = index;
}

int Node::get_skin_index()
{
	return skin_index;
}

void Node::set_skin(Skin &s)
{
	skin = &s;
}

Skin *Node::get_skin()
{
	return skin;
}

void Node::update_joint_matrix()
{
	if (skin)
	{
		glm::mat4 inverse_transform = glm::inverse(transform.get_world_matrix());
		
		size_t num_joints        = (uint32_t) skin->joints.size();
		std::vector<glm::mat4> joint_matrices(num_joints);
		for (size_t i = 0; i < num_joints; i++)
		{
			joint_matrices[i] = skin->joints[i]->get_transform().get_world_matrix() * skin->inverse_bind_matrices[i];
			joint_matrices[i] = inverse_transform * joint_matrices[i];
		}

		skin->inverse_bind_matrices_ssbo->update(joint_matrices.data(), sizeof(glm::mat4) * num_joints, 0);

		for (auto child : children)
		{
			child->update_joint_matrix();
		}
	}
}

}        // namespace sg
}        // namespace vkb
