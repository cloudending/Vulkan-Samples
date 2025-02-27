#include "MetaSceneParser.h"
#include "st3dlib/MetaScene.h"
#include "common/vk_common.h"
#include "filesystem/legacy.h"
#include "scene_graph/components/pbr_material.h"
#include "gltf_loader.h"
#include "gui.h"

#include "rendering/subpasses/forward_subpass.h"
#include "rendering/subpasses/gltf_model_subpass.h"
#include "stats/stats.h"
using namespace strender;

static bool ParseNumberArrayProperty(std::vector<double>    *ret,
                                     std::string            *err,
                                     const picojson::object &o,
                                     const std::string      &property,
                                     bool                    required)
{
	picojson::object::const_iterator it = o.find(property);
	if (it == o.end())
	{
		if (required)
		{
			if (err)
			{
				(*err) += "'" + property + "' property is missing.\n";
			}
		}
		return false;
	}

	if (!it->second.is<picojson::array>())
	{
		if (required)
		{
			if (err)
			{
				(*err) += "'" + property + "' property is not an array.\n";
			}
		}
		return false;
	}

	ret->clear();
	const picojson::array &arr = it->second.get<picojson::array>();
	for (size_t i = 0; i < arr.size(); i++)
	{
		if (!arr[i].is<double>())
		{
			if (required)
			{
				if (err)
				{
					(*err) += "'" + property + "' property is not a number.\n";
				}
			}
			return false;
		}
		ret->push_back(arr[i].get<double>());
	}

	return true;
}

MetaSceneParser::MetaSceneParser(vkb::core::HPPDevice &d) :
    device(reinterpret_cast<vkb::Device &>(d))
{
	
}

MetaScene *MetaSceneParser::load_scene_from_file(const std::string &path)
{
	int index = (int)path.find_last_of("/");
	model_path = path.substr(0, index + 1);
	vkb::GLTFLoader loader(device);

	auto scene = loader.read_scene_from_file(path.substr(0, path.size() - 10) + ".gltf");
	if (!scene)
	{
		LOGE("Cannot load scene: {}", path.c_str());
		throw std::runtime_error("Cannot load scene: " + path);
	}



	auto             meta_scene_data = vkb::fs::read_asset(path);
	const char     *pstrbuffer      = reinterpret_cast<const char *>(meta_scene_data.data());
	picojson::value v;
	std::string     perr = picojson::parse(v, pstrbuffer, pstrbuffer + meta_scene_data.size());

	auto metascene = new MetaScene;
	
	metascene->scene = scene.release();

	auto parse_asset_func = [](strender::Asset *asset, std::string *err, const picojson::object &o) -> void {
		ParseStringProperty(&asset->generator, err, o, "generator", false);
		ParseStringProperty(&asset->exporterVersion, err, o, "exporterVersion", false);
		ParseStringProperty(&asset->tag, err, o, "tag", false);
	};

	if (v.contains("asset") && v.get("asset").is<picojson::object>())
	{
		const picojson::object &root = v.get("asset").get<picojson::object>();
		parse_asset_func(&metascene->asset, &perr, root);
	}


	std::unordered_map<int, vkb::sg::Node *> goMap;
	if (v.contains("gameobject") && v.get("gameobject").is<picojson::array>())
	{
		const picojson::array &objectarray = v.get("gameobject").get<picojson::array>();
		vkb::sg::Node         *root        = nullptr;
		for (size_t i = 0; i < objectarray.size(); i++)
		{
			parse_game_node(metascene->scene, (int) i, &perr, objectarray[i].get<picojson::object>());

			/*m_GameObjectArray.push_back(pgameobj);
			root = m_GameObjectArray.front();
			root->SetFullName(root->GetName());
			if (root != pgameobj)
			    pgameobj->SetRoot(root);

			goMap[pgameobj->GetInstanceId()] = pgameobj;*/
		}
		if (root != nullptr)
		{
			/*GameObject::AddNewGameObject(root);

			root->CopySkinnedMeshPathMap(m_pMetaScene->GetSkinnedMeshPath());*/
		}
	}

	return metascene;
}

void strender::MetaSceneParser::parse_game_node(vkb::sg::Scene *scene, int index, std::string *err, const picojson::object &o)
{
	
	std::string obj_name;
	ParseStringProperty(&obj_name, err, o, "name", false);
	double inst_id = 0;
	ParseNumberProperty(&inst_id, err, o, "instanceId", false);

	auto *node = scene->find_node(obj_name + "_$" + std::to_string((int)inst_id) + "$");

	if (!node)
		return;

	picojson::object::const_iterator skinnedmeshrendererIt = o.find("skinnedmeshrenderer");
	if ((skinnedmeshrendererIt != o.end()) && (skinnedmeshrendererIt->second).is<picojson::object>())
	{
		const picojson::object &renderer_object = (skinnedmeshrendererIt->second).get<picojson::object>();
		auto parse_skinned_mesh_renderer = [this, scene](vkb::sg::Node *node, int inst_id, const picojson::object &renderer_object) {
            picojson::object::const_iterator matdefIt = renderer_object.find("materialdefs");
            if ((matdefIt != renderer_object.end()) && (matdefIt->second).is<picojson::array>())
            {
                const picojson::array &matdef_array = matdefIt->second.get<picojson::array>();
                for (int i = 0; i < matdef_array.size(); ++i)
                {
                    std::string mat_name, err;
                    auto        matdef_obj = matdef_array[i].get<picojson::object>();

                    ParseStringProperty(&mat_name, &err, matdef_obj, "name", false);
                    auto        material = std::make_unique<vkb::sg::PBRMaterial>(mat_name);
                    std::string albedomap_path;
					std::string property_name = "albedomap";
					ParseStringProperty(&albedomap_path, &err, matdef_obj, property_name, false);
					auto tex = parse_texture(scene, property_name, albedomap_path);
                    material->textures.insert({property_name, tex.get()});

					vkb::sg::Mesh& mesh = node->get_component<vkb::sg::Mesh>();
                    for (auto &sub_mesh : mesh.get_submeshes())
                    {
                        sub_mesh->set_material(*material);
					}

					scene->add_component(std::move(material));
                    scene->add_component(std::move(tex));
                }
            }
		};
		parse_skinned_mesh_renderer(node, (int)inst_id, renderer_object);
	}
}

static void upload_image_to_gpu(vkb::CommandBuffer &command_buffer, vkb::core::BufferC &staging_buffer, vkb::sg::Image &image)
{
	// Clean up the image data, as they are copied in the staging buffer
	image.clear_data();

	{
		vkb::ImageMemoryBarrier memory_barrier{};
		memory_barrier.old_layout      = VK_IMAGE_LAYOUT_UNDEFINED;
		memory_barrier.new_layout      = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		memory_barrier.src_access_mask = 0;
		memory_barrier.dst_access_mask = VK_ACCESS_TRANSFER_WRITE_BIT;
		memory_barrier.src_stage_mask  = VK_PIPELINE_STAGE_HOST_BIT;
		memory_barrier.dst_stage_mask  = VK_PIPELINE_STAGE_TRANSFER_BIT;

		command_buffer.image_memory_barrier(image.get_vk_image_view(), memory_barrier);
	}

	// Create a buffer image copy for every mip level
	auto &mipmaps = image.get_mipmaps();

	std::vector<VkBufferImageCopy> buffer_copy_regions(mipmaps.size());

	for (size_t i = 0; i < mipmaps.size(); ++i)
	{
		auto &mipmap      = mipmaps[i];
		auto &copy_region = buffer_copy_regions[i];

		copy_region.bufferOffset     = mipmap.offset;
		copy_region.imageSubresource = image.get_vk_image_view().get_subresource_layers();
		// Update miplevel
		copy_region.imageSubresource.mipLevel = mipmap.level;
		copy_region.imageExtent               = mipmap.extent;
	}

	command_buffer.copy_buffer_to_image(staging_buffer, image.get_vk_image(), buffer_copy_regions);

	{
		vkb::ImageMemoryBarrier memory_barrier{};
		memory_barrier.old_layout      = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		memory_barrier.new_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		memory_barrier.src_access_mask = VK_ACCESS_TRANSFER_WRITE_BIT;
		memory_barrier.dst_access_mask = VK_ACCESS_SHADER_READ_BIT;
		memory_barrier.src_stage_mask  = VK_PIPELINE_STAGE_TRANSFER_BIT;
		memory_barrier.dst_stage_mask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

		command_buffer.image_memory_barrier(image.get_vk_image_view(), memory_barrier);
	}
}

std::unique_ptr<vkb::sg::Texture> strender::MetaSceneParser::parse_texture(vkb::sg::Scene *scene, std::string &name, std::string &path)
{
	auto image = vkb::sg::Image::load(name + "vkimage", model_path + "/" + path, vkb::sg::Image::Unknown);
	image->create_vk_image(device);
	std::vector<vkb::core::BufferC> transient_buffers;
	auto &command_buffer = device.request_command_buffer();
	command_buffer.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, 0);
	// Wait for this image to complete loading, then stage for upload
	vkb::core::Buffer stage_buffer = vkb::core::BufferC::create_staging_buffer(device, image->get_data());

	upload_image_to_gpu(command_buffer, stage_buffer, *image);

	transient_buffers.push_back(std::move(stage_buffer));
	command_buffer.end();

	auto &queue = device.get_queue_by_flags(VK_QUEUE_GRAPHICS_BIT, 0);

	queue.submit(command_buffer, device.request_fence());

	device.get_fence_pool().wait();
	device.get_fence_pool().reset();
	device.get_command_pool().reset_pool();
	device.wait_idle();

	transient_buffers.clear();
	


	VkFilter min_filter = VK_FILTER_NEAREST;
	VkFilter mag_filter = VK_FILTER_NEAREST;

	VkSamplerMipmapMode mipmap_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST;

	VkSamplerAddressMode address_mode_u = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	VkSamplerAddressMode address_mode_v = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	VkSamplerAddressMode address_mode_w = VK_SAMPLER_ADDRESS_MODE_REPEAT;

	VkSamplerCreateInfo sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};

	sampler_info.magFilter    = mag_filter;
	sampler_info.minFilter    = min_filter;
	sampler_info.mipmapMode   = mipmap_mode;
	sampler_info.addressModeU = address_mode_u;
	sampler_info.addressModeV = address_mode_v;
	sampler_info.addressModeW = address_mode_w;
	sampler_info.borderColor  = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
	sampler_info.maxLod       = std::numeric_limits<float>::max();

	vkb::core::Sampler vk_sampler(device, sampler_info);
	vk_sampler.set_debug_name(name + "_vksampler");

	auto sampler = std::make_unique<vkb::sg::Sampler>(name + "_vksampler", std::move(vk_sampler));
	auto texture = std::make_unique<vkb::sg::Texture>(name + "_texture");

	texture->set_image(*image);
	texture->set_sampler(*sampler);
	scene->add_component(std::move(image));
	scene->add_component(std::move(sampler));
	return std::move(texture);
	
}

	
