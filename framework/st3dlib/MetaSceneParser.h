#pragma once
#include <memory>
#include "MetaScene.h"

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

#include "common/hpp_utils.h"
#include "hpp_gltf_loader.h"

#include "picojson.h"


namespace strender {
    class MetaSceneParser
    {        
    public:
	    MetaSceneParser(vkb::core::HPPDevice &device);
        MetaScene* load_scene_from_file(const std::string& filename);
	private:
	    void parse_game_node(vkb::sg::Scene *scene, int index, std::string *err, const picojson::object &o);
	  std::unique_ptr<vkb::sg::Texture> parse_texture(vkb::sg::Scene *scene, std::string &name, std::string &path);
	    vkb::Device &device;
	    std::string                                 model_path;

    };
    
}  // namespace strender


