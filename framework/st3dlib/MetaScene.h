#pragma once
#include <string>

namespace vkb
{
	namespace sg
	{
    class Node;
    class Scene;
    }
}
namespace strender {

	class Asset
	{
	  public:
		std::string generator;
		std::string exporterVersion;
		std::string tag;
		// double version;
	};

	class MetaScene
	{
	public:
		MetaScene();
		Asset asset;
	    vkb::sg::Scene *scene;
	protected:
		std::string file_path;
		std::string	name;
	};
}

