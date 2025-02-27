#include "AssetLoader.h"
#include <algorithm>
#include <list>
#include "MetaScene.h"
#include "MetaSceneParser.h"

namespace strender
{

	MetaScene* AssetLoader::LoadPackage(const std::string& filepath)
	{
		std::string modelstr = filepath;
		std::string::size_type strpos = modelstr.find(".metascene");
		if (strpos == std::string::npos)
			return nullptr;


		return nullptr;
		
	}

	

}
