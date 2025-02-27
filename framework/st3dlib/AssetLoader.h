#pragma once

#include<string>
#include<vector>

namespace strender
{
	class GameObject;
	class MetaScene;
	struct _render_handle;
	
	class AssetLoader
	{
	public:	
		static MetaScene* LoadPackage(const std::string& filepath);
	};

}
