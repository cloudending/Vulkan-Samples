

#version 450
const vec3 diffuse = vec3(0.5,0.5,0.5);
const vec3 specular = vec3(0.7, 0.7, 0.7);
const vec3 emissive = vec3(0, 0, 0);
const float roughness = 0.9;
const float metalness = 0.4;
const float smoothness = 0.2;
const float opacity = 1.0;


layout(set = 0, binding = 0) uniform sampler2D albedomap;

layout(location = 0) in vec3 worldPos;
layout(location = 1) in vec2 vUv;
layout(location = 2) in vec3 wNormal;
layout(location = 3) in vec3 eyeVec;
layout(location = 4) in vec3 lightPosition;

layout (location = 0) out vec4 outFragColor;


#define PI 3.14159265359
#define PI2 6.28318530718
#define PI_HALF 1.5707963267949
#define RECIPROCAL_PI 0.31830988618
#define RECIPROCAL_PI2 0.15915494
#define LOG2 1.442695
#define EPSILON 1e-6

#define saturate(a) clamp( a, 0.0, 1.0 )
#define whiteCompliment(a) ( 1.0 - saturate( a ) )

float pow2( const in float x ) { return x*x; }
float pow3( const in float x ) { return x*x*x; }
float pow4( const in float x ) { float x2 = x*x; return x2*x2; }
float Pow5( const in float x ) { float x2 = x*x; return x2*x2*x; }
float average( const in vec3 color ) { return dot( color, vec3( 0.3333 ) ); }
// expects values in the range of [0,1]x[0,1], returns values in the [0,1] range.
// do not collapse into a single function per: http://byteblacksmith.com/improvements-to-the-canonical-one-liner-glsl-rand-for-opengl-es-2-0/
//highp float rand( const in vec2 uv ) {
//	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
//	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
//	return fract(sin(sn) * c);
//}

struct IncidentLight {
	vec3 color;
	vec3 direction;
	float intensity;
	bool visible;
};

struct ReflectedLight {
	vec3 directDiffuse;
	vec3 directSpecular;
	vec3 indirectDiffuse;
	vec3 indirectSpecular;
};

struct GeometricContext {
	vec3 position;
	vec3 normal;
	vec3 viewDir;
};

struct PhysicalMaterial {

	vec3	diffuseColor;
	vec3	albedoColor;//origin albedo map color*diffuse color
	float	specularRoughness;
	vec3	specularColor;
	float	smoothness;
	float	metalness;
	float	oneMinusReflectivity;
};

#define MAXIMUM_SPECULAR_COEFFICIENT 0.16
#define DEFAULT_SPECULAR_COEFFICIENT 0.220916301

vec3 transformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );
}

// http://en.wikibooks.org/wiki/GLSL_Programming/Applying_Matrix_Transformations
vec3 inverseTransformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( vec4( dir, 0.0 ) * matrix ).xyz );
}

vec3 projectOnPlane(in vec3 point, in vec3 pointOnPlane, in vec3 planeNormal ) {
	float distance = dot( planeNormal, point - pointOnPlane );
	return - distance * planeNormal + point;
}

float sideOfPlane( in vec3 point, in vec3 pointOnPlane, in vec3 planeNormal ) {
	return sign( dot( point - pointOnPlane, planeNormal ) );
}

vec3 linePlaneIntersect( in vec3 pointOnLine, in vec3 lineDirection, in vec3 pointOnPlane, in vec3 planeNormal ) {
	return lineDirection * ( dot( planeNormal, pointOnPlane - pointOnLine ) / dot( planeNormal, lineDirection ) ) + pointOnLine;
}

mat3 transposeMat3( const in mat3 m ) {
	mat3 tmp;
	tmp[ 0 ] = vec3( m[ 0 ].x, m[ 1 ].x, m[ 2 ].x );
	tmp[ 1 ] = vec3( m[ 0 ].y, m[ 1 ].y, m[ 2 ].y );
	tmp[ 2 ] = vec3( m[ 0 ].z, m[ 1 ].z, m[ 2 ].z );
	return tmp;
}

// https://en.wikipedia.org/wiki/Relative_luminance
float linearToRelativeLuminance( const in vec3 color ) {
	vec3 weights = vec3( 0.2126, 0.7152, 0.0722 );
	return dot( weights, color.rgb );
}

vec3 BRDF_Diffuse_Lambert( const in vec3 diffuseColor ) {

	return RECIPROCAL_PI * diffuseColor;

} // validated

float BRDF_DisneyDiffuse( const in float NdotV,const in float NdotL,const in float LdotH, const in float perceptualRoughness ) {

	float fd90 = 0.5 + 2.0 * LdotH * LdotH * perceptualRoughness;

	 float lightScatter   = (1.0 + (fd90 - 1.0) * Pow5(1.0 - NdotL));

	 float viewScatter    = (1.0 + (fd90 - 1.0) * Pow5(1.0 - NdotV));

	 return lightScatter * viewScatter;

} // validated

vec3 F_Schlick( const in vec3 specularColor, const in float dotLH ) {

	// Original approximation by Christophe Schlick '94
	// float fresnel = pow( 1.0 - dotLH, 5.0 );

	// Optimized variant (presented by Epic at SIGGRAPH '13)
	// https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
	float fresnel = exp2( ( -5.55473 * dotLH - 6.98316 ) * dotLH );

	return ( 1.0 - specularColor ) * fresnel + specularColor;

} // validated

// Microfacet Models for Refraction through Rough Surfaces - equation (34)
// http://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html
// alpha is roughness squared in Disney’s reparameterization
float G_GGX_Smith( const in float alpha, const in float dotNL, const in float dotNV ) {

	// geometry term (normalized) = G(l)⋅G(v) / 4(n⋅l)(n⋅v)
	// also see #12151

	float a2 = pow2( alpha );

	float gl = dotNL + sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNL ) );
	float gv = dotNV + sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNV ) );

	return 1.0 / ( gl * gv );

} // validated

// Moving Frostbite to Physically Based Rendering 3.0 - page 12, listing 2
// https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
float G_GGX_SmithCorrelated( const in float alpha, const in float dotNL, const in float dotNV ) {

	float a2 = pow2( alpha );

	// dotNL and dotNV are explicitly swapped. This is not a mistake.
	float gv = dotNL * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNV ) );
	float gl = dotNV * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNL ) );

	return 0.5 / max( gv + gl, EPSILON );

}

// Microfacet Models for Refraction through Rough Surfaces - equation (33)
// http://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html
// alpha is roughness squared in Disney’s reparameterization
float D_GGX( const in float alpha, const in float dotNH ) {

	float a2 = pow2( alpha );

	float denom = pow2( dotNH ) * ( a2 - 1.0 ) + 1.0; // avoid alpha = 0 with dotNH = 1

	return RECIPROCAL_PI * a2 / pow2( denom );

}

// GGX Distribution, Schlick Fresnel, GGX-Smith Visibility
vec3 BRDF_Specular_GGX( const in IncidentLight incidentLight, const in GeometricContext geometry, const in vec3 specularColor, const in float roughness ) {

	float alpha = pow2( roughness ); // UE4's roughness

	vec3 halfDir = normalize( incidentLight.direction + geometry.viewDir );

	float dotNL = saturate( dot( geometry.normal, incidentLight.direction ) );
	float dotNV = saturate( dot( geometry.normal, geometry.viewDir ) );
	float dotNH = saturate( dot( geometry.normal, halfDir ) );
	float dotLH = saturate( dot( incidentLight.direction, halfDir ) );

	vec3 F = F_Schlick( specularColor, dotLH );

	float G = G_GGX_SmithCorrelated( alpha, dotNL, dotNV );

	float D = D_GGX( alpha, dotNH );

	return F * ( G * D );

} // validated

// Ref: http://jcgt.org/published/0003/02/03/paper.pdf
float SmithJointGGXVisibilityTerm_ST( const in float NdotL , const in float NdotV, const in float roughness ) {

	float a = roughness;

	float lambdaV = NdotL * (NdotV * (1.0 - a) + a);

	float lambdaL = NdotV * (NdotL * (1.0 - a) + a);

	return 0.5 / (lambdaV + lambdaL + 1e-5);

} // validated

vec3 FresnelTerm_ST( const in vec3 F0 , const in float cosA ) {

	float t = Pow5 (1.0 - cosA);   // ala Schlick interpoliation

	return F0 + (1.0 - F0) * t;

} // validated


vec3 FresnelLerp_ST(const in vec3 F0, const in vec3 F90,const in float cosA) {

	float t = Pow5 (1.0 - cosA);   // ala Schlick interpoliation

	return mix(F0,F90,t);

}

vec3 fresnelSchlickRoughness( const in vec3 F0 , const in float cosA,float roughness ) {

	float t = Pow5 (1.0 - cosA);   // ala Schlick interpoliation

	return F0 + (max(vec3(1.0-roughness),F0)-F0) * t;

} // validated

// Ref: http://jcgt.org/published/0003/02/03/paper.pdf
float GGXTerm_ST( const in float NdotH , const in float roughness ) {

	float a2 = roughness * roughness;

	 float d = (NdotH * a2 - NdotH) * NdotH + 1.0; // 2 mad

	return 0.31830988618 * a2 / (d * d + 1e-7);

} // validated

// Rect Area Light

// Real-Time Polygonal-Light Shading with Linearly Transformed Cosines
// by Eric Heitz, Jonathan Dupuy, Stephen Hill and David Neubelt
// code: https://github.com/selfshadow/ltc_code/

vec2 LTC_Uv( const in vec3 N, const in vec3 V, const in float roughness ) {

	const float LUT_SIZE  = 64.0;
	const float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;
	const float LUT_BIAS  = 0.5 / LUT_SIZE;

	float dotNV = saturate( dot( N, V ) );

	// texture parameterized by sqrt( GGX alpha ) and sqrt( 1 - cos( theta ) )
	vec2 uv = vec2( roughness, sqrt( 1.0 - dotNV ) );

	uv = uv * LUT_SCALE + LUT_BIAS;

	return uv;

}

float LTC_ClippedSphereFormFactor( const in vec3 f ) {

	// Real-Time Area Lighting: a Journey from Research to Production (p.102)
	// An approximation of the form factor of a horizon-clipped rectangle.

	float l = length( f );

	return max( ( l * l + f.z ) / ( l + 1.0 ), 0.0 );

}

vec3 LTC_EdgeVectorFormFactor( const in vec3 v1, const in vec3 v2 ) {

	float x = dot( v1, v2 );

	float y = abs( x );

	// rational polynomial approximation to theta / sin( theta ) / 2PI
	float a = 0.8543985 + ( 0.4965155 + 0.0145206 * y ) * y;
	float b = 3.4175940 + ( 4.1616724 + y ) * y;
	float v = a / b;

	float theta_sintheta = ( x > 0.0 ) ? v : 0.5 * inversesqrt( max( 1.0 - x * x, 1e-7 ) ) - v;

	return cross( v1, v2 ) * theta_sintheta;

}

vec3 LTC_Evaluate( const in vec3 N, const in vec3 V, const in vec3 P, const in mat3 mInv, const in vec3 rectCoords[ 4 ] ) {

	// bail if point is on back side of plane of light
	// assumes ccw winding order of light vertices
	vec3 v1 = rectCoords[ 1 ] - rectCoords[ 0 ];
	vec3 v2 = rectCoords[ 3 ] - rectCoords[ 0 ];
	vec3 lightNormal = cross( v1, v2 );

	if( dot( lightNormal, P - rectCoords[ 0 ] ) < 0.0 ) return vec3( 0.0 );

	// construct orthonormal basis around N
	vec3 T1, T2;
	T1 = normalize( V - N * dot( V, N ) );
	T2 = - cross( N, T1 ); // negated from paper; possibly due to a different handedness of world coordinate system

	// compute transform
	mat3 mat = mInv * transposeMat3( mat3( T1, T2, N ) );

	// transform rect
	vec3 coords[ 4 ];
	coords[ 0 ] = mat * ( rectCoords[ 0 ] - P );
	coords[ 1 ] = mat * ( rectCoords[ 1 ] - P );
	coords[ 2 ] = mat * ( rectCoords[ 2 ] - P );
	coords[ 3 ] = mat * ( rectCoords[ 3 ] - P );

	// project rect onto sphere
	coords[ 0 ] = normalize( coords[ 0 ] );
	coords[ 1 ] = normalize( coords[ 1 ] );
	coords[ 2 ] = normalize( coords[ 2 ] );
	coords[ 3 ] = normalize( coords[ 3 ] );

	// calculate vector form factor
	vec3 vectorFormFactor = vec3( 0.0 );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 0 ], coords[ 1 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 1 ], coords[ 2 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 2 ], coords[ 3 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 3 ], coords[ 0 ] );

	// adjust for horizon clipping
	float result = LTC_ClippedSphereFormFactor( vectorFormFactor );

/*
	// alternate method of adjusting for horizon clipping (see referece)
	// refactoring required
	float len = length( vectorFormFactor );
	float z = vectorFormFactor.z / len;

	const float LUT_SIZE  = 64.0;
	const float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;
	const float LUT_BIAS  = 0.5 / LUT_SIZE;

	// tabulated horizon-clipped sphere, apparently...
	vec2 uv = vec2( z * 0.5 + 0.5, len );
	uv = uv * LUT_SCALE + LUT_BIAS;

	float scale = texture2D( ltc_2, uv ).w;

	float result = len * scale;
*/

	return vec3( result );

}

// End Rect Area Light

// ref: https://www.unrealengine.com/blog/physically-based-shading-on-mobile - environmentBRDF for GGX on mobile
vec3 BRDF_Specular_GGX_Environment( const in GeometricContext geometry, const in vec3 specularColor, const in float roughness ) {

	float dotNV = saturate( dot( geometry.normal, geometry.viewDir ) );

	const vec4 c0 = vec4( - 1, - 0.0275, - 0.572, 0.022 );

	const vec4 c1 = vec4( 1, 0.0425, 1.04, - 0.04 );

	vec4 r = roughness * c0 + c1;

	float a004 = min( r.x * r.x, exp2( - 9.28 * dotNV ) ) * r.x + r.y;

	vec2 AB = vec2( -1.04, 1.04 ) * a004 + r.zw;

	return specularColor * AB.x + AB.y;

} // validated


float G_BlinnPhong_Implicit( /* const in float dotNL, const in float dotNV */ ) {

	// geometry term is (n dot l)(n dot v) / 4(n dot l)(n dot v)
	return 0.25;

}

float D_BlinnPhong( const in float shininess, const in float dotNH ) {

	return RECIPROCAL_PI * ( shininess * 0.5 + 1.0 ) * pow( dotNH, shininess );

}

vec3 BRDF_Specular_BlinnPhong( const in IncidentLight incidentLight, const in GeometricContext geometry, const in vec3 specularColor, const in float shininess ) {

	vec3 halfDir = normalize( incidentLight.direction + geometry.viewDir );

	//float dotNL = saturate( dot( geometry.normal, incidentLight.direction ) );
	//float dotNV = saturate( dot( geometry.normal, geometry.viewDir ) );
	float dotNH = saturate( dot( geometry.normal, halfDir ) );
	float dotLH = saturate( dot( incidentLight.direction, halfDir ) );

	vec3 F = F_Schlick( specularColor, dotLH );

	float G = G_BlinnPhong_Implicit( /* dotNL, dotNV */ );

	float D = D_BlinnPhong( shininess, dotNH );

	return F * ( G * D );

} // validated

// source: http://simonstechblog.blogspot.ca/2011/12/microfacet-brdf.html
float GGXRoughnessToBlinnExponent( const in float ggxRoughness ) {
	return ( 2.0 / pow2( ggxRoughness + 0.0001 ) - 2.0 );
}

float BlinnExponentToGGXRoughness( const in float blinnExponent ) {
	return sqrt( 2.0 / ( blinnExponent + 2.0 ) );
}

void RE_Direct_Physical( const in IncidentLight directLight, const in GeometricContext geometry, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {

	float perceptualRoughness=(1.0 - material.smoothness);

	float dotNL = saturate( dot( geometry.normal, directLight.direction ) );

	vec3 halfDir = normalize( directLight.direction + geometry.viewDir );

	float nh = saturate( dot( geometry.normal, halfDir ) );
	float nv = abs( dot( geometry.normal, geometry.viewDir ) );

	float lv = saturate( dot( directLight.direction, geometry.viewDir ) );

	float lh = saturate( dot( directLight.direction, halfDir ) );

	reflectedLight.directDiffuse += material.diffuseColor * directLight.color * directLight.intensity * BRDF_DisneyDiffuse( nv, dotNL, lh, perceptualRoughness) * dotNL;

	float mRoughness = perceptualRoughness * perceptualRoughness;

	mRoughness = max(mRoughness, 0.002);

	float V = SmithJointGGXVisibilityTerm_ST (dotNL, nv, mRoughness);

	float D = GGXTerm_ST (nh, mRoughness);

	float specularTerm=V*D*PI;

	specularTerm = sqrt(max(1e-4, specularTerm));

	specularTerm = max(0.0, specularTerm * dotNL);

//	specularTerm *= any(material.specularColor) ? 1.0 : 0.0;

	reflectedLight.directSpecular += specularTerm * directLight.color * directLight.intensity * FresnelTerm_ST (material.specularColor, lh);

}


struct DirectionalLight {
    vec3 direction;
    vec3 color;
    float intensity;

    int shadow;
    float shadowBias;
    float shadowRadius;
    vec2 shadowMapSize;
};



void getDirectionalDirectLightIrradiance( const in DirectionalLight directionalLight, const in GeometricContext geometry, out IncidentLight directLight ) {

    directLight.color = directionalLight.color;
    directLight.direction =normalize( directionalLight.direction );
    directLight.visible = true;
    directLight.intensity = directionalLight.intensity;

}



void main() {

	vec4 diffuseColor = vec4( diffuse, opacity );
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;

	vec4 texelColor = texture( albedomap, vUv );
	diffuseColor *= texelColor;
	float alpha = texelColor.a;

    float specularStrength;
	vec3 specColor = specular;//this should comes from spec map
	float metalnessFactor = metalness;
	float smoothnessFactor = smoothness;



    PhysicalMaterial material;
	float oneMinusReflectivity;
	material.specularColor = mix( vec3( DEFAULT_SPECULAR_COEFFICIENT ), diffuseColor.rgb, metalnessFactor );
	oneMinusReflectivity = (1.0 - DEFAULT_SPECULAR_COEFFICIENT) - metalnessFactor * (1.0 - DEFAULT_SPECULAR_COEFFICIENT);
	material.diffuseColor = diffuseColor.rgb * oneMinusReflectivity;
	material.albedoColor = diffuseColor.rgb ;

	material.smoothness = smoothnessFactor;
	material.metalness = metalnessFactor;
	material.specularRoughness = 1.0 -  smoothnessFactor * smoothnessFactor;
	material.oneMinusReflectivity = oneMinusReflectivity;


    GeometricContext geometry;

    geometry.normal = normalize(wNormal);

    geometry.position = worldPos;
    geometry.viewDir = - normalize(eyeVec);

    IncidentLight directLight;

    DirectionalLight directionalLight;
	directionalLight.direction = vec3(0.5,0.4,0.3);
	directionalLight.color = vec3(1.0,1.0,1.0);
	directionalLight.intensity = 1.0;
	  
    getDirectionalDirectLightIrradiance( directionalLight, geometry, directLight );
    RE_Direct_Physical( directLight, geometry, material, reflectedLight );

	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.directSpecular + reflectedLight.indirectDiffuse + reflectedLight.indirectSpecular + totalEmissiveRadiance.rgb;
	outFragColor = vec4(outgoingLight, alpha );

}
