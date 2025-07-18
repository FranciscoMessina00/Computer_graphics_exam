#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNorm;
layout(location = 2) in vec2 fragUV;
layout(location = 3) in vec4 fragTan;
layout(location = 4) in vec2 fragUV_world;

layout(location = 0) out vec4 outColor;

layout(binding = 1, set = 1) uniform sampler2D albedoMap;
layout(binding = 2, set = 1) uniform sampler2D normalMap;
layout(binding = 3, set = 1) uniform sampler2D groundOcclusionMap;
layout(binding = 4, set = 1) uniform sampler2D groundRoughnessMap;
layout(binding = 5, set = 1) uniform sampler2D waterMap;
layout(binding = 6, set = 1) uniform sampler2D sandMap;
layout(binding = 7, set = 1) uniform sampler2D rockMap;

layout(binding = 0, set = 0) uniform GlobalUniformBufferObject {
    vec3 lightDir;
    vec4 lightColor;
    vec3 eyePos;
    vec3 airplanePos;
    vec4 otherParams; // ground height, waterLevel, grassLevel, rockLevel
} gubo;

const float PI = 3.14159265359;
const float stretchFactor = 0.7; // stretch factor for shadow
const float shadowSize = 3.0; // size of the shadow
const float maxShadowStrength = 0.5; // 0.0 = no shadow, 1.0 = full shadow
const float minShadowStrength = 0.01;

mat3 computeTBN(vec3 N, vec3 T, float tangentW) {
    vec3 B = cross(N, T) * tangentW;
    return mat3(normalize(T), normalize(B), normalize(N));
}

vec3 getNormalFromMap(mat3 TBN, vec2 worldUV) {
    vec3 tangentNormal = texture(normalMap, worldUV).xyz * 2.0 - 1.0;
    return normalize(TBN * tangentNormal);
}

vec3 Fresnel(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

float DistributionGGX(vec3 N, vec3 H, float roughness) {
    // squared for reduced smoothness
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0001f);
    float NdotH2 = NdotH * NdotH;

    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    return a2 / (PI * denom * denom);
}

float GeometryGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float Geometry(vec3 N, vec3 V, vec3 L, float roughness) {
    return GeometryGGX(max(dot(N, V), 0.0001f), roughness) *
    GeometryGGX(max(dot(N, L), 0.0001f), roughness);
}

// ------------------------------------------------------------
// a tiny 2D hash:  maps integer 2D coords -> [0,1)
float random_value(vec2 p) {
    // a classic “sin dot” trick
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

// rotate a 2D vector around (0,0) by angle θ:
vec2 rotate2D(vec2 v, float theta) {
    float c = cos(theta), s = sin(theta);
    return vec2(c*v.x - s*v.y,
    s*v.x + c*v.y);
}

void main() {
    // -------------- simple shadow
    // how high above the ground am I?
    float hAbove = gubo.airplanePos.y - gubo.otherParams.x; // ground height

    // project that point onto the ground plane, along the light‑direction:
    //   P + t·L  such that y == groundY
    float t = hAbove / gubo.lightDir.y;
    vec3 planeProjected = gubo.airplanePos - gubo.lightDir * t;
    vec2 shadowCenter   = planeProjected.xz;

    vec2 fragXZ = fragPos.xz;

    vec2 lightXZ = normalize(gubo.lightDir.xz);
    vec2 dir = fragXZ - shadowCenter;
    float d = dot(dir, lightXZ); // stretch along light
    float ortho = length(dir - lightXZ * d); // perpendicular
    float dist = sqrt((d * d) * stretchFactor + ortho * ortho);
    //float dist = length(fragXZ - shadowCenter);

    // Adjust shadow size based on airplane height
    float shadowRadius = clamp(shadowSize - hAbove * 0.05, 0.0, shadowSize);

    float shadowStrength = mix(maxShadowStrength, minShadowStrength, clamp(hAbove / 50.0, 0.0, 1.0));

    // soft circular shadow (dark center, smooth edges)
    float shadow = 1.0 - smoothstep(shadowRadius * 0.4, shadowRadius, dist);

    // --------------

    const float TILE_SIZE = 10.0f;
    vec2 worldUV = fragUV_world / TILE_SIZE;
    vec2 offset  = gubo.airplanePos.xz / TILE_SIZE;
    worldUV += offset;

    vec2 tileID = floor(worldUV);
    vec2 tiledUV = fract(worldUV);

    // --- generate per‐tile randomness ---
    float rnd = random_value(tileID);
    float angle = 0.0;
    if (rnd < 0.33) {
        angle = PI / 2.0; // 90 degrees
    } else if (rnd < 0.66) {
        angle = PI; // 180 degrees
    }
    // else angle is 0, for the rest

    // --- apply them around the tile‐center (0.5,0.5) ---
    vec2 uv = tiledUV - 0.5;
    uv = rotate2D(uv, angle);
    vec2 rotatedWorldUV = tileID + uv + 0.5;
    // --------------

    float waterLevel = gubo.otherParams.y;
    float grassLevel = gubo.otherParams.z;
    float rockLevel = gubo.otherParams.w;
    float shoreBlend = 0.3; // Blending factor

    // Get colors from textures
    vec3 waterColor = texture(waterMap, rotatedWorldUV).rgb;
    vec3 sandColor = texture(sandMap, rotatedWorldUV).rgb;
    vec3 grassColor = texture(albedoMap, rotatedWorldUV).rgb; // Use albedoMap for grass
    vec3 rockColor = texture(rockMap, rotatedWorldUV).rgb;

    // Blend between water and sand
    float waterSandMix = smoothstep(waterLevel - shoreBlend, waterLevel + shoreBlend, fragPos.y);
    vec3 albedo = mix(waterColor, sandColor, waterSandMix);

    // Blend between sand and grass
    float sandGrassMix = smoothstep(grassLevel - shoreBlend, grassLevel + shoreBlend, fragPos.y);
    albedo = mix(albedo, grassColor, sandGrassMix);

    // Blend between grass and rock
    float grassRockMix = smoothstep(rockLevel - shoreBlend, rockLevel + shoreBlend, fragPos.y);
    albedo = mix(albedo, rockColor, grassRockMix);

    float occlusion  = texture(groundOcclusionMap, worldUV).r;
    float roughness = texture(groundRoughnessMap, worldUV).r * 0.5f;

    vec3 N = normalize(fragNorm);
    vec3 T = normalize(fragTan.xyz);
    float w = fragTan.w;
    mat3 TBN = computeTBN(N, T, w);
    vec3 Nmap = getNormalFromMap(TBN, worldUV);

    vec3 V = normalize(gubo.eyePos - fragPos);
    vec3 L = normalize(gubo.lightDir);
    vec3 H = normalize(V + L);
    vec3 radiance = gubo.lightColor.rgb;


    vec3 F0 = mix(vec3(0.04), albedo, 0.1);

    float NDF = DistributionGGX(Nmap, H, roughness);
    float G   = Geometry(Nmap, V, L, roughness);
    vec3 F    = Fresnel(max(dot(H, V), 0.0), F0);

    vec3 specular = (NDF * G * F) /
    max(4.0 * max(dot(Nmap, V), 0.0001f) * max(dot(Nmap, L), 0.0), 0.0001f);

    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - 0.1;

    float NdotL = max(dot(Nmap, L), 0.0);
    vec3 Lo = (kD * albedo / PI + specular) * radiance * NdotL;


    // Ambient light contribution colors
    const float scaling = 0.01f;
    // blu cielo
    const vec3 ambient = vec3(0.2,0.7,1.0) * scaling;
    //vec3 ambient = vec3(0.015f) * albedo;

    vec3 color = (ambient + Lo * occlusion);
    color *= mix(vec3(1.0), vec3(1.0 - shadowStrength), shadow);

    outColor = vec4(color, 1.0);
    //outColor = vec4(albedo * vec3(clamp(dot(N, L),0.0,1.0)) + vec3(pow(clamp(dot(N, H),0.0,1.0), 160.0)) + ambient, 1.0);
    //outColor = vec4(albedo * vec3(clamp(dot(Nmap, L),0.0,1.0)) + vec3(pow(clamp(dot(Nmap, H),0.0,1.0), 160.0)) + ambient, 1.0);
	//outColor = vec4((Nmap+1.0f)*0.5f, 1.0);
	//outColor = vec4(texture(normalMap, fragUV).xyz, 1.0);
}
