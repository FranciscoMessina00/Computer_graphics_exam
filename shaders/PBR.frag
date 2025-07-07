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
layout(binding = 3, set = 1) uniform sampler2D occlusionMap;
layout(binding = 4, set = 1) uniform sampler2D roughnessMap;

layout(binding = 0, set = 0) uniform GlobalUniformBufferObject {
    vec3 lightDir;
    vec4 lightColor;
    vec3 eyePos;
    vec3 airplanePos;
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

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0001f);
    float NdotH2 = NdotH * NdotH;

    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    return a2 / (PI * denom * denom);
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    return GeometrySchlickGGX(max(dot(N, V), 0.0001f), roughness) *
    GeometrySchlickGGX(max(dot(N, L), 0.0001f), roughness);
}

void main() {
    // -------------- simple shadow
    vec3 planeProjected = gubo.airplanePos - gubo.lightDir * gubo.airplanePos.y / gubo.lightDir.y;
    vec2 shadowCenter = planeProjected.xz;

    vec2 fragXZ = fragPos.xz;

    vec2 lightXZ = normalize(gubo.lightDir.xz);
    vec2 dir = fragXZ - shadowCenter;
    float d = dot(dir, lightXZ); // stretch along light
    float ortho = length(dir - lightXZ * d); // perpendicular
    float dist = sqrt((d * d) * stretchFactor + ortho * ortho);
    //float dist = length(fragXZ - shadowCenter);

    // Adjust shadow size based on airplane height
    float shadowRadius = clamp(shadowSize - gubo.airplanePos.y * 0.05, 0.0, shadowSize);

    float height = gubo.airplanePos.y;
    float shadowStrength = mix(maxShadowStrength, minShadowStrength, clamp(height / 50.0, 0.0, 1.0));

    // soft circular shadow (dark center, smooth edges)
    float shadow = 1.0 - smoothstep(shadowRadius * 0.4, shadowRadius, dist);

    // --------------

    const float TILE_SIZE = 10.0;
    vec2 worldUV = fragUV_world / TILE_SIZE;
    vec2 offset  = gubo.airplanePos.xz / TILE_SIZE;
    worldUV += offset;

    vec3 albedo     = texture(albedoMap, worldUV).rgb;
    float occlusion  = texture(occlusionMap, worldUV).r;
    float roughness = texture(roughnessMap, worldUV).r;


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
    float G   = GeometrySmith(Nmap, V, L, roughness);
    vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 specular = (NDF * G * F) /
    max(4.0 * max(dot(Nmap, V), 0.0001f) * max(dot(Nmap, L), 0.0), 0.0001f);

    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - 0.1;

    float NdotL = max(dot(Nmap, L), 0.0);
    vec3 Lo = (kD * albedo / PI + specular) * radiance * NdotL;


    // Ambient light contribution colors
    const float scaling = 0.03f;
    // blu
    const vec3 cxp = vec3(0.2,0.5,0.9) * scaling;
    // blu arancione
    const vec3 cxn = vec3(0.9,0.6,0.7) * scaling;
    // blu cielo
    const vec3 cyp = vec3(0.2,0.7,1.0) * scaling;
    // verde prato
    const vec3 cyn = vec3(0.34,0.76,0.4) * scaling;
    // arancione
    const vec3 czp = vec3(0.9,0.6,0.0) * scaling;
    // blu scuro
    const vec3 czn = vec3(0.24,0.0,0.91) * scaling;

    vec3 ambient =((N.x > 0 ? cxp : cxn) * (N.x * N.x) +
    (N.y > 0 ? cyp : cyn) * (N.y * N.y) +
    (N.z > 0 ? czp : czn) * (N.z * N.z)) * albedo;
    //vec3 ambient = vec3(0.015f) * albedo;

    vec3 color = (ambient + Lo * occlusion);
    color *= mix(vec3(1.0), vec3(1.0 - shadowStrength), shadow);

    outColor = vec4(color, 1.0);
    //outColor = vec4(albedo * vec3(clamp(dot(N, L),0.0,1.0)) + vec3(pow(clamp(dot(N, H),0.0,1.0), 160.0)) + ambient, 1.0);
    //outColor = vec4(albedo * vec3(clamp(dot(Nmap, L),0.0,1.0)) + vec3(pow(clamp(dot(Nmap, H),0.0,1.0), 160.0)) + ambient, 1.0);
	//outColor = vec4((Nmap+1.0f)*0.5f, 1.0);
	//outColor = vec4(texture(normalMap, fragUV).xyz, 1.0);
}
