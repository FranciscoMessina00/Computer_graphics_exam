// This has been adapted from the Vulkan tutorial
#include <sstream>

#include <json.hpp>

#include "modules/Starter.hpp"
#include "modules/TextMaker.hpp"
#include "modules/Scene.hpp"
#include "modules/Animations.hpp"
#include <random>

#include <AL/al.h>
#include <AL/alc.h>
#define DR_WAV_IMPLEMENTATION
#include <dr_wav.h>
#include <thread>
#include <chrono>
#include <FastNoise.h>

#include <ode/ode.h>

#include <fstream>
#include <vector>
#include <cstdint>

struct VertexSimp
{
    glm::vec3 pos;
    glm::vec3 norm;
    glm::vec2 UV;
};

struct skyBoxVertex
{
    glm::vec3 pos;
};

struct VertexTan
{
    glm::vec3 pos;
    glm::vec3 norm;
    glm::vec2 UV;
    glm::vec4 tan;
};

struct GlobalUniformBufferObject
{
    alignas(16) glm::vec3 lightDir;
    alignas(16) glm::vec4 lightColor;
    alignas(16) glm::vec3 eyePos;
};

struct GlobalUniformBufferGround
{
    alignas(16) glm::vec3 lightDir;
    alignas(16) glm::vec4 lightColor;
    alignas(16) glm::vec3 eyePos;
    alignas(16) glm::vec3 eyePosNoSmooth;
    alignas(16) float groundHeight;
};

struct UniformBufferObjectSimp
{
    alignas(16) glm::mat4 mvpMat;
    alignas(16) glm::mat4 mMat;
    alignas(16) glm::mat4 nMat;
};

struct UniformBufferObjectGround
{
    alignas(16) glm::mat4 mvpMat;
    alignas(16) glm::mat4 mMat;
    alignas(16) glm::mat4 nMat;
    alignas(16) glm::mat4 worldMat;
};

struct skyBoxUniformBufferObject
{
    alignas(16) glm::mat4 mvpMat;
};

struct TerrainHeightmap {
    float minX, minZ;       // World-space origin of terrain grid
    float cellW, cellD;     // Width and depth of each cell
    int W, D;               // Grid resolution: number of samples along width and depth
    std::vector<float> heights; // Flattened heightmap (row-major order)

    // Initializes the heightmap
    void init(int widthSamples, int depthSamples,
              float cellWidth, float cellDepth,
              float worldOriginX, float worldOriginZ,
              std::function<float(float, float)> heightFunc)
    {
        W = widthSamples;
        D = depthSamples;
        cellW = cellWidth;
        cellD = cellDepth;
        minX = worldOriginX;
        minZ = worldOriginZ;

        heights.resize(W * D);
        for (int z = 0; z < D; ++z) {
            for (int x = 0; x < W; ++x) {
                float worldX = minX + x * cellW;
                float worldZ = minZ + z * cellD;
                heights[z * W + x] = heightFunc(worldX, worldZ);
            }
        }
    }

    // Sample height at world-space coordinates (with bilinear interpolation)
    float sampleHeight(float x, float z) const {
        float fx = (x - minX) / cellW;
        float fz = (z - minZ) / cellD;

        fx = std::clamp(fx, 0.0f, float(W - 1));
        fz = std::clamp(fz, 0.0f, float(D - 1));

        int ix = int(floor(fx));
        int iz = int(floor(fz));
        int ix1 = std::min(ix + 1, W - 1);
        int iz1 = std::min(iz + 1, D - 1);

        float tx = fx - ix;
        float tz = fz - iz;

        float h00 = heights[ iz  * W + ix  ];
        float h10 = heights[ iz  * W + ix1 ];
        float h01 = heights[ iz1 * W + ix  ];
        float h11 = heights[ iz1 * W + ix1 ];

        float h0 = glm::mix(h00, h10, tx);
        float h1 = glm::mix(h01, h11, tx);
        return glm::mix(h0, h1, tz) * cellW;
    }
};

// MAIN !
class E09 : public BaseProject
{
protected:
    enum CameraMode { FIRST_PERSON, THIRD_PERSON };

    // Nuova macchina a stati per il comportamento dell'aereo

    // Costanti per il nuovo modello di volo
    CameraMode currentCameraMode = THIRD_PERSON;
    // Here you list all the Vulkan objects you need:

    // Descriptor Layouts [what will be passed to the shaders]
    DescriptorSetLayout DSLlocalSimp, DSLlocalGem, DSLlocalPBR, DSLglobal, DSLglobalGround, DSLskyBox;

    // Vertex formants, Pipelines [Shader couples] and Render passes
    VertexDescriptor VDsimp;
    VertexDescriptor VDgem;
    VertexDescriptor VDskyBox;
    VertexDescriptor VDtan;
    RenderPass RP;
    Pipeline PsimpObj, PskyBox, P_PBR, Pgem;
    //*DBG*/Pipeline PDebug;

    // Models, textures and Descriptors (values assigned to the uniforms)
    Scene SC;
    std::vector<VertexDescriptorRef> VDRs;
    std::vector<TechniqueRef> PRs;
    //*DBG*/Model MS;
    //*DBG*/DescriptorSet SSD;


    // to provide textual feedback
    TextMaker txt;

    // Other application parameters
    float Ar = 0.f; // Aspect ratio

    glm::mat4 ViewPrj = {};
    glm::mat4 World = {};
    glm::vec3 Pos = glm::vec3(0, 0, 5);
    glm::vec3 cameraPos = {};
    glm::vec3 cameraLookAt = glm::vec3(0.0f);
    glm::vec3 targetCameraPos = {};

    std::vector<glm::mat4> gemWorlds, treeWorld; // world transforms for each spawned gem
    std::vector<bool> gemsCatched = {false, false, false, false, false, false, false, false, false, false};
    int gemsCollected = 0;
    int gemsToCollect = 10; // total number of gems to collect
    float gemScale = 0.20f; // scale of the gem model
    float catchRadius = 2.5f;
    float timer = 0.f;
    bool timerDone = false;
    float gemAngle = 0.0f;
    float spinAngle = 0.0f;
    float menuCameraAngle = 0.0f;
    float gameOverCameraAngle = 0.0f;
    float spinVelocity = 5.f;
    float targetSpinVelocity = 5.f;
    const float maxSpinVelocity = 25.f;
    const float minSpinVelocity = 5.f;

    enum GameState { START_MENU, PLAYING, GAME_OVER };
    enum TextID { FPS,
                  GAME_OVER_TEXT,
                  COLLECTED_GEMS_TEXT,
                  TIMER_TEXT,
                  COUNTDOWN_TEXT,
                  INSTRUCTIONS_TEXT };

    GameState gameState = START_MENU;

    float minFov = glm::radians(30.0f);
    float maxFov = glm::radians(80.0f);
    float baseFov = glm::radians(45.0f);
    float boostFovIncrease = glm::radians(15.0f);
    float currentFov = 0.f;
    FastNoise noise, noiseGround;
    std::vector<unsigned char> rawVB_original = {};
    int counter = 0;
    float noiseOffset = 0.0f;
    float shakeIntensity = 0.2f;
    float shakeSpeed = 100.0f;
    std::mt19937 rng;
    std::uniform_real_distribution<float> shakeDist;
    std::uniform_real_distribution<float> distX;
    std::uniform_real_distribution<float> distY;
    std::uniform_real_distribution<float> distZ;

    // Indici per l'aereo
    int airplaneTechIdx = -1;
    int airplaneInstIdx = -1;
    int airplaneRotor = -1;

    // Variabili per il controllo dell'aereo
    glm::vec3 airplanePosition = {};
    glm::vec3 airplaneVelocity = {};
    glm::quat airplaneOrientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    const glm::quat airplaneModelCorrection = glm::angleAxis(glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    // const glm::mat4 rotorModelCorrection = glm::translate(glm::mat4(1.f), glm::vec3(-2.4644f * 0.25, 3.9877f * 0.25, 0.f)) * glm::rotate(glm::mat4(1.f), glm::radians(22.68f), glm::vec3(0.0f, 0.0f, -1.0f)) * glm::scale(glm::mat4(1.f), glm::vec3(0.25f));
    const glm::mat4 rotorModelCorrection = glm::translate(glm::mat4(1.f), glm::vec3(-2.4644f * 0.25, 3.9877f * 0.25, 0.f)) * glm::rotate(glm::mat4(1.f), glm::radians(22.68f), glm::vec3(0.0f, 0.0f, 1.0f)) * glm::rotate(glm::mat4(1.f), glm::radians(180.f), glm::vec3(0.0f, 1.0f, 0.0f)) * glm::scale(glm::mat4(1.f), glm::vec3(0.25f));
    glm::vec3 airplaneScale = glm::vec3(1.0f);
    bool airplaneInitialized = false;
    bool isEngineOn = false;
    bool isAirplaneOnGround = true;
    float visualRollAngle = 0.0f;
    float thrustCoefficient = 200.0f; // Coefficiente di spinta
    float speed = 5.0f;
    float dragCoefficient = 1.0f;
    const float maxSpeed = 20.0f; // m/s, tune to your liking
    glm::vec3 currentShakeOffset = glm::vec3(0.0f);

    dWorldID odeWorld = nullptr;
    dSpaceID odeSpace = nullptr;
    dBodyID odeAirplaneBody = nullptr;
    dGeomID odeAirplaneGeom = nullptr;
    dMass odeAirplaneMass = {};
    // dGeomID odeGroundPlane = nullptr;
    dJointGroupID contactgroup = nullptr;
    std::vector<dReal> triVertices = {};
    std::vector<uint32_t> triIndices = {};
    dTriMeshDataID meshData = nullptr;
    TerrainHeightmap terrain = {};

    const int HF_ROWS = 256;
    const int HF_COLS = 256;
    const float CELL_SIZE = 0.1f;           // world‑space spacing between samples
    const float NOISE_SCALE = 0.004f;       // noise frequency
    const float HEIGHT_SCALE = 0.05f;       // noise amplitude
    std::vector<float> heightSamples; // height samples for the heightfield

    // 1) create the data object & the geom once:
    dHeightfieldDataID hfData = dGeomHeightfieldDataCreate();
    dGeomID            groundHF = nullptr;
    void rebuildHeightSamples(float worldX, float worldZ, float scale){
        // center the grid on the airplane
        float startX = worldX - HF_COLS/2 * CELL_SIZE;
        float startZ = worldZ - HF_ROWS/2 * CELL_SIZE;

        for(int rz = 0; rz < HF_ROWS; ++rz){
            for(int cx = 0; cx < HF_COLS; ++cx){
                float wx = startX + cx * CELL_SIZE;
                float wz = startZ + rz * CELL_SIZE;
                float h  = noiseGround.GetNoise(wx * NOISE_SCALE,
                                                wz * NOISE_SCALE)
                           * HEIGHT_SCALE;
                heightSamples[rz*HF_COLS + cx] = h * scale;
                // std::cout << "Sample at (" << wx << ", " << wz << ") = " << h << "\n";
            }
        }
        groundY = heightSamples[HF_ROWS/2 * HF_COLS + HF_COLS/2];
        // std::cout << "Sample at airplane position: " << heightSamples[HF_ROWS/2 * HF_COLS + HF_COLS/2] << "\n";
    };

    ALCdevice* device = nullptr;
    ALCcontext* context = nullptr;
    ALuint audio_source = -1;
    ALuint audio_buffer = -1;
    ALuint engineBuffers[2];
    ALuint engineSources[2];
    ALuint gemBuffers[10];
    ALuint gemSources[10];
    ALuint gemCollectedSource = -1;
    ALuint gemCollectedBuffer = -1;

    float sourceGains[2] = { 1.f, 0.f };
    float gemSourceGain = 1.f;
    float gemCollectedGain = 0.3f;
    float enginePitch = 1.0f; // Pitch of the engine sound
    const float fadeSpeed = 1.5f; // larger = faster cross‑fade

    // Indici per il pavimento
    int groundTechIdx = -1;
    int groundInstIdx = -1;
    // fix the ground’s Y (height) to whatever you want—say groundY = 0.0f:
    float groundY = 0.1f;
    glm::mat4 groundBaseWm = glm::mat4(1.f);
    Model* ground = nullptr;

    bool activateCountdownTimer = false;
    float timerCountdown = 3.0f;

    bool gameTimerActive = false;
    float gameTime = 120.f;


    // Here you set the main application parameters
    void setWindowParameters()
    {
        // window size, titile and initial background
        windowWidth = 1280;
        windowHeight = 720;
        windowTitle = "Computer Graphics Exam - Airplane Flight";
        windowResizable = GLFW_TRUE;

        // Initial aspect ratio
        Ar = 16.0f / 9.0f;
    }

    // What to do when the window changes size
    void onWindowResize(int w, int h)
    {
        std::cout << "Window resized to: " << w << " x " << h << "\n";
        Ar = (float)w / (float)h;
        // Update Render Pass
        RP.width = w;
        RP.height = h;

        // updates the textual output
        txt.resizeScreen(w, h);
    }

    void updateGroundHeightfield(float scale)
    {
        // 1) refill the sample array around the current airplane XZ
        rebuildHeightSamples( airplanePosition.x, airplanePosition.z, scale);

        // 2) rebuild the heightfield data in place
        dGeomHeightfieldDataBuildSingle(
          hfData,
          heightSamples.data(),
          false,
          HF_COLS * CELL_SIZE,
          HF_ROWS * CELL_SIZE,
          HF_COLS,
          HF_ROWS,
          1.0, 0.0, 1.0, false
        );

        // 3) reposition the geom so it stays centered under the airplane
        dGeomSetPosition(groundHF,
                         airplanePosition.x,
                         0.0f,
                         airplanePosition.z);
    }

    // Here you load and setup all your Vulkan Models and Texutures.
    // Here you also create your Descriptor set layouts and load the shaders for the pipeline
    void localInit()
    {
        currentFov = baseFov;

        glfwSetWindowUserPointer(window, this);
        glfwSetScrollCallback(window, scroll_callback);

        dInitODE();


        std::cout << "Init done!\n";
        std::cout << "ODE Initialized successfully.\n";


        // Descriptor Layouts [what will be passed to the shaders]
        DSLglobal.init(this, {
                           // this array contains the binding:
                           // first  element : the binding number
                           // second element : the type of element (buffer or texture)
                           // third  element : the pipeline stage where it will be used
                           {
                               0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL_GRAPHICS,
                               sizeof(GlobalUniformBufferObject), 1
                           }
                       });

        DSLglobalGround.init(this, {
                                 // this array contains the binding:
                                 // first  element : the binding number
                                 // second element : the type of element (buffer or texture)
                                 // third  element : the pipeline stage where it will be used
                                 {
                                     0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL_GRAPHICS,
                                     sizeof(GlobalUniformBufferGround), 1
                                 }
                             });

        DSLlocalSimp.init(this, {
                              // this array contains the binding:
                              // first  element : the binding number
                              // second element : the type of element (buffer or texture)
                              // third  element : the pipeline stage where it will be used
                              {
                                  0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT,
                                  sizeof(UniformBufferObjectSimp), 1
                              },
                              {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 1},
                              {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1, 1}
                          });

        DSLlocalGem.init(this, {
                             // this array contains the binding:
                             // first  element : the binding number
                             // second element : the type of element (buffer or texture)
                             // third  element : the pipeline stage where it will be used
                             {
                                 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT,
                                 sizeof(UniformBufferObjectSimp), 1
                             },
                             {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 1},
                             {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1, 1}
                         });

        DSLskyBox.init(this, {
                           {
                               0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT,
                               sizeof(skyBoxUniformBufferObject), 1
                           },
                           {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 1}
                       });

        DSLlocalPBR.init(this, {
                             // this array contains the binding:
                             // first  element : the binding number
                             // second element : the type of element (buffer or texture)
                             // third  element : the pipeline stage where it will be used
                             {
                                 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT,
                                 sizeof(UniformBufferObjectGround), 1
                             },
                             {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 1},
                             {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1, 1},
                             {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 2, 1},
                             {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 3, 1}
                         });

        VDsimp.init(this, {
                        {0, sizeof(VertexSimp), VK_VERTEX_INPUT_RATE_VERTEX}
                    }, {
                        {
                            0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexSimp, pos),
                            sizeof(glm::vec3), POSITION
                        },
                        {
                            0, 1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexSimp, norm),
                            sizeof(glm::vec3), NORMAL
                        },
                        {
                            0, 2, VK_FORMAT_R32G32_SFLOAT, offsetof(VertexSimp, UV),
                            sizeof(glm::vec2), UV
                        }
                    });

        VDgem.init(this, {
                       {0, sizeof(VertexSimp), VK_VERTEX_INPUT_RATE_VERTEX}
                   }, {
                       {
                           0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexSimp, pos),
                           sizeof(glm::vec3), POSITION
                       },
                       {
                           0, 1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexSimp, norm),
                           sizeof(glm::vec3), NORMAL
                       },
                       {
                           0, 2, VK_FORMAT_R32G32_SFLOAT, offsetof(VertexSimp, UV),
                           sizeof(glm::vec2), UV
                       }
                   });

        VDskyBox.init(this, {
                          {0, sizeof(skyBoxVertex), VK_VERTEX_INPUT_RATE_VERTEX}
                      }, {
                          {
                              0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(skyBoxVertex, pos),
                              sizeof(glm::vec3), POSITION
                          }
                      });

        VDtan.init(this, {
                       {0, sizeof(VertexTan), VK_VERTEX_INPUT_RATE_VERTEX}
                   }, {
                       {
                           0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexTan, pos),
                           sizeof(glm::vec3), POSITION
                       },
                       {
                           0, 1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexTan, norm),
                           sizeof(glm::vec3), NORMAL
                       },
                       {
                           0, 2, VK_FORMAT_R32G32_SFLOAT, offsetof(VertexTan, UV),
                           sizeof(glm::vec2), UV
                       },
                       {
                           0, 3, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(VertexTan, tan),
                           sizeof(glm::vec4), TANGENT
                       }
                   });

        VDRs.resize(4);
        VDRs[0].init("VDsimp", &VDsimp);
        VDRs[1].init("VDskybox", &VDskyBox);
        VDRs[2].init("VDtan", &VDtan);
        VDRs[3].init("VDgem", &VDgem);

        // initializes the render passes
        RP.init(this);
        // sets the blue sky
        RP.properties[0].clearValue = {0.0f, 0.9f, 1.0f, 1.0f};


        // Pipelines [Shader couples]
        // The last array, is a vector of pointer to the layouts of the sets that will
        // be used in this pipeline. The first element will be set 0, and so on..


        PsimpObj.init(this, &VDsimp, "shaders/SimplePosNormUV.vert.spv", "shaders/CookTorrance.frag.spv",
                      {&DSLglobal, &DSLlocalSimp});
        Pgem.init(this, &VDgem, "shaders/Gem.vert.spv", "shaders/CookTorranceGem.frag.spv",
                  {&DSLglobal, &DSLlocalGem});

        PskyBox.init(this, &VDskyBox, "shaders/SkyBoxShader.vert.spv", "shaders/SkyBoxShader.frag.spv", {&DSLskyBox});
        PskyBox.setCompareOp(VK_COMPARE_OP_LESS_OR_EQUAL);
        PskyBox.setCullMode(VK_CULL_MODE_BACK_BIT);
        PskyBox.setPolygonMode(VK_POLYGON_MODE_FILL);

        P_PBR.init(this, &VDtan, "shaders/SimplePosNormUvTan.vert.spv", "shaders/PBR.frag.spv",
                   {&DSLglobalGround, &DSLlocalPBR});

        PRs.resize(4);

        PRs[0].init("CookTorranceNoiseSimp", {
                        {
                            &PsimpObj, {
                                //Pipeline and DSL for the first pass
                                /*DSLglobal*/{},
                                /*DSLlocalSimp*/{
                                    /*t0*/{true, 0, {}}, // index 0 of the "texture" field in the json file
                                    /*t1*/{true, 1, {}} // index 1 of the "texture" field in the json file
                                }
                            }
                        }
                    }, /*TotalNtextures*/2, &VDsimp);
        PRs[1].init("CookTorranceGem", {
                        {
                            &Pgem, {
                                //Pipeline and DSL for the first pass
                                /*DSLglobal*/{},
                                /*DSLlocalSimp*/{
                                    /*t0*/{true, 0, {}}, // index 0 of the "texture" field in the json file
                                    /*t1*/{true, 1, {}} // index 1 of the "texture" field in the json file
                                }
                            }
                        }
                    }, /*TotalNtextures*/2, &VDgem);
        PRs[2].init("SkyBox", {
                        {
                            &PskyBox, {
                                //Pipeline and DSL for the first pass
                                /*DSLskyBox*/{
                                    /*t0*/{true, 0, {}} // index 0 of the "texture" field in the json file
                                }
                            }
                        }
                    }, /*TotalNtextures*/1, &VDskyBox);
        PRs[3].init("PBR", {
                        {
                            &P_PBR, {
                                //Pipeline and DSL for the first pass
                                /*DSLglobal*/{},
                                /*DSLlocalPBR*/{
                                    /*t0*/{true, 0, {}}, // index 0 of the "texture" field in the json file
                                    /*t1*/{true, 1, {}}, // index 1 of the "texture" field in the json file
                                    /*t2*/{true, 2, {}}, // index 2 of the "texture" field in the json file
                                    /*t3*/{true, 3, {}} // index 3 of the "texture" field in the json file
                                }
                            }
                        }
                    }, /*TotalNtextures*/4, &VDtan);


        // Models, textures and Descriptors (values assigned to the uniforms)

        // sets the size of the Descriptor Set Pool
        DPSZs.uniformBlocksInPool = 30;
        DPSZs.texturesInPool = 30;
        DPSZs.setsInPool = 30;

        std::cout << "\nLoading the scene\n\n";
        if (SC.init(this, /*Npasses*/1, VDRs, PRs, "assets/models/scene.json") != 0)
        {
            std::cout << "ERROR LOADING THE SCENE\n";
            exit(0);
        }

        // Cerca l'indice della tecnica e dell'istanza dell'aereo e del pavimento
        for (int i = 0; i < PRs.size(); i++)
        {
            for (int j = 0; j < SC.TI[i].InstanceCount; j++)
            {
                if (SC.TI[i].I[j].id != nullptr && *(SC.TI[i].I[j].id) == "ap")
                {
                    airplaneTechIdx = i;
                    airplaneInstIdx = j;
                    airplaneRotor = j + 1;
                    break;
                }
            }
            if (airplaneTechIdx != -1)
            {
                break;
            }
        }

        if (airplaneTechIdx != -1)
        {
            std::cout << "Airplane 'ap' found at Technique " << airplaneTechIdx << ", Instance " << airplaneInstIdx <<
                "\n";
            const dReal lx = 2.0, ly = 0.5, lz = 3.0; // example box dimensions

            odeWorld = dWorldCreate();
            odeSpace = dSimpleSpaceCreate(0);
            dWorldSetGravity(odeWorld, 0, -9.81, 0); // Imposta la gravità!
            contactgroup = dJointGroupCreate(0);
            // odeGroundPlane = dCreatePlane(odeSpace, 0, 1, 0, groundY);

            // build the heightfield data once
            heightSamples = std::vector<float>(HF_ROWS * HF_COLS);
            rebuildHeightSamples( airplanePosition.x, airplanePosition.z, 100.f);
            dGeomHeightfieldDataBuildSingle(
              hfData,
              heightSamples.data(),           // pointer to your float array
              /*bCopyHeightData=*/false,      // keep our array alive, no internal copy
              /*width=*/ HF_COLS * CELL_SIZE, // X‑extent in world units
              /*depth=*/ HF_ROWS * CELL_SIZE, // Z‑extent in world units
              /*widthSamples=*/  HF_COLS,
              /*depthSamples=*/  HF_ROWS,
              /*scale=*/ 1.0f,                // scale the raw height values
              /*offset=*/ 0.0f,               // add this to every height
              /*thickness=*/ 1.0f,            // thickness under the lowest sample
              /*bWrap=*/ false                // do not tile
            );

            // now create the placeable geom in your collision space
            groundHF = dCreateHeightfield(odeSpace, hfData, /*bPlaceable=*/true);
            dGeomSetPosition(groundHF,
                             airplanePosition.x,
                             0.0f,
                             airplanePosition.z);
            // Crea il corpo rigido per l'aereo
            odeAirplaneBody = dBodyCreate(odeWorld);
            dBodySetPosition(odeAirplaneBody, airplanePosition.x, airplanePosition.y, airplanePosition.z);
            dBodySetAngularDamping(odeAirplaneBody, 0.5f);
            // Imposta la massa del corpo. È importante per la simulazione.
            // Iniziamo con una massa di 100kg e la distribuiamo come una sfera.
            dMassSetZero(&odeAirplaneMass);
            dMassSetBoxTotal(&odeAirplaneMass, 80.0f, lx, ly, lz);
            dBodySetMass(odeAirplaneBody, &odeAirplaneMass);

            // Crea una geometria per le collisioni (per ora una semplice sfera)
            // Anche se non hai collisioni, è buona norma averla.

            odeAirplaneGeom = dCreateBox(odeSpace, lx, ly, lz);
            dGeomSetBody(odeAirplaneGeom, odeAirplaneBody);
            // --- FINE SETUP ODE ---
        }
        else
        {
            std::cout << "WARNING: Airplane 'ap' not found in scene.\n";
        }

        // initializes the textual output
        txt.init(this, windowWidth, windowHeight);

        // submits the main command buffer
        submitCommandBuffer("main", 0, populateCommandBufferAccess, this);

        // Prepares for showing the FPS count
        txt.print(1.0f, 1.0f, "FPS:", FPS, "CO", false, false, true, TAL_RIGHT, TRH_RIGHT, TRV_BOTTOM,
                  {1.0f, 0.0f, 0.0f, 1.0f}, {0.8f, 0.8f, 0.0f, 1.0f});

        // Adding randomisation of gems
        std::random_device rd;
        rng = std::mt19937(rd());
        shakeDist = std::uniform_real_distribution<float>(-0.1f, 0.1f);
        distX = std::uniform_real_distribution<float>(-100.0f, 100.0f);
        distY = std::uniform_real_distribution<float>(10.0f, 80.0f);
        distZ = std::uniform_real_distribution<float>(-100.0f, 100.0f);

        noise.SetSeed(1337);
        noise.SetNoiseType(FastNoise::Perlin);

        noiseGround.SetSeed(1356);
        noiseGround.SetFrequency(2.f);
        noiseGround.SetNoiseType(FastNoise::Perlin);
        noiseGround.SetFractalOctaves(2);
        noiseGround.SetFractalGain(0.8f);

        gemWorlds.resize(10);
        for (auto& M : gemWorlds)
        {
            M =
                glm::translate(glm::mat4(1.0f),
                               glm::vec3(distX(rng), distY(rng), distZ(rng)))
                * glm::scale(glm::mat4(1.0f), glm::vec3(0.f));
        }


        audioInit();

        for (int i = 0; i < PRs.size(); i++)
        {
            for (int j = 0; j < SC.TI[i].InstanceCount; j++)
            {
                if (SC.TI[i].I[j].id != nullptr && *(SC.TI[i].I[j].id) == "2DplaneTan")
                {
                    groundTechIdx = i;
                    groundInstIdx = j;
                    break;
                }
            }
            if (groundTechIdx != -1)
            {
                break;
            }
        }
        if (groundTechIdx != -1)
        {
            std::cout << "Ground '2Dplane' found at Technique " << groundTechIdx << ", Instance " << groundInstIdx <<
                "\n";
            groundBaseWm = SC.TI[groundTechIdx].I[groundInstIdx].Wm;
        }
        else
        {
            std::cout << "WARNING: Ground '2Dplane' not found in scene.\n";
        }

        auto groundMeshId = SC.MeshIds["2DplaneTan"];
        if (groundMeshId == -1)
        {
            std::cout << "ERROR: Ground mesh '2DplaneTan' not found in scene.\n";
            exit(0);
        }
        else {
            std::cout << "Ground mesh '2DplaneTan' found with ID: " << groundMeshId << "\n";
            ground = SC.M[ groundMeshId ];
            rawVB_original = ground->vertices;
            // right after you fill `ground->vertices` for the very first time:
            size_t byteSize = ground->vertices.size();  // bytes of your interleaved array
            ground->initDynamicVertexBuffer(this /* your BaseProject ptr */, byteSize);
            ground->updateVertexBuffer();


            int totalVerts = rawVB_original.size() / ground->VD->Bindings[0].stride;
            int widthSamples = sqrt(totalVerts);      // assuming square grid
            int depthSamples = widthSamples;


            glm::vec3 scale;
            scale.x = glm::length(glm::vec3(groundBaseWm[0]));
            scale.y = glm::length(glm::vec3(groundBaseWm[1]));
            scale.z = glm::length(glm::vec3(groundBaseWm[2]));

            float meshWidth  = scale.x * (widthSamples - 1);  // world size in X
            float meshDepth  = scale.z * (depthSamples - 1);  // world size in Z

            float cellW = meshWidth / (widthSamples - 1);
            float cellD = meshDepth / (depthSamples - 1);

            std::cout << "Ground mesh size: " << meshWidth << " x " << meshDepth << "\n";
            std::cout << "Ground mesh samples: " << widthSamples << " x " << depthSamples << "\n";
            std::cout << "Ground mesh cell size: " << cellW << " x " << cellD << "\n";
            terrain.init(
                widthSamples,
                depthSamples,
                cellW,
                cellD,
                airplanePosition.x - meshWidth * 0.5f,
                airplanePosition.z - meshDepth * 0.5f,
                [this](float x, float z) {
                    return noiseGround.GetNoise(x * 0.004f, z * 0.004f) * 0.05f;
                }
            );
        }

        treeWorld.resize(16);
        for (auto& M : treeWorld)
        {
            auto X = distX(rng);
            auto Y = distY(rng);
            auto Z = distZ(rng);
            M =
                glm::translate(glm::mat4(1.0f),
                               glm::vec3(X, Y, Z));
        }

        assert(groundTechIdx >= 0 && groundInstIdx >= 0);

        std::cout << "Init done!\n";
    }

    // Here you create your pipelines and Descriptor Sets!
    void pipelinesAndDescriptorSetsInit()
    {
        // creates the render pass
        RP.create();

        // This creates a new pipeline (with the current surface), using its shaders for the provided render pass
        PsimpObj.create(&RP);
        PskyBox.create(&RP);
        P_PBR.create(&RP);
        Pgem.create(&RP);

        SC.pipelinesAndDescriptorSetsInit();
        txt.pipelinesAndDescriptorSetsInit();
    }

    // Here you destroy your pipelines and Descriptor Sets!
    void pipelinesAndDescriptorSetsCleanup()
    {
        PsimpObj.cleanup();
        PskyBox.cleanup();
        P_PBR.cleanup();
        RP.cleanup();
        Pgem.cleanup();

        SC.pipelinesAndDescriptorSetsCleanup();
        txt.pipelinesAndDescriptorSetsCleanup();
    }

    // Here you destroy all the Models, Texture and Desc. Set Layouts you created!
    // You also have to destroy the pipelines
    void localCleanup()
    {
        // --- Cleanup di ODE ---
        if (airplaneInitialized)
        {
            dGeomDestroy(odeAirplaneGeom);
            dBodyDestroy(odeAirplaneBody);
            dSpaceDestroy(odeSpace);
            dWorldDestroy(odeWorld);
        }

        dCloseODE();

        DSLlocalSimp.cleanup();
        DSLlocalPBR.cleanup();
        DSLskyBox.cleanup();
        DSLglobal.cleanup();
        DSLlocalGem.cleanup();

        PsimpObj.destroy();
        PskyBox.destroy();
        P_PBR.destroy();
        Pgem.destroy();

        RP.destroy();

        SC.localCleanup();
        txt.localCleanup();

        audioCleanUp();
    }

    // Here it is the creation of the command buffer:
    // You send to the GPU all the objects you want to draw,
    // with their buffers and textures
    static void populateCommandBufferAccess(VkCommandBuffer commandBuffer, int currentImage, void* Params)
    {
        // Simple trick to avoid having always 'T->'
        // in che code that populates the command buffer!
        std::cout << "Populating command buffer for " << currentImage << "\n";
        E09* T = (E09*)Params;
        T->populateCommandBuffer(commandBuffer, currentImage);
    }

    // This is the real place where the Command Buffer is written
    void populateCommandBuffer(VkCommandBuffer commandBuffer, int currentImage)
    {
        std::cout << "Let's command buffer!";
        // begin standard pass
        RP.begin(commandBuffer, currentImage);

        SC.populateCommandBuffer(commandBuffer, 0, currentImage);

        RP.end(commandBuffer);
    }

    // =================================================================================
    // Funzioni Helper Modulari
    // =================================================================================
    void shift2Dplane() {
        if (gameState != GAME_OVER) {
            ground->vertices = rawVB_original;
            std::vector<unsigned char>& rawVB = ground->vertices;

            size_t stride    = ground->VD->Bindings[0].stride;
            size_t posOffset = ground->VD->Position.offset;  // byte‑offset in each vertex

            glm::vec3 worldOffset = airplanePosition;
            glm::vec3 scale;
            scale.x = glm::length(glm::vec3(groundBaseWm[0]));
            scale.y = glm::length(glm::vec3(groundBaseWm[1]));
            scale.z = glm::length(glm::vec3(groundBaseWm[2]));

            const float NOISE_SCALE  = 0.004f;
            const float HEIGHT_SCALE = 0.05f;

            float lx = 0.f, lz = 0.f, wx = 0.f, wz = 0.f, h = 0.f;

            for (size_t i = 0; i < rawVB.size(); i += stride) {
                glm::vec3* p =
                    reinterpret_cast<glm::vec3*>(&rawVB[i + posOffset]);

                // local XZ:
                lx = p->x * scale.x , lz = p->z * scale.z;

                wx = lx + worldOffset.x;
                wz = lz + worldOffset.z;

                h = noiseGround.GetNoise(wx * NOISE_SCALE,
                                               wz * NOISE_SCALE)
                        * HEIGHT_SCALE;
                p->y = h;

            }
            // ------- Normal, tangent and bi-tanget modification -------

            // 1) Allocate accumulators
            size_t vertexCount = rawVB.size() / stride;
            std::vector<glm::vec3> nAccum(vertexCount, glm::vec3(0.0f));
            std::vector<glm::vec3> tAccum(vertexCount, glm::vec3(0.0f));
            std::vector<glm::vec3> bAccum(vertexCount, glm::vec3(0.0f));

            // 2) Loop over every triangle to accumulate normals & tangents
            for (size_t tri = 0; tri < ground->indices.size(); tri += 3) {
                uint32_t i0 = ground->indices[tri + 0];
                uint32_t i1 = ground->indices[tri + 1];
                uint32_t i2 = ground->indices[tri + 2];

                // read positions
                auto p0 = reinterpret_cast<glm::vec3*>(&rawVB[i0*stride + posOffset]);
                auto p1 = reinterpret_cast<glm::vec3*>(&rawVB[i1*stride + posOffset]);
                auto p2 = reinterpret_cast<glm::vec3*>(&rawVB[i2*stride + posOffset]);

                glm::vec3 P0 = *p0, P1 = *p1, P2 = *p2;

                // read UVs
                auto uv0 = reinterpret_cast<glm::vec2*>(&rawVB[i0*stride + ground->VD->UV.offset]);
                auto uv1 = reinterpret_cast<glm::vec2*>(&rawVB[i1*stride + ground->VD->UV.offset]);
                auto uv2 = reinterpret_cast<glm::vec2*>(&rawVB[i2*stride + ground->VD->UV.offset]);

                glm::vec2 UV0 = *uv0, UV1 = *uv1, UV2 = *uv2;

                // face normal
                glm::vec3 edge1 = P1 - P0;
                glm::vec3 edge2 = P2 - P0;
                glm::vec3 faceN = glm::normalize(glm::cross(edge1, edge2));

                // accumulate
                nAccum[i0] += faceN;
                nAccum[i1] += faceN;
                nAccum[i2] += faceN;

                // compute tangent & bitangent
                glm::vec2 dUV1 = UV1 - UV0;
                glm::vec2 dUV2 = UV2 - UV0;
                float r = 1.0f / (dUV1.x * dUV2.y - dUV2.x * dUV1.y);
                glm::vec3 tangent = (edge1 * dUV2.y - edge2 * dUV1.y) * r;
                glm::vec3 bitan   = (edge2 * dUV1.x - edge1 * dUV2.x) * r;

                tAccum[i0] += tangent;  bAccum[i0] += bitan;
                tAccum[i1] += tangent;  bAccum[i1] += bitan;
                tAccum[i2] += tangent;  bAccum[i2] += bitan;
            }

            // 3) Orthonormalize per-vertex and write back into rawVB
            for (size_t vi = 0; vi < vertexCount; ++vi) {
                // normalize accumulated normal
                glm::vec3 n = glm::normalize(nAccum[vi]);

                // Gram‑Schmidt tangent
                glm::vec3 t = tAccum[vi];
                t = glm::normalize(t - n * glm::dot(n, t));

                // handedness
                float h = (glm::dot(glm::cross(n, t), bAccum[vi]) < 0.0f) ? -1.0f : +1.0f;

                // write back
                auto dstN = reinterpret_cast<glm::vec3*>(&rawVB[vi*stride + ground->VD->Normal.offset]);
                *dstN = n;

                auto dstT = reinterpret_cast<glm::vec4*>(&rawVB[vi*stride + ground->VD->Tangent.offset]);
                *dstT = glm::vec4(t, h);
            }

            // ---------------------------------------------------
            ground->updateVertexBuffer();

            int totalVerts = rawVB_original.size() / stride;
            int widthSamples = sqrt(totalVerts);      // assuming square grid
            int depthSamples = widthSamples;

            float meshWidth  = scale.x * (widthSamples - 1);  // world size in X
            float meshDepth  = scale.z * (depthSamples - 1);  // world size in Z

            float cellW = meshWidth / (widthSamples - 1);
            float cellD = meshDepth / (depthSamples - 1);

            terrain.init(
                widthSamples,
                depthSamples,
                cellW,
                cellD,
                airplanePosition.x - meshWidth * 0.5f,
                airplanePosition.z - meshDepth * 0.5f,
                [this](float x, float z) {
                    return noiseGround.GetNoise(x * 0.004f, z * 0.004f) * 0.05f;
                }
            );
        }
        updateGroundHeightfield(glm::length(glm::vec3(groundBaseWm[1])));
    }

    void handleMouseScroll(double yoffset)
    {
        const float FOV_SENSITIVITY = glm::radians(2.5f);
        baseFov -= yoffset * FOV_SENSITIVITY;
        baseFov = glm::clamp(baseFov, minFov, maxFov);
    }

    // Callback per le collisioni
    static void nearCallback(void* data, dGeomID o1, dGeomID o2)
    {
        E09* app = (E09*)data; // Recupera il puntatore all'applicazione

        dBodyID b1 = dGeomGetBody(o1);
        dBodyID b2 = dGeomGetBody(o2);

        // Ignora collisioni tra oggetti statici
        if (b1 && b2 && dBodyIsKinematic(b1) && dBodyIsKinematic(b2)) return;

        const int MAX_CONTACTS = 10; // Massimo numero di punti di contatto
        dContact contact[MAX_CONTACTS];

        int numc = dCollide(o1, o2, MAX_CONTACTS, &contact[0].geom, sizeof(dContact));

        if (numc > 0)
        {
            for (int i = 0; i < numc; i++)
            {
                contact[i].surface.mode = dContactBounce | dContactSoftCFM;
                contact[i].surface.mu = 0.9; // Attrito
                contact[i].surface.bounce = 0.0; // Rimbalzo
                contact[i].surface.bounce_vel = 0.0;
                contact[i].surface.soft_cfm = 0.001;

                dJointID c = dJointCreateContact(app->odeWorld, app->contactgroup, &contact[i]);
                dJointAttach(c, b1, b2);
            }
        }
    }

    // --- Helper per la gestione dell'input con debouncing ---
    // Restituisce true solo sul primo frame in cui il tasto viene premuto.
    bool handleDebouncedKeyPress(int key)
    {
        static std::map<int, bool> keyDebounceState;
        if (glfwGetKey(window, key) == GLFW_PRESS)
        {
            if (!keyDebounceState[key])
            {
                keyDebounceState[key] = true;
                return true;
            }
        }
        else
        {
            keyDebounceState[key] = false;
        }
        return false;
    }

    // --- Gestione input da tastiera per debug/azioni ---
    void handleKeyboardInput()
    {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE))
        {
            glfwSetWindowShouldClose(window, GL_TRUE);
        }
        if (handleDebouncedKeyPress(GLFW_KEY_1)) currentCameraMode = FIRST_PERSON;
        if (handleDebouncedKeyPress(GLFW_KEY_2)) currentCameraMode = THIRD_PERSON;
        if (handleDebouncedKeyPress(GLFW_KEY_H))
        {
            if (gameState == PLAYING)
            {
                activateCountdownTimer = true;
                timerCountdown = 3.0f;
            }
        }

        if (handleDebouncedKeyPress(GLFW_KEY_F))
        {
            isEngineOn = !isEngineOn;
            if (isEngineOn) targetSpinVelocity = maxSpinVelocity;
            else targetSpinVelocity = minSpinVelocity;
            std::cout << "Engine state: " << (isEngineOn ? "ON" : "OFF") << "\n";
        }
    }

    // --- Prototipo della funzione per gestire l'accelerazione ---
    void handleAirplaneBoost(GLFWwindow* window, float deltaT, float& currentSpeedMultiplier,
                             float AIRPLANE_FORWARD_SPEED, glm::vec3& localForward, glm::vec3& airplanePosition)
    {
        const float BOOST_MULTIPLIER = 2.0f; // Moltiplicatore di velocità quando si preme spazio
        const float ACCELERATION_RATE = 2.0f; // Velocità di accelerazione/decelerazione

        bool isBoosting = (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS);

        // maxSpeed = isBoosting ? 40.0f : 20.0f; // Velocità massima in base allo stato del motore
        // // Calcola il moltiplicatore di velocità target
        // float targetSpeedMultiplier = isBoosting ? BOOST_MULTIPLIER : 1.0f;
        //
        // // Interpola gradualmente il moltiplicatore di velocità corrente verso il target
        // float deltaSpeedMultiplier = targetSpeedMultiplier - currentSpeedMultiplier;
        // currentSpeedMultiplier += deltaSpeedMultiplier * ACCELERATION_RATE * deltaT;
        //
        // // Applica il moltiplicatore alla velocità di movimento
        // float currentForwardSpeed = AIRPLANE_FORWARD_SPEED * currentSpeedMultiplier;
        //
        // airplanePosition += localForward * (currentForwardSpeed * deltaT);
    }

    // --- Aggiorna stato e animazioni ---
    void updateState(float deltaT)
    {
        // Animazione di rotazione delle gemme
        const float GEM_SPIN_SPEED = glm::two_pi<float>() / 5.0f;
        gemAngle += GEM_SPIN_SPEED * deltaT;
        if (gemAngle > glm::two_pi<float>())
        {
            gemAngle -= glm::two_pi<float>();
        }
    }

    // --- Aggiorna tutti gli Uniform Buffer per il frame corrente ---
    void updateUniforms(uint32_t currentImage, float deltaT)
    {
        shift2Dplane();
        const int SIMP_TECH_INDEX = 0, GEM_TECH_INDEX = 1, SKY_TECH_INDEX = 2, PBR_TECH_INDEX = 3;

        const glm::mat4 lightView = glm::rotate(glm::mat4(1), glm::radians(-30.0f), glm::vec3(0.0f, 1.0f, 0.0f)) *
            glm::rotate(glm::mat4(1), glm::radians(-45.0f), glm::vec3(1.0f, 0.0f, 0.0f));
        GlobalUniformBufferObject gubo{};
        gubo.lightDir = glm::vec3(lightView * glm::vec4(0.0f, 0.0f, 1.0f, 1.0f));
        gubo.lightColor = glm::vec4(1.0f);
        gubo.eyePos = cameraPos;
        GlobalUniformBufferGround guboground{};
        guboground.lightDir = glm::vec3(lightView * glm::vec4(0.0f, 0.0f, 1.0f, 1.0f));
        guboground.lightColor = glm::vec4(1.0f);
        guboground.eyePos = cameraPos;
        guboground.eyePosNoSmooth = gameState != GAME_OVER ? airplanePosition : cameraPos;
        guboground.groundHeight = groundY; // Y component of the ground base world matrix

        UniformBufferObjectSimp ubos{};
        for (int inst_idx = 0; inst_idx < SC.TI[SIMP_TECH_INDEX].InstanceCount; ++inst_idx)
        {
            if (inst_idx <= 1) ubos.mMat = SC.TI[SIMP_TECH_INDEX].I[inst_idx].Wm;
            else ubos.mMat = treeWorld[inst_idx - 2];
            ubos.mvpMat = ViewPrj * ubos.mMat;
            ubos.nMat = glm::inverse(glm::transpose(ubos.mMat));
            SC.TI[SIMP_TECH_INDEX].I[inst_idx].DS[0][0]->map(currentImage, &gubo, 0);
            SC.TI[SIMP_TECH_INDEX].I[inst_idx].DS[0][1]->map(currentImage, &ubos, 0);
        }

        UniformBufferObjectGround ubogpbr{};
        for (int inst_idx = 0; inst_idx < SC.TI[PBR_TECH_INDEX].InstanceCount; ++inst_idx)
        {
            ubogpbr.mMat = SC.TI[PBR_TECH_INDEX].I[inst_idx].Wm;
            ubogpbr.mvpMat = ViewPrj * ubogpbr.mMat;
            ubogpbr.nMat = glm::inverse(glm::transpose(ubogpbr.mMat));
            // glm::mat4 groundBaseWmYshift = glm::translate(groundBaseWm, glm::vec3(0.0f, groundY, 0.0f));
            ubogpbr.worldMat = groundBaseWm;
            SC.TI[PBR_TECH_INDEX].I[inst_idx].DS[0][0]->map(currentImage, &guboground, 0);
            SC.TI[PBR_TECH_INDEX].I[inst_idx].DS[0][1]->map(currentImage, &ubogpbr, 0);
        }

        UniformBufferObjectSimp uboGem{};
        glm::mat4 spinY = glm::rotate(glm::mat4(1.0f), gemAngle, glm::vec3(0, 1, 0));
        for (int inst_idx = 0; inst_idx < SC.TI[GEM_TECH_INDEX].InstanceCount; ++inst_idx)
        {
            uboGem.mMat = gemWorlds[inst_idx] * spinY * glm::rotate(glm::mat4(1.0f), glm::radians(90.0f),
                                                                    glm::vec3(1, 0, 0)) * glm::scale(
                glm::mat4(1.0f), glm::vec3(gemScale));
            uboGem.mvpMat = ViewPrj * uboGem.mMat;
            uboGem.nMat = glm::inverse(glm::transpose(uboGem.mMat));
            SC.TI[GEM_TECH_INDEX].I[inst_idx].DS[0][0]->map(currentImage, &gubo, 0);
            SC.TI[GEM_TECH_INDEX].I[inst_idx].DS[0][1]->map(currentImage, &uboGem, 0);
        }


        if (SC.TI[SKY_TECH_INDEX].InstanceCount > 0)
        {
            skyBoxUniformBufferObject sbubo{};
            sbubo.mvpMat = ViewPrj * glm::translate(glm::mat4(1), cameraPos) * glm::scale(
                glm::mat4(1), glm::vec3(100.0f));
            SC.TI[SKY_TECH_INDEX].I[0].DS[0][0]->map(currentImage, &sbubo, 0);
        }


        // Aggiornamento HUD (precedentemente in una funzione separata)
        static float elapsedT = 0.0f;
        static int countedFrames = 0;

        countedFrames++;
        elapsedT += deltaT;
        if (elapsedT > 1.0f)
        {
            float fps = (float)countedFrames / elapsedT;
            std::ostringstream oss;
            oss << "FPS: " << std::fixed << std::setprecision(1) << fps;
            txt.print(1.0f, 1.0f, oss.str(), FPS, "CO", false, false, true, TAL_RIGHT, TRH_RIGHT, TRV_BOTTOM,
                      {1.0f, 0.0f, 0.0f, 1.0f}, {0.8f, 0.8f, 0.0f, 1.0f});
            elapsedT = 0.0f;
            countedFrames = 0;
        }
        txt.updateCommandBuffer();
    }

    // =================================================================================
    // Funzione Principale di Aggiornamento - Versione Unificata
    // =================================================================================

    void updateUniformBuffer(uint32_t currentImage)
    {

        float deltaT;
        glm::vec3 m, r;
        bool fire;
        getSixAxis(deltaT, m, r, fire);
        // 4. Esegui la logica principale a doppia modalità (Aereo o Personaggio)
        glm::mat4 viewMatrix;

        // Assicuriamoci che deltaT non sia zero o negativo per evitare instabilità
        if (deltaT <= 0.0f) deltaT = 0.0001f;

        if (airplaneTechIdx != -1 && !airplaneInitialized)
        {
            const glm::mat4& initialWm = SC.TI[airplaneTechIdx].I[airplaneInstIdx].Wm;
            airplanePosition = glm::vec3(initialWm[3]);
            dBodySetPosition(odeAirplaneBody, airplanePosition.x, airplanePosition.y, airplanePosition.z);
            dBodySetLinearDamping(odeAirplaneBody, 0.f);
            // FIX: Aggiunto damping angolare per maggiore stabilità
            dBodySetAngularDamping(odeAirplaneBody, 0.5f);
            airplaneScale = glm::vec3(glm::length(glm::vec3(initialWm[0])), glm::length(glm::vec3(initialWm[1])),
                                      glm::length(glm::vec3(initialWm[2])));
            if (airplaneScale.x == 0.0f) airplaneScale.x = 1.0f;
            if (airplaneScale.y == 0.0f) airplaneScale.y = 1.0f;
            if (airplaneScale.z == 0.0f) airplaneScale.z = 1.0f;
            glm::mat3 rotationPart = glm::mat3(initialWm);
            rotationPart[0] /= airplaneScale.x;
            rotationPart[1] /= airplaneScale.y;
            rotationPart[2] /= airplaneScale.z;
            airplaneOrientation = glm::normalize(glm::quat_cast(rotationPart) * airplaneModelCorrection);
            airplaneInitialized = true;

            dBodySetLinearDamping(odeAirplaneBody, 0.005f); // adjust between 0.1–10.0

            // dBodySetAngularDamping(odeAirplaneBody, 0.9f); // adjust between 0.1–1.0
            // dBodySetAngularDampingThreshold(odeAirplaneBody, 0.01f); // adjust to your liking
        }


        if (gameState == START_MENU && airplaneInitialized)
        {
            const float ROTATION_SPEED = 0.4f; // Velocità di rotazione in radianti al secondo
            const float CAMERA_DISTANCE = 12.0f; // Distanza dall'aereo
            const float CAMERA_HEIGHT = 3.0f;   // Altezza
            menuCameraAngle += ROTATION_SPEED * deltaT;
            if (menuCameraAngle > glm::two_pi<float>()) {
                menuCameraAngle -= glm::two_pi<float>();
            }
            glm::vec3 cameraOffset(
                sin(menuCameraAngle) * CAMERA_DISTANCE,
                CAMERA_HEIGHT,
                cos(menuCameraAngle) * CAMERA_DISTANCE
            );
            cameraPos = airplanePosition + cameraOffset;

            cameraLookAt = airplanePosition;
            ViewPrj = glm::perspective(currentFov, Ar, 1.f, 500.0f);
            glm::mat4 projectionMatrix = glm::perspective(currentFov, Ar, 1.f, 500.f);
            projectionMatrix[1][1] *= -1;
            viewMatrix = glm::lookAt(cameraPos, cameraLookAt, glm::vec3(0.0f, 1.0f, 0.0f));

            glm::mat4 airplaneGlobal =
                                glm::translate(glm::mat4(1.0f), airplanePosition) *
                                glm::mat4_cast( airplaneOrientation /* skip modelCorrection here */ ) *
                                glm::scale  (glm::mat4(1.0f), airplaneScale);

            glm::mat4 rotorLocal =
                glm::translate(glm::mat4(1.0f), glm::vec3(-2.4644f, 3.9877f, 0.0f)) *
                glm::rotate   (glm::mat4(1.0f), glm::radians(22.68f), glm::vec3(0,0,-1));

            spinAngle += deltaT * spinVelocity;
            if (spinAngle > glm::two_pi<float>()) {
                spinAngle -= glm::two_pi<float>();
            }
            glm::mat4 rotorSpin =
                glm::rotate(glm::mat4(1.0f), spinAngle, glm::vec3(1,0,0));
            SC.TI[airplaneTechIdx].I[airplaneRotor].Wm = airplaneGlobal * rotorLocal * rotorSpin;

            ViewPrj = projectionMatrix * viewMatrix;

            // updateUniforms(currentImage, deltaT);

            // 4) Stampa il testo “Premi P per iniziare”
            txt.print(0.f, 0.f, "PREMI P PER INIZIARE", INSTRUCTIONS_TEXT, "CO", true, false, true, TAL_CENTER, TRH_CENTER, TRV_MIDDLE, {1, 1, 1, 1}, {0, 0, 0, 1}, {0, 0, 0, 0}, 2, 2);
            txt.updateCommandBuffer();
            // 5) Controlla P
            if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS)
            {
                txt.removeText(INSTRUCTIONS_TEXT);
                gameState = PLAYING;
            }
        }
        else if (gameState == PLAYING)
        {
            handleKeyboardInput();

            bool isBoosting = (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS);
            float targetFov = baseFov;
            if (isBoosting && currentCameraMode == THIRD_PERSON)
            {
                targetFov += boostFovIncrease;
            }
            float fovInterpSpeed = 5.0f;
            currentFov = glm::mix(currentFov, targetFov, fovInterpSpeed * deltaT);

            if (activateCountdownTimer)
            {
                timerCountdown -= deltaT;

                if (timerCountdown > 0.0f)
                {
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(0) << timerCountdown;
                    txt.print(0.f, -0.f, oss.str(), COUNTDOWN_TEXT, "CO", true, false, true, TAL_CENTER, TRH_CENTER, TRV_MIDDLE,
                              {1, 1, 1, 1}, {0, 0, 0, 1}, {0, 0, 0, 0}, 2, 2);
                }
                else
                {
                    activateCountdownTimer = false; // Resetta il timer una volta scaduto
                    txt.removeText(COUNTDOWN_TEXT);
                    gameTimerActive = true;
                    initGems();
                    // gameTime = 5.0f;
                }
            }

            if (gameTimerActive)
            {
                timer += deltaT;
                if (timer < gameTime && gemsCollected < gemsToCollect)
                {
                    // Formatta il tempo in MM:SS
                    int minutes = static_cast<int>(gameTime - timer) / 60;
                    int seconds = static_cast<int>(gameTime - timer) % 60;
                    std::ostringstream oss;
                    oss << std::setw(2) << std::setfill('0') << minutes << ":"
                        << std::setw(2) << std::setfill('0') << seconds;

                    // Visualizza il secondo timer (1 minuto) usando un ID diverso (es. 5)
                    txt.print(0.5f, 0.5f, oss.str(), TIMER_TEXT, "CO", true, false, true, TAL_CENTER, TRH_CENTER);
                }
                else
                {
                    // Il secondo timer è scaduto
                    gameTimerActive = false;
                    txt.removeText(TIMER_TEXT); // Rimuove il testo del timer
                    txt.removeText(COLLECTED_GEMS_TEXT); // Rimuove il testo delle gemme
                    gameState = GAME_OVER;
                    // Calcola l'offset corrente della telecamera rispetto all'aereo
                    glm::vec3 cameraOffset = cameraPos - airplanePosition;
                    // Calcola l'angolo iniziale per la rotazione della telecamera
                    // usando atan2 per ottenere l'angolo nel piano XZ
                    gameOverCameraAngle = atan2(cameraOffset.x, cameraOffset.z);
                }
            }

            if (airplaneInitialized)
            {
                const dReal* velocity = dBodyGetLinearVel(odeAirplaneBody);
                const dReal* pos = dBodyGetPosition(odeAirplaneBody);
                glm::vec3 globalVel{ velocity[0], velocity[1], velocity[2] };
                float magSpeed = glm::length(globalVel);


                const float basePitchAccel = 25.f; // m/s^2, tune to your liking
                const float baseYawAccel = 20.f; // m/s^2, tune to your liking
                const float baseRollAccel = 100.f; // m/s^2, tune to your liking

                float a_pitch = basePitchAccel * (magSpeed / maxSpeed);
                float a_yaw   = baseYawAccel   * (magSpeed / maxSpeed);
                float a_roll  = baseRollAccel * (magSpeed / maxSpeed);

                // assuming inertia tensor is diagonal in body frame
                const float inertiaScale = 1.f; // tune to your liking
                const dReal Ixx = odeAirplaneMass.I[0] * inertiaScale;
                const dReal Iyy = odeAirplaneMass.I[5] * inertiaScale;
                const dReal Izz = odeAirplaneMass.I[10] * inertiaScale;

                if (magSpeed > 0.001f) {
                    // compute drag magnitude
                    float dragMag = dragCoefficient * magSpeed * magSpeed;

                    // drag always opposes motion
                    glm::vec3 dragDir = -globalVel / magSpeed;
                    glm::vec3 dragForce = dragMag * dragDir;

                    // simplest: apply in world‐space
                    dBodyAddForce(odeAirplaneBody,
                                  dragForce.x,
                                  dragForce.y,
                                  dragForce.z);
                }
                const float takeoffSpeed = 5.0f; // m/s, tune to your liking
                // if (magSpeed > takeoffSpeed) {
                //     const float rho = 1.225f; // kg/m^3, density of air at sea level
                //     const float wingArea = 10.0f; // m^2, tune to your liking
                //     const float CL = 1.0f; // lift coefficient, tune to your liking
                //     float liftMag = 0.5f * rho * magSpeed * magSpeed * wingArea * CL;
                //     const dReal* q = dBodyGetQuaternion(odeAirplaneBody);
                //     glm::quat orient(q[0], q[1], q[2], q[3]);
                //     glm::vec3 localUp = orient * glm::vec3(0,1,0);
                //     dBodyAddForce(odeAirplaneBody,
                //                   liftMag * localUp.x,
                //                   liftMag * localUp.y,
                //                   liftMag * localUp.z);
                // }

                isAirplaneOnGround = (pos[1] <= groundY + 0.1f);
                bool keysPressed = false;
                // const dReal* q = dBodyGetQuaternion(odeAirplaneBody);
                // glm::quat Q{
                //     static_cast<float>(q[0]), static_cast<float>(q[1]), static_cast<float>(q[2]),
                //     static_cast<float>(q[3])
                // };

                if (isAirplaneOnGround)
                {
                    // --- CONTROLLO A TERRA (TIPO AUTOMOBILE) ---

                    // if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) // gira a sinistra
                    // if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) // gira a destra
                }
                else
                {
                    // --- CONTROLLO IN VOLO (AERODINAMICO) ---
                    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
                    {
                        keysPressed = true;
                        dBodyAddRelTorque(odeAirplaneBody,
                                          0,
                                          +Iyy * a_yaw,
                                          0);
                        dBodyAddRelTorque(odeAirplaneBody,
                                          +Izz * a_roll,
                                          0,
                                          0);

                        // --- 3) Lateral “skid” force ---
                        //  a) get the body → world rotation quaternion
                        const dReal* q = dBodyGetQuaternion(odeAirplaneBody);
                        glm::quat Q{
                            static_cast<float>(q[0]), static_cast<float>(q[1]), static_cast<float>(q[2]),
                            static_cast<float>(q[3])
                        };

                        // b) compute body‑space right axis ( +Z or +Y depending on convention;
                        //    here we assume body +Z is right wing, adjust if yours is different )
                        glm::vec3 leftB = glm::normalize(glm::vec3(-0.5, 0, 1));

                        // c) rotate it into world space:
                        glm::vec3 leftW = Q * leftB;

                        // d) pick a lateral force magnitude (tune this!)
                        float lateralForceMag = 500.0f; // e.g. 500 N

                        // e) apply that force sideways at the CG
                        glm::vec3 F = leftW * lateralForceMag;
                        dBodyAddForce(odeAirplaneBody, F.x, F.y, F.z);
                    }
                    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
                    {
                        keysPressed = true;
                        // --- 1) Roll torque (roll to the right) ---
                        dBodyAddRelTorque(odeAirplaneBody,
                                          -Izz * a_roll, // body‑x axis roll
                                          0,
                                          0);

                        // --- 2) Yaw torque (turn nose right) ---
                        dBodyAddRelTorque(odeAirplaneBody,
                                          0,
                                          -Iyy * a_yaw,
                                          0);

                        // --- 3) Lateral “skid” force ---
                        //  a) get the body → world rotation quaternion
                        const dReal* q = dBodyGetQuaternion(odeAirplaneBody);
                        glm::quat Q{
                            static_cast<float>(q[0]), static_cast<float>(q[1]), static_cast<float>(q[2]),
                            static_cast<float>(q[3])
                        };

                        // b) compute body‑space right axis ( +Z or +Y depending on convention;
                        //    here we assume body +Z is right wing, adjust if yours is different )
                        glm::vec3 rightB = glm::normalize(glm::vec3(-0.5, 0, -1));

                        // c) rotate it into world space:
                        glm::vec3 rightW = Q * rightB;

                        // d) pick a lateral force magnitude (tune this!)
                        float lateralForceMag = 500.0f; // e.g. 500 N

                        // e) apply that force sideways at the CG
                        glm::vec3 F = rightW * lateralForceMag;
                        dBodyAddForce(odeAirplaneBody, F.x, F.y, F.z);
                    }

                    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
                    {
                        keysPressed = true;
                        // positive pitch (nose up)
                        dBodyAddRelTorque(odeAirplaneBody,
                                          0,
                                          0, +Ixx * a_pitch);

                        const float rho = 1.225f; // kg/m^3, density of air at sea level
                        const float wingArea = 10.0f; // m^2, tune to your liking
                        const float CL = 1.0f; // lift coefficient, tune to your liking
                        float liftMag = 0.5f * rho * magSpeed * magSpeed * wingArea * CL;
                        const dReal* q = dBodyGetQuaternion(odeAirplaneBody);
                        glm::quat orient(q[0], q[1], q[2], q[3]);
                        glm::vec3 localUp = orient * glm::vec3(0, -1, 0);
                        dBodyAddForce(odeAirplaneBody,
                                      liftMag * localUp.x,
                                      liftMag * localUp.y,
                                      liftMag * localUp.z);
                    }
                    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
                    {
                        keysPressed = true;
                        // positive pitch (nose up)
                        dBodyAddRelTorque(odeAirplaneBody,
                                          0,
                                          0, -Ixx * a_pitch);

                        const float rho = 1.225f; // kg/m^3, density of air at sea level
                        const float wingArea = 10.0f; // m^2, tune to your liking
                        const float CL = 1.0f; // lift coefficient, tune to your liking
                        float liftMag = 0.5f * rho * magSpeed * magSpeed * wingArea * CL;
                        const dReal* q = dBodyGetQuaternion(odeAirplaneBody);
                        glm::quat orient(q[0], q[1], q[2], q[3]);
                        glm::vec3 localUp = orient * glm::vec3(0, 1, 0);
                        dBodyAddForce(odeAirplaneBody,
                                      liftMag * localUp.x,
                                      liftMag * localUp.y,
                                      liftMag * localUp.z);
                    }
                }

                if (!keysPressed) {
                    // Stabilizzatore di solo rollio
                    const dReal* q = dBodyGetQuaternion(odeAirplaneBody);
                    glm::quat currentOrientation(q[0], q[1], q[2], q[3]);

                    // Calcola il vettore "destra" dell'aereo nello spazio del mondo
                    glm::vec3 worldRight = currentOrientation * glm::vec3(0, 0, 1); // Assumendo +Z come destra nel modello

                    // Proietta il vettore "destra" sul piano orizzontale del mondo (XZ)
                    glm::vec3 projectedRight = glm::normalize(glm::vec3(worldRight.x, 0.0f, worldRight.z));

                    // Calcola l'angolo di rollio (inclinazione)
                    // L'asse Y del vettore "destra" del mondo indica l'inclinazione
                    float rollAngle = -worldRight.y; // Usa il negativo a seconda della convenzione

                    // Calcola la coppia di correzione attorno all'asse avanti dell'aereo
                    glm::vec3 worldForward = currentOrientation * glm::vec3(-1, 0, 0); // Assumendo -X come avanti
                    glm::vec3 rollTorque = worldForward * rollAngle * 4000.0f; // Aumenta la costante per una correzione più forte

                    // Applica la coppia per stabilizzare il rollio
                    dBodyAddTorque(odeAirplaneBody, rollTorque.x, rollTorque.y, rollTorque.z);
                    // pitch stabilizer (body torque)
                    float pitchError = worldForward.z;
                    glm::vec3 pitchTorque = glm::vec3(0, 0, 1) * -pitchError * 5000.0f;
                    // dBodyAddRelTorque(odeAirplaneBody, pitchTorque.x, pitchTorque.y, pitchTorque.z);
                }

                if (isEngineOn && magSpeed > takeoffSpeed) {
                    dWorldSetGravity(odeWorld, 0, 0.f, 0); // Imposta la gravità!
                }else {
                    dWorldSetGravity(odeWorld, 0, -9.81, 0); // Imposta la gravità!
                }

                if (isEngineOn) {
                    const float thrustMagnitude = thrustCoefficient * speed; // tune this
                    dReal fx = thrustMagnitude;
                    dReal fy = 0;
                    dReal fz = 0;
                    dBodyAddRelForce(odeAirplaneBody, -fx, fy, fz);
                }


                if (magSpeed > maxSpeed) {
                    // glm::dvec3 v_clamped = v * (maxSpeed / speed);
                    // dBodySetLinearVel(odeAirplaneBody, v_clamped.x, v_clamped.y, v_clamped.z);
                    // Optionally clear applied forces so they don't instantly re-accelerate:
                    dBodySetForce(odeAirplaneBody, 0, 0, 0);
                }

                dSpaceCollide(odeSpace, this, &nearCallback);
                const dReal stepSize = deltaT;
                dWorldStep(odeWorld, stepSize);

                dJointGroupEmpty(contactgroup);

                pos = dBodyGetPosition(odeAirplaneBody);
                const dReal* rot = dBodyGetQuaternion(odeAirplaneBody);
                airplanePosition = glm::vec3(pos[0], pos[1], pos[2]);
                airplaneOrientation = glm::quat(rot[0], rot[1], rot[2], rot[3]);

                glm::quat finalOrientation = airplaneOrientation * airplaneModelCorrection;

                SC.TI[airplaneTechIdx].I[airplaneInstIdx].Wm =
                    glm::translate(glm::mat4(1.0f), airplanePosition) *
                    glm::mat4_cast(finalOrientation) *
                    glm::scale(glm::mat4(1.0f), airplaneScale);

                glm::mat4 airplaneGlobal =
                                    glm::translate(glm::mat4(1.0f), airplanePosition) *
                                    glm::mat4_cast( airplaneOrientation /* skip modelCorrection here */ ) *
                                    glm::scale  (glm::mat4(1.0f), airplaneScale);

                glm::mat4 rotorLocal =
                    glm::translate(glm::mat4(1.0f), glm::vec3(-2.4644f, 3.9877f, 0.0f)) *
                    glm::rotate   (glm::mat4(1.0f), glm::radians(22.68f), glm::vec3(0,0,-1));

                const float ROTOR_SMOOTHING = 1.0f;
                float rotorInterpFactor = 1.0f - glm::exp(-ROTOR_SMOOTHING * deltaT);
                spinVelocity = glm::mix(spinVelocity, targetSpinVelocity, rotorInterpFactor);

                spinAngle += deltaT * spinVelocity;
                if (spinAngle > glm::two_pi<float>()) {
                    spinAngle -= glm::two_pi<float>();
                }
                glm::mat4 rotorSpin =
                    glm::rotate(glm::mat4(1.0f), spinAngle, glm::vec3(1,0,0));
                SC.TI[airplaneTechIdx].I[airplaneRotor].Wm = airplaneGlobal * rotorLocal * rotorSpin;

                // --- Logica della Telecamera ---
                glm::vec3 cameraOffset;
                glm::vec3 targetCameraLookAt = airplanePosition;

                if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
                {
                    cameraOffset = glm::vec3(0.0f, 5.0f, -20.0f);
                }
                else if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
                {
                    cameraOffset = glm::vec3(0.0f, 5.0f, 20.0f);
                }
                else if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS)
                {
                    cameraOffset = glm::vec3(-20.0f, 5.0f, 0.0f);
                }
                else if (currentCameraMode == FIRST_PERSON)
                {
                    cameraOffset = glm::vec3(-0.5f, 0.5f, 0.0f);
                    glm::vec3 forwardDirection = airplaneOrientation * glm::vec3(-1.0f, 0.0f, 0.0f);
                    targetCameraLookAt = airplanePosition + (airplaneOrientation * cameraOffset) + forwardDirection;
                }
                else
                {
                    // THIRD_PERSON (default)
                    cameraOffset = glm::vec3(15.0f, 1.5f, 0.0f);
                }

                targetCameraPos = airplanePosition + (airplaneOrientation * cameraOffset);

                const float CAMERA_SMOOTHING = 5.0f;
                float cameraInterpFactor = 1.0f - glm::exp(-CAMERA_SMOOTHING * deltaT);
                cameraPos = glm::mix(cameraPos, targetCameraPos, cameraInterpFactor);
                cameraLookAt = glm::mix(cameraLookAt, targetCameraLookAt, cameraInterpFactor);

                glm::vec3 targetShakeOffset = glm::vec3(0.0f);
                if (isBoosting)
                {
                    noiseOffset += deltaT * shakeSpeed;
                    glm::vec3 localShake = glm::vec3(
                        0.0f,
                        noise.GetNoise(noiseOffset, 10.0f) * shakeIntensity,
                        noise.GetNoise(noiseOffset, 20.0f) * shakeIntensity
                    );
                    targetShakeOffset = finalOrientation * localShake;
                    const float thrustMagnitude = thrustCoefficient * speed * 30.0f; // tune this
                    dReal fx = thrustMagnitude;
                    dReal fy = 0;
                    dReal fz = 0;
                    dBodyAddRelForce(odeAirplaneBody, -fx, fy, fz);
                    const float PITCH_INTERP_SPEED = 1.0f;
                    float pitchInterpFactor = 1.0f - glm::exp(-PITCH_INTERP_SPEED * deltaT);
                    enginePitch = glm::mix(enginePitch, 1.3f, pitchInterpFactor);
                    alSourcef(engineSources[1], AL_PITCH, enginePitch); // Aumenta il pitch del suono del motore
                }
                else
                {
                    const float PITCH_INTERP_SPEED = 1.0f;
                    float pitchInterpFactor = 1.0f - glm::exp(-PITCH_INTERP_SPEED * deltaT);
                    enginePitch = glm::mix(enginePitch, 1.f, pitchInterpFactor);
                    alSourcef(engineSources[1], AL_PITCH, enginePitch); // Resetta il pitch del suono del motore
                }
                const float SHAKE_INTERP_SPEED = 10.0f;
                float shakeInterpFactor = 1.0f - glm::exp(-SHAKE_INTERP_SPEED * deltaT);
                currentShakeOffset = glm::mix(currentShakeOffset, targetShakeOffset, shakeInterpFactor);
                glm::vec3 finalCameraPos = cameraPos + currentShakeOffset;

                // FIX: L'asse "up" della camera deve seguire il rollio dell'aereo per evitare scatti
                glm::vec3 cameraUp = glm::normalize(airplaneOrientation * glm::vec3(0.0f, 1.0f, 0.0f));
                viewMatrix = glm::lookAt(finalCameraPos, cameraLookAt, cameraUp);
            }
            GameLogic();
        }
        else if (gameState == GAME_OVER)
        {
            if (glfwGetKey(window, GLFW_KEY_ESCAPE))
            {
                glfwSetWindowShouldClose(window, GL_TRUE);
            }

            if (airplaneInitialized)
            {
                /*const float ROTATION_SPEED = 0.4f;
                const float CAMERA_DISTANCE = 15.0f;
                const float CAMERA_HEIGHT = 5.0f;

                gameOverCameraAngle += ROTATION_SPEED * deltaT;
                if (gameOverCameraAngle > glm::two_pi<float>()) {
                    gameOverCameraAngle -= glm::two_pi<float>();
                }

                // Calcola la nuova posizione della telecamera basata sull'angolo
                glm::vec3 cameraOffset(
                    sin(gameOverCameraAngle) * CAMERA_DISTANCE,
                    CAMERA_HEIGHT,
                    cos(gameOverCameraAngle) * CAMERA_DISTANCE
                );
                cameraPos = airplanePosition + cameraOffset;
                cameraLookAt = airplanePosition;*/


                // Aggiorna le matrici di vista e proiezione
                glm::mat4 projectionMatrix = glm::perspective(currentFov, Ar, 1.f, 500.f);
                projectionMatrix[1][1] *= -1;
                viewMatrix = glm::lookAt(cameraPos, cameraLookAt, glm::vec3(0.0f, 1.0f, 0.0f));
                ViewPrj = projectionMatrix * viewMatrix;

                dSpaceCollide(odeSpace, this, &nearCallback);
                dWorldStep(odeWorld, deltaT);
                dJointGroupEmpty(contactgroup);

                // Aggiorna posizione e orientamento dell'aereo dal motore fisico
                const dReal* pos = dBodyGetPosition(odeAirplaneBody);
                const dReal* rot = dBodyGetQuaternion(odeAirplaneBody);
                airplanePosition = glm::vec3(pos[0], pos[1], pos[2]);
                airplaneOrientation = glm::quat(rot[0], rot[1], rot[2], rot[3]);
                const float thrustMagnitude = thrustCoefficient * speed; // tune this
                dReal fx = thrustMagnitude;
                dReal fy = 0;
                dReal fz = 0;
                dBodyAddRelForce(odeAirplaneBody, -fx, fy, fz);

                // Aggiorna la matrice del modello dell'aereo
                SC.TI[airplaneTechIdx].I[airplaneInstIdx].Wm =
                    glm::translate(glm::mat4(1.0f), airplanePosition) *
                    glm::mat4_cast(airplaneOrientation * airplaneModelCorrection) *
                    glm::scale(glm::mat4(1.0f), airplaneScale);

                glm::mat4 airplaneGlobal =
                                glm::translate(glm::mat4(1.0f), airplanePosition) *
                                glm::mat4_cast( airplaneOrientation /* skip modelCorrection here */ ) *
                                glm::scale  (glm::mat4(1.0f), airplaneScale);

                glm::mat4 rotorLocal =
                    glm::translate(glm::mat4(1.0f), glm::vec3(-2.4644f, 3.9877f, 0.0f)) *
                    glm::rotate   (glm::mat4(1.0f), glm::radians(22.68f), glm::vec3(0,0,-1));


                const float ROTOR_SMOOTHING = 5.0f;
                float rotorInterpFactor = 1.0f - glm::exp(-ROTOR_SMOOTHING * deltaT);
                spinVelocity = glm::mix(spinVelocity, targetSpinVelocity, rotorInterpFactor);

                spinAngle += deltaT * spinVelocity;
                if (spinAngle > glm::two_pi<float>()) {
                    spinAngle -= glm::two_pi<float>();
                }
                glm::mat4 rotorSpin =
                    glm::rotate(glm::mat4(1.0f), spinAngle, glm::vec3(1,0,0));
                SC.TI[airplaneTechIdx].I[airplaneRotor].Wm = airplaneGlobal * rotorLocal * rotorSpin;
            }

            // Mostra il messaggio di fine gioco
            std::ostringstream oss;
            if (gemsCollected == gemsToCollect) {
                oss << "Hai raccolto tutte le gemme!";
            }
            else {
                oss << "Hai raccolto solo " << gemsCollected << " gemme su " << gemsToCollect << ". Hai perso!";
            }

            oss << "\nFine del gioco! Premi esc per uscire";

            txt.print(0.f, 0.f, oss.str(), GAME_OVER_TEXT, "SS", false, true, true, TAL_CENTER,
                      TRH_CENTER, TRV_MIDDLE, {0, 0, 0, 1}, {0, 0, 0, 1}, {0, 0, 0, 0}, 1, 1);

            // Aggiorna gli uniformi e il command buffer alla fine
            // updateUniforms(currentImage, deltaT);
        }

        glm::mat4 projectionMatrix = glm::perspective(currentFov, Ar, 1.f, 500.f);
        projectionMatrix[1][1] *= -1;

        ViewPrj = projectionMatrix * viewMatrix;

        updateUniforms(currentImage, deltaT);

        alListener3f(AL_POSITION, cameraPos.x, cameraPos.y, cameraPos.z);
        alListener3f(AL_VELOCITY, airplaneVelocity.x, airplaneVelocity.y, airplaneVelocity.z);
        for (unsigned int engineSource : engineSources) {
            alSource3f(engineSource, AL_POSITION, airplanePosition.x, airplanePosition.y, airplanePosition.z);
        }

        glm::vec3 forward = glm::normalize(cameraLookAt - cameraPos);
        glm::vec3 worldUp(0.0f, 1.0f, 0.0f);
        glm::vec3 right = glm::normalize(glm::cross(forward, worldUp));
        glm::vec3 up = glm::cross(right, forward);
        float ori[6] = {
            forward.x, forward.y, forward.z,
            up.x, up.y, up.z
        };
        alListenerfv(AL_ORIENTATION, ori);

        glm::mat4 groundXzFollow = glm::translate(
            glm::mat4(1.0f),
            glm::vec3(airplanePosition.x, 0, airplanePosition.z)
        );

        if (groundTechIdx >= 0 && groundInstIdx >= 0 && gameState != GAME_OVER)
        {
            SC.TI[groundTechIdx].I[groundInstIdx].Wm = groundXzFollow * groundBaseWm;
        }

        updateState(deltaT);
        updateEngineAudio(deltaT);

    }


    void GameLogic()
    {
        for (int i = 0; i < gemWorlds.size(); i++)
        {
            auto gemPos = glm::vec3(gemWorlds[i][3]);

            if (const float dist = glm::distance(gemPos, airplanePosition); dist < catchRadius && !gemsCatched[i])
            {
                gemsCatched[i] = true;
                gemWorlds[i] = glm::translate(glm::mat4(1.0f), gemPos)
                    * glm::scale(glm::mat4(1.0f), glm::vec3(0.0f));
                gemsCollected++;
                std::ostringstream oss;
                oss << "Gems collected: " << std::fixed << std::setprecision(1) << gemsCollected << "/" << gemWorlds.size();
                txt.print(-1.0f, -1.0f, oss.str(), COLLECTED_GEMS_TEXT, "SS", false, false, true, TAL_LEFT, TRH_LEFT, TRV_TOP,
                          {0.0f, 0.0f, 0.0f, 1.0f}, {1.f, 1.f, 1.f, 1.0f});
                alSourceStop(gemSources[i]);
                alSourcePlay(gemCollectedSource);
            }
        }
    }

    void initGems() {
        distX = std::uniform_real_distribution<float>(airplanePosition.x - 100.0f, airplanePosition.x + 100.0f);
        distY = std::uniform_real_distribution<float>(10.0f, 80.0f);
        distZ = std::uniform_real_distribution<float>(airplanePosition.z - 100.0f, airplanePosition.z + 100.0f);
        for (int i = 0; i < gemWorlds.size(); i++)
        {
            alSourceStop(gemSources[i]);
            auto& M = gemWorlds[i];
            auto xRandom = distX(rng);
            auto zRandom = distZ(rng);
            auto yRandom = distY(rng) + terrain.sampleHeight(xRandom, zRandom);
            M = glm::translate(glm::mat4(1.0f), {xRandom, yRandom, zRandom}) * glm::scale(
                glm::mat4(1.0f), glm::vec3(gemScale));

            // Move source audio position to gem position
            alSource3f(gemSources[i], AL_POSITION, xRandom, yRandom, zRandom);
            gemsCatched[i] = false; // Reset catch status
            gemsCollected = 0; // Reset collected gems count
            alSourcePlay(gemSources[i]);
        }
        std::ostringstream oss;
        oss << "Gems collected: " << std::fixed << std::setprecision(1) << gemsCollected << "/" << gemWorlds.size();
        txt.print(-1.0f, -1.0f, oss.str(), COLLECTED_GEMS_TEXT, "SS", false, false, true, TAL_LEFT, TRH_LEFT, TRV_TOP,
                          {0.0f, 0.0f, 0.0f, 1.0f}, {1.f, 1.f, 1.f, 1.0f});
    }

    // void initGroundCollision()
    // {
    //     size_t stride    = ground->VD->Bindings[0].stride;
    //     size_t posOffset = ground->VD->Position.offset;  // byte‑offset in each vertex
    //     // 1) build triVertices & triIndices from your rawVB_original / ground->indices:
    //     triVertices.reserve(rawVB_original.size() / stride * 3);
    //     triIndices   = ground->indices;                // copy once
    //
    //     for (size_t i = 0; i < rawVB_original.size(); i += stride) {
    //         glm::vec3* p = reinterpret_cast<glm::vec3*>(&rawVB_original[i + posOffset]);
    //         triVertices.push_back(p->x);
    //         triVertices.push_back(0.0f);  // start flat
    //         triVertices.push_back(p->z);
    //     }
    //
    //     // 2) create and build the meshData
    //     meshData = dGeomTriMeshDataCreate();
    //     dGeomTriMeshDataBuildSimple(
    //       meshData,
    //       triVertices.data(),  triVertices.size()/3,
    //       triIndices.data(),   triIndices.size()/3
    //     );
    //
    //     // 3) make a single trimesh geom
    //     // odeGroundPlane = dCreateTriMesh(odeSpace, meshData,
    //                                     // nullptr, nullptr, nullptr);
    // }

    void audioInit()
    {
        // 1) Open default device & create context
        device = alcOpenDevice(nullptr);
        if (!device)
        {
            std::cerr << "Failed to open audio device\n";
            return;
        }
        context = alcCreateContext(device, nullptr);
        if (!context || !alcMakeContextCurrent(context))
        {
            std::cerr << "Failed to create/make context\n";
            if (context) alcDestroyContext(context);
            alcCloseDevice(device);
            return;
        }

        // 3) Set up the listener (camera) defaults
        //    Position at origin, no velocity
        alListener3f(AL_POSITION, 0.0f, 0.0f, 0.0f);
        alListener3f(AL_VELOCITY, 0.0f, 0.0f, 0.0f);

        //    Orientation: facing down −Z, with +Y as up
        float listenerOri[] = {
            0.0f, 0.0f, -1.0f, // “forward” vector
            0.0f, 1.0f, 0.0f // “up” vector
        };
        alListenerfv(AL_ORIENTATION, listenerOri);

        alGenSources(1, &audio_source);
        // alSource3f(audio_source, AL_POSITION, 0, 0, 0);
        // alSource3f(audio_source, AL_VELOCITY, 0, 0, 0);
        alDistanceModel(AL_INVERSE_DISTANCE_CLAMPED);

        loadWavToBuffer(audio_buffer, "assets/audios/audio.wav");
        loadWavToBuffer(engineBuffers[0], "assets/audios/engine_idle.wav");
        loadWavToBuffer(engineBuffers[1], "assets/audios/engine_running.wav");
        loadWavToBuffer(gemCollectedBuffer, "assets/audios/gem_collected.wav");

        alSourcei(audio_source, AL_BUFFER, audio_buffer);
        alSourcei(audio_source, AL_LOOPING, AL_TRUE);
        alSourcef(audio_source, AL_GAIN, 0.1f);
        alSourcePlay(audio_source);

        alGenSources(2, engineSources);
        for (int i = 0; i < 2; ++i) {
            alSourcei(engineSources[i], AL_BUFFER, engineBuffers[i]);
            alSourcei(engineSources[i], AL_LOOPING, AL_TRUE);
            alSourcef(engineSources[i], AL_GAIN, sourceGains[i]);
            alSourcei(engineSources[i], AL_SOURCE_RELATIVE, AL_FALSE);
            alSource3f(engineSources[i], AL_POSITION, airplanePosition.x, airplanePosition.y, airplanePosition.z);
            alSourcePlay(engineSources[i]);
        }

        alGenSources(10, gemSources);
        for (int i = 0; i < 10; ++i) {
            loadWavToBuffer(gemBuffers[i], "assets/audios/gem_ambient.wav");
            alSourcei(gemSources[i], AL_BUFFER, gemBuffers[i]);
            alSourcei(gemSources[i], AL_LOOPING, AL_TRUE);
            alSourcef(gemSources[i], AL_GAIN, gemSourceGain);
            //alSourcePlay(gemSources[i]);
        }

        alGenSources(1, &gemCollectedSource);
        alSourcei(gemCollectedSource, AL_BUFFER, gemCollectedBuffer);
        alSourcei(gemCollectedSource, AL_LOOPING, AL_FALSE);
        alSourcef(gemCollectedSource, AL_GAIN, gemCollectedGain);
        alSourcei(gemCollectedSource, AL_SOURCE_RELATIVE, AL_TRUE);
        alSource3f(gemCollectedSource, AL_POSITION, 0.0f, 0.0f, 0.0f);


    }

    void audioCleanUp()
    {
        alDeleteSources(1, &audio_source);
        alDeleteBuffers(1, &audio_buffer);
        alcMakeContextCurrent(nullptr);
        alcDestroyContext(context);
        alcCloseDevice(device);
    }

    void loadWavToBuffer(ALuint& buffer, const char* fileName)
    {
        // 2) Load WAV into an OpenAL buffer
        drwav wav;
        if (!drwav_init_file(&wav, fileName, nullptr))
        {
            std::cerr << "Could not open audio.wav\n";
            return;
        }
        size_t totalSamples = wav.totalPCMFrameCount * wav.channels;
        int16_t* pcmData = (int16_t*)malloc(totalSamples * sizeof(int16_t));
        drwav_read_pcm_frames_s16(&wav, wav.totalPCMFrameCount, pcmData);
        drwav_uninit(&wav);

        ALenum format = (wav.channels == 1 ? AL_FORMAT_MONO16 : AL_FORMAT_STEREO16);
        std::cerr << "Format: " << format << std::endl;

        alGenBuffers(1, &buffer);
        alBufferData(buffer, format, pcmData,
                     (ALsizei)(totalSamples * sizeof(int16_t)),
                     wav.sampleRate);
        free(pcmData);
    }

    void updateEngineAudio(float deltaTime) {

        // Decide target gains
        float targetGains[2] = {
            isEngineOn ? 0.f : 1.f,  // idle fades out when engine on
            isEngineOn ? 1.f : 0.f   // running fades in when engine on
        };

        // Smooth‑step towards targets
        for (int i = 0; i < 2; ++i) {
            float diff = targetGains[i] - sourceGains[i];
            // clamp step so we don’t overshoot
            float step = fadeSpeed * deltaTime;
            if (fabs(diff) < step) {
                sourceGains[i] = targetGains[i];
            } else {
                sourceGains[i] += (diff > 0 ? +step : -step);
            }
            alSourcef(engineSources[i], AL_GAIN, sourceGains[i]);
        }
    }

    void exportTriMeshToOBJ(const std::string &path,
                            const std::vector<dReal> &triVertices,
                            const std::vector<uint32_t> &triIndices)
    {
        std::ofstream out(path);
        if(!out) {
            std::cerr << "Failed to open " << path << " for writing\n";
            return;
        }


        // write all vertices
        size_t vcount = triVertices.size()/3;
        out << "# OBJ export of " << vcount << " verts, "
            << (triIndices.size()/3) << " tris\n";
        for(size_t i = 0; i < vcount; ++i) {
            dReal x = triVertices[3*i+0];
            dReal y = triVertices[3*i+1];
            dReal z = triVertices[3*i+2];
            out << "v " << x << " " << y << " " << z << "\n";
        }
        out << "\n";

        // write all faces (OBJ uses 1‑based indices)
        size_t tcount = triIndices.size()/3;
        for(size_t t = 0; t < tcount; ++t) {
            uint32_t i0 = triIndices[3*t+0] + 1;
            uint32_t i1 = triIndices[3*t+1] + 1;
            uint32_t i2 = triIndices[3*t+2] + 1;
            out << "f " << i0 << " " << i1 << " " << i2 << "\n";
        }

        out.close();
        std::cout << "Wrote OBJ mesh to " << path << "\n";
    }

private:
    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
    {
        E09* app = (E09*)glfwGetWindowUserPointer(window);
        if (app)
        {
            app->handleMouseScroll(yoffset);
        }
    }
};


// This is the main: probably you do not need to touch this!
int main()
{
    E09 app;

    try
    {
        app.run();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
