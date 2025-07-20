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
    alignas(16) glm::vec3 referencePosition;
    alignas(16) glm::vec4 otherParams;
};

struct UniformBufferObjectSimp
{
    alignas(16) glm::mat4 mvpMat;
    alignas(16) glm::mat4 mMat;
    alignas(16) glm::mat4 nMat;
};

struct skyBoxUniformBufferObject
{
    alignas(16) glm::mat4 mvpMat;
};

static std::vector<dJointFeedback*> jointFeedbacks;

// MAIN !
class CG_Exam : public BaseProject
{
protected:
    enum CameraMode { FIRST_PERSON, THIRD_PERSON };
    enum ProjectionMode { PERSPECTIVE, ORTHOGRAPHIC, ISOMETRIC };
    ProjectionMode currentProjectionMode = PERSPECTIVE;
    float orthoZoom = 20.0f;

    bool changeTangents = true;

    CameraMode currentCameraMode = THIRD_PERSON;
    // Here you list all the Vulkan objects you need:

    // Descriptor Layouts [what will be passed to the shaders]
    DescriptorSetLayout DSLlocalSimp, DSLlocalPBR, DSLglobal, DSLglobalGround, DSLskyBox;

    // Vertex formants, Pipelines [Shader couples] and Render passes
    VertexDescriptor VDsimp;
    VertexDescriptor VDskyBox;
    VertexDescriptor VDtan;
    RenderPass RP;
    Pipeline PsimpObj, PskyBox, P_PBR, Pgem;

    // Models, textures and Descriptors (values assigned to the uniforms)
    Scene SC;
    std::vector<VertexDescriptorRef> VDRs;
    std::vector<TechniqueRef> PRs;

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
    std::vector<bool> gemsCatched = {true, true, true, true, true, true, true, true, true, true};
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
    float counterGlobal = 0.f;

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
    float noiseOffset = 0.0f;
    float shakeIntensity = 0.2f;
    float shakeSpeed = 100.0f;
    std::mt19937 rng;
    std::uniform_real_distribution<float> shakeDist;
    std::uniform_real_distribution<float> distX;
    std::uniform_real_distribution<float> distY;
    std::uniform_real_distribution<float> distZ;

    std::uniform_real_distribution<float> treeX;
    std::uniform_real_distribution<float> treeZ;

    // Airplane indexes
    int airplaneTechIdx = -1;
    int airplaneInstIdx = -1;
    int airplaneRotor = -1;

    // Airplane parameters
    glm::vec3 airplanePosition = {};
    glm::vec3 airplaneVelocity = {};
    glm::quat airplaneOrientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    const glm::quat airplaneModelCorrection = glm::angleAxis(glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    // const glm::mat4 rotorModelCorrection = glm::translate(glm::mat4(1.f), glm::vec3(-2.4644f * 0.25, 3.9877f * 0.25, 0.f)) * glm::rotate(glm::mat4(1.f), glm::radians(22.68f), glm::vec3(0.0f, 0.0f, -1.0f)) * glm::scale(glm::mat4(1.f), glm::vec3(0.25f));
    const glm::mat4 rotorModelCorrection = glm::translate(glm::mat4(1.f), glm::vec3(-2.4644f * 0.25, 3.9877f * 0.25, 0.f)) * glm::rotate(glm::mat4(1.f), glm::radians(22.68f), glm::vec3(0.0f, 0.0f, 1.0f)) * glm::rotate(glm::mat4(1.f), glm::radians(180.f), glm::vec3(0.0f, 1.0f, 0.0f)) * glm::scale(glm::mat4(1.f), glm::vec3(0.25f));
    glm::vec3 airplaneScale = glm::vec3(1.0f);
    bool airplaneInitialized = false;
    bool isEngineOn = false;
    float thrustCoefficient = 200.0f; // Coefficiente di spinta
    float speed = 5.0f;
    float dragCoefficient = 1.0f;
    const float maxSpeed = 20.0f; // m/s, tune to your liking
    glm::vec3 currentShakeOffset = glm::vec3(0.0f);
    float crashThreshold = 6000.f;
    bool hardImpact = false;
    bool isBoosting = false;
    bool inWater = false;
    float waterLevel = -2.5f;
    float grassLevel = -1.5f;
    float rockLevel = 15.f;

    // ODE parameters
    dWorldID odeWorld = nullptr;
    dSpaceID odeSpace = nullptr;
    dBodyID odeAirplaneBody = nullptr;
    dGeomID odeAirplaneGeom = nullptr;
    dMass odeAirplaneMass = {};
    dJointGroupID contactgroup = nullptr;

    const int HF_ROWS = 256;
    const int HF_COLS = 256;
    const float CELL_SIZE = 0.1f;           // world‑space spacing between samples
    const float NOISE_SCALE = 0.004f;       // noise frequency
    const float HEIGHT_SCALE = 0.05f;       // noise amplitude
    std::vector<float> heightSamples;

    // Where ODE will store the heightfield data
    dHeightfieldDataID hfData = dGeomHeightfieldDataCreate();
    dGeomID            groundHF = nullptr;

    // Noise generator for the ground heightfield, this will update the height of ODE ground geometry
    void rebuildHeightSamples(float worldX, float worldZ, float scale){
        // center the grid on the coordinates passed
        float startX = worldX - HF_COLS/2 * CELL_SIZE;
        float startZ = worldZ - HF_ROWS/2 * CELL_SIZE;
        // loop over the grid and sample the noise
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
        // Set the ground height to the sample at the center of the grid (airplane position)
        groundY = heightSamples[HF_ROWS/2 * HF_COLS + HF_COLS/2];
        // std::cout << "Sample at airplane position: " << heightSamples[HF_ROWS/2 * HF_COLS + HF_COLS/2] << "\n";
    };

    // Audio parameters
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

    // ground indexes
    int groundTechIdx = -1;
    int groundInstIdx = -1;
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

    // This function build the ground heightfield geometry
    void updateGroundHeightfield(float scale)
    {
        // refill the sample array around the current airplane XZ
        rebuildHeightSamples( airplanePosition.x, airplanePosition.z, scale);

        // rebuild the ODE ground geometry
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

        // reposition the geom so it stays centered under the airplane
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
                                 sizeof(UniformBufferObjectSimp), 1
                             },
                             {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 1},
                             {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1, 1},
                             {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 2, 1},
                             {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 3, 1},
                             {5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 4, 1},
                             {6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 5, 1},
                             {7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 6, 1},

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

        VDRs.resize(3);
        VDRs[0].init("VDsimp", &VDsimp);
        VDRs[1].init("VDskybox", &VDskyBox);
        VDRs[2].init("VDtan", &VDtan);

        // initializes the render passes
        RP.init(this);
        // sets the blue sky
        RP.properties[0].clearValue = {0.0f, 0.9f, 1.0f, 1.0f};


        // Pipelines [Shader couples]
        // The last array, is a vector of pointer to the layouts of the sets that will
        // be used in this pipeline. The first element will be set 0, and so on..


        PsimpObj.init(this, &VDsimp, "shaders/SimplePosNormUV.vert.spv", "shaders/CookTorrance.frag.spv",
                      {&DSLglobal, &DSLlocalSimp});
        Pgem.init(this, &VDsimp, "shaders/SimplePosNormUV.vert.spv", "shaders/CookTorranceGem.frag.spv",
                  {&DSLglobal, &DSLlocalSimp});

        PskyBox.init(this, &VDskyBox, "shaders/SkyBoxShader.vert.spv", "shaders/SkyBoxShader.frag.spv", {&DSLskyBox});
        // Here we assure that the skybox is rendered before the other objects, where there is nothing else
        PskyBox.setCompareOp(VK_COMPARE_OP_LESS_OR_EQUAL);

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
                    }, /*TotalNtextures*/2, &VDsimp);
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
                                     /*t3*/{true, 3, {}}, // index 3 of the "texture" field in the json file
                                     /*t4*/{true, 4, {}}, // index 4 of the "texture" field in the json file
                                     /*t5*/{true, 5, {}}, // index 5 of the "texture" field in the json file
                                     /*t6*/{true, 6, {}} // index 6 of the "texture" field in the json file
                                 }
                            }
                        }
                    }, /*TotalNtextures*/7, &VDtan);

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

        // Finding index of airplain and rotor
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

            // ODE ground creation
            groundHF = dCreateHeightfield(odeSpace, hfData, /*bPlaceable=*/true);
            dGeomSetPosition(groundHF,
                             airplanePosition.x,
                             0.0f,
                             airplanePosition.z);
            // create rigid body of airplane
            odeAirplaneBody = dBodyCreate(odeWorld);
            dBodySetPosition(odeAirplaneBody, airplanePosition.x, airplanePosition.y, airplanePosition.z);
            dBodySetAngularDamping(odeAirplaneBody, 0.5f);
            // set the initial position and mass of the airplane
            dMassSetZero(&odeAirplaneMass);
            dMassSetBoxTotal(&odeAirplaneMass, 80.0f, lx, ly, lz);
            dBodySetMass(odeAirplaneBody, &odeAirplaneMass);
            odeAirplaneGeom = dCreateBox(odeSpace, lx, ly, lz);
            dGeomSetBody(odeAirplaneGeom, odeAirplaneBody);
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
                  {1.0f, 0.0f, 0.0f, 1.0f}, {0.8f, 0.8f, 0.0f, 1.0f}, {0, 0, 0, 1});

        // setting randomisation variables
        std::random_device rd;
        rng = std::mt19937(rd());
        shakeDist = std::uniform_real_distribution<float>(-0.1f, 0.1f);
        distX = std::uniform_real_distribution<float>(-100.0f, 100.0f);
        distY = std::uniform_real_distribution<float>(10.0f, 80.0f);
        distZ = std::uniform_real_distribution<float>(-100.0f, 100.0f);

        treeX = std::uniform_real_distribution<float>(-500.0f, 500.0f);
        treeZ = std::uniform_real_distribution<float>(-500.0f, 500.0f);

        noise.SetSeed(1337);
        noise.SetNoiseType(FastNoise::Perlin);

        noiseGround.SetSeed(1356);
        noiseGround.SetFrequency(2.f);
        noiseGround.SetNoiseType(FastNoise::Perlin);
        noiseGround.SetFractalOctaves(2);
        noiseGround.SetFractalGain(0.8f);

        // init the gems position randomly but will have no scale
        gemWorlds.resize(10);
        for (auto& M : gemWorlds)
        {
            M =
                glm::translate(glm::mat4(1.0f),
                               glm::vec3(distX(rng), distY(rng), distZ(rng)))
                * glm::scale(glm::mat4(1.0f), glm::vec3(0.f));
        }

        // initialize the audio part
        audioInit();

        // find the 2Dplane
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
            // initialize dynamically the ground
            std::cout << "Ground mesh '2DplaneTan' found with ID: " << groundMeshId << "\n";
            ground = SC.M[ groundMeshId ];
            rawVB_original = ground->vertices;
            // right after you fill `ground->vertices` for the very first time:
            size_t byteSize = ground->vertices.size();  // bytes of your interleaved array
            ground->initDynamicVertexBuffer(this /* your BaseProject ptr */, byteSize);
            ground->updateVertexBuffer();
        }
        // initialize the trees
        treeWorld.resize(400);
        for (auto& M : treeWorld)
        {
            float X, Z;
            float Y = -100.f;
            do {
                X = treeX(rng);
                Z = treeZ(rng);
                Y = noiseGround.GetNoise(X * 0.004f, Z * 0.004f) * 0.05f * 500;
            }while (Y < -0.5f);
            M =
                glm::translate(glm::mat4(1.0f),
                               glm::vec3(X, Y, Z));
        }

        assert(groundTechIdx >= 0 && groundInstIdx >= 0);

        if (airplaneTechIdx != -1)
        {
            // Set real airplane position and orientation if found
            const glm::mat4& initialWm = SC.TI[airplaneTechIdx].I[airplaneInstIdx].Wm;
            airplanePosition = glm::vec3(initialWm[3]);
            dBodySetPosition(odeAirplaneBody, airplanePosition.x, airplanePosition.y, airplanePosition.z);
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
            airplaneOrientation = glm::quat_cast(rotationPart) * airplaneModelCorrection;
            airplaneInitialized = true;

            dBodySetLinearDamping(odeAirplaneBody, 0.005f);
        }

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
        // --- ODE cleanup ---
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

        PsimpObj.destroy();
        PskyBox.destroy();
        P_PBR.destroy();
        Pgem.destroy();

        RP.destroy();

        SC.localCleanup();
        txt.localCleanup();

        audioCleanUp();

        for (auto fb : jointFeedbacks) delete fb;
        jointFeedbacks.clear();

    }

    // Here it is the creation of the command buffer:
    // You send to the GPU all the objects you want to draw,
    // with their buffers and textures
    static void populateCommandBufferAccess(VkCommandBuffer commandBuffer, int currentImage, void* Params)
    {
        // Simple trick to avoid having always 'T->'
        // in che code that populates the command buffer!
        std::cout << "Populating command buffer for " << currentImage << "\n";
        CG_Exam* T = (CG_Exam*)Params;
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

    // This is called every frame, to update the 2Dplane
    void shift2Dplane() {
        if (gameState != GAME_OVER) {
            // Retreving raw bytes of the ground mesh
            ground->vertices = rawVB_original;
            std::vector<unsigned char>& rawVB = ground->vertices;

            // Stride is the size of a single vertex in bytes
            size_t stride    = ground->VD->Bindings[0].stride;
            // Get the position offset in the vertex structure (therefore lower than stride)
            size_t posOffset = ground->VD->Position.offset;

            glm::vec3 worldOffset = airplanePosition;
            glm::vec3 scale;
            scale.x = glm::length(glm::vec3(groundBaseWm[0]));
            scale.y = glm::length(glm::vec3(groundBaseWm[1]));
            scale.z = glm::length(glm::vec3(groundBaseWm[2]));

            const float NOISE_SCALE  = 0.004f;
            const float HEIGHT_SCALE = 0.05f;

            float lx = 0.f, lz = 0.f, wx = 0.f, wz = 0.f, h = 0.f;
            // for each vertex in the ground mesh
            for (size_t i = 0; i < rawVB.size(); i += stride) {
                glm::vec3* p =
                    reinterpret_cast<glm::vec3*>(&rawVB[i + posOffset]);

                // local XZ
                lx = p->x * scale.x , lz = p->z * scale.z;
                // world XZ
                wx = lx + worldOffset.x;
                wz = lz + worldOffset.z;

                // computing height of that vertex
                float rawH    = noiseGround.GetNoise(wx * NOISE_SCALE,
                                                     wz * NOISE_SCALE)
                              * HEIGHT_SCALE;

                // water level
                const float floorY      = (waterLevel - 0.2f) / 500.f;

                // how wide (in world‑units) the blend region is around floorY
                const float blendWidth  = 0.5f / 500.f;

                // blend factor t that goes 0 to 1 as rawH goes from (floorY - blendWidth) up to (floorY + blendWidth)
                float t = glm::smoothstep(floorY - blendWidth,
                                          floorY + blendWidth,
                                          rawH);

                // small animation of water level
                float flatH = floorY
                            - std::abs(noiseGround.GetNoise(wx * NOISE_SCALE,
                                                            wz * NOISE_SCALE,
                                                            counterGlobal * 0.3f)
                                       * 0.001f);

                // mix between the two:
                h = glm::mix(flatH, rawH, t);

                // write back
                p->y = h;

            }
            // ------- Normal, tangent and bi-tanget modification -------
            if (changeTangents) {
                // Allocate accumulators
                size_t vertexCount = rawVB.size() / stride;
                std::vector<glm::vec3> nAccum(vertexCount, glm::vec3(0.0f));
                std::vector<glm::vec3> tAccum(vertexCount, glm::vec3(0.0f));
                std::vector<glm::vec3> bAccum(vertexCount, glm::vec3(0.0f));

                // Loop over every triangle to accumulate normals & tangents
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

                // Orthonormalize per-vertex and write back into rawVB
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

            }
            // ---------------------------------------------------
            ground->updateVertexBuffer();
        }
        // Update the ground heightfield of ODE
        updateGroundHeightfield(glm::length(glm::vec3(groundBaseWm[1])));
    }

    void handleMouseScroll(double yoffset)
    {
        const float FOV_SENSITIVITY = glm::radians(2.5f);
        baseFov -= yoffset * FOV_SENSITIVITY;
        baseFov = glm::clamp(baseFov, minFov, maxFov);
    }

    // Collision callback
    static void nearCallback(void* data, dGeomID o1, dGeomID o2)
    {
        CG_Exam* app = (CG_Exam*)data;

        dBodyID b1 = dGeomGetBody(o1);
        dBodyID b2 = dGeomGetBody(o2);

        if (b1 && b2 && dBodyIsKinematic(b1) && dBodyIsKinematic(b2)) return;

        const int MAX_CONTACTS = 5;
        dContact contact[MAX_CONTACTS];
        // dCollide does the collision test between two geometries
        int numc = dCollide(o1, o2, MAX_CONTACTS, &contact[0].geom, sizeof(dContact));

        if (numc > 0) // if number of contacts is greater than 0...
        {
            // If the bodies are not kinematic, we need to apply forces
            if (b1 && !dBodyIsKinematic(b1)) dBodyAddForce(b1, 0, 0, 0);
            if (b2 && !dBodyIsKinematic(b2)) dBodyAddForce(b2, 0, 0, 0);

            // If the bodies are kinematic, we do not apply forces
            if (b1 && dBodyIsKinematic(b1)) b1 = nullptr;
            if (b2 && dBodyIsKinematic(b2)) b2 = nullptr;
        }
        {
            for (int i = 0; i < numc; i++)
            {
                contact[i].surface.mode = dContactBounce | dContactSoftCFM;
                contact[i].surface.mu = 50.f; // Friction
                contact[i].surface.bounce = 0.f; // Bounce
                contact[i].surface.bounce_vel = 0.0;
                contact[i].surface.soft_cfm = 0.001;

                dJointID c = dJointCreateContact(app->odeWorld, app->contactgroup, &contact[i]);
                dJointAttach(c, b1, b2);

                // allocate feedback struct on the heap (or reuse one per joint)
                dJointFeedback* fb = new dJointFeedback();
                jointFeedbacks.push_back(fb);
                dJointSetFeedback(c, fb);
            }
        }
    }

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

    void handleKeyboardInput()
    {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE))
        {
            glfwSetWindowShouldClose(window, GL_TRUE);
        }
        if (handleDebouncedKeyPress(GLFW_KEY_1)) currentCameraMode = FIRST_PERSON;
        if (handleDebouncedKeyPress(GLFW_KEY_2)) currentCameraMode = THIRD_PERSON;
        if (handleDebouncedKeyPress(GLFW_KEY_F1)) currentProjectionMode = PERSPECTIVE;
        if (handleDebouncedKeyPress(GLFW_KEY_F2)) currentProjectionMode = ORTHOGRAPHIC;
        if (handleDebouncedKeyPress(GLFW_KEY_F3)) currentProjectionMode = ISOMETRIC;

        if (handleDebouncedKeyPress(GLFW_KEY_H))
        {
            if (gameState == PLAYING && !gameTimerActive)
            {
                activateCountdownTimer = true;
                timerCountdown = 3.0f;
            }
        }

        if (handleDebouncedKeyPress(GLFW_KEY_F))
        {
            toggleEngineState();
        }

        if (handleDebouncedKeyPress(GLFW_KEY_U))
        {
            changeTangents = !changeTangents;
        }
        isBoosting = false;
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
            isBoosting = true;
        }
    }

    void updateGemsRotation(float deltaT)
    {
        // gem rotation update
        const float GEM_SPIN_SPEED = glm::two_pi<float>() / 5.0f;
        gemAngle += GEM_SPIN_SPEED * deltaT;
        if (gemAngle > glm::two_pi<float>())
        {
            gemAngle -= glm::two_pi<float>();
        }
    }

    // --- Update all uniform buffers ---
    void updateUniforms(uint32_t currentImage, float deltaT)
    {
        shift2Dplane();
        const int SIMP_TECH_INDEX = 0, GEM_TECH_INDEX = 1, SKY_TECH_INDEX = 2, PBR_TECH_INDEX = 3;

        // Setting uniform buffers
        const glm::mat4 lightView = glm::rotate(glm::mat4(1), glm::radians(-30.0f), glm::vec3(0.0f, 1.0f, 0.0f)) *
            glm::rotate(glm::mat4(1), glm::radians(-45.0f), glm::vec3(1.0f, 0.0f, 0.0f));
        GlobalUniformBufferObject gubo{};
        gubo.lightDir = glm::vec3(lightView * glm::vec4(0.0f, 0.0f, 1.0f, 1.0f));
        gubo.lightColor = glm::vec4(6.0f);
        gubo.eyePos = cameraPos;
        GlobalUniformBufferGround guboground{};
        guboground.lightDir = glm::vec3(lightView * glm::vec4(0.0f, 0.0f, 1.0f, 1.0f));
        guboground.lightColor = glm::vec4(1.0f);
        guboground.eyePos = cameraPos;
        guboground.referencePosition = gameState != GAME_OVER ? airplanePosition : cameraPos;
        guboground.otherParams = glm::vec4(groundY, waterLevel, grassLevel, rockLevel);

        UniformBufferObjectSimp ubos{};
        for (int inst_idx = 0; inst_idx < SC.TI[SIMP_TECH_INDEX].InstanceCount; ++inst_idx)
        {
            // check if the instance is the airplane and rotor (index 0 and 1) or the trees (index 2 and above)
            if (inst_idx <= 1) ubos.mMat = SC.TI[SIMP_TECH_INDEX].I[inst_idx].Wm;
            else ubos.mMat = treeWorld[inst_idx - 2];

            ubos.mvpMat = ViewPrj * ubos.mMat;
            ubos.nMat = glm::inverse(glm::transpose(ubos.mMat));
            SC.TI[SIMP_TECH_INDEX].I[inst_idx].DS[0][0]->map(currentImage, &gubo, 0);
            SC.TI[SIMP_TECH_INDEX].I[inst_idx].DS[0][1]->map(currentImage, &ubos, 0);
        }

        if (SC.TI[PBR_TECH_INDEX].InstanceCount > 0)
        {
            UniformBufferObjectSimp ubogpbr{};
            // Here mMat contains the real world matrix of the ground
            ubogpbr.mMat = SC.TI[PBR_TECH_INDEX].I[0].Wm;
            ubogpbr.mvpMat = ViewPrj * ubogpbr.mMat;
            ubogpbr.nMat = glm::inverse(glm::transpose(ubogpbr.mMat));
            // Here we set the ground position in local coordinates
            // ubogpbr.worldMat = groundBaseWm;
            SC.TI[PBR_TECH_INDEX].I[0].DS[0][0]->map(currentImage, &guboground, 0);
            SC.TI[PBR_TECH_INDEX].I[0].DS[0][1]->map(currentImage, &ubogpbr, 0);
        }

        UniformBufferObjectSimp uboGem{};
        glm::mat4 spinY = glm::rotate(glm::mat4(1.0f), gemAngle, glm::vec3(0, 1, 0));
        for (int inst_idx = 0; inst_idx < SC.TI[GEM_TECH_INDEX].InstanceCount; ++inst_idx)
        {
            // apply gem rotation animation
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
                      {1.0f, 0.0f, 0.0f, 1.0f}, {0.8f, 0.8f, 0.0f, 1.0f}, {0, 0, 0, 1});
            elapsedT = 0.0f;
            countedFrames = 0;
        }
        txt.updateCommandBuffer();
    }

    void updateUniformBuffer(uint32_t currentImage)
    {
        float deltaT;
        glm::vec3 m, r;
        bool fire;
        getSixAxis(deltaT, m, r, fire);
        glm::mat4 viewMatrix;
        counterGlobal += deltaT;
        // just to avoid problems with zeros
        if (deltaT <= 0.0f) deltaT = 0.0001f;

        // START MENU
        if (gameState == START_MENU && airplaneInitialized)
        {
            if (glfwGetKey(window, GLFW_KEY_ESCAPE))
            {
                glfwSetWindowShouldClose(window, GL_TRUE);
            }

            const float ROTATION_SPEED = 0.4f; // Rotation speed
            const float CAMERA_DISTANCE = 12.0f; // distance from the airplane
            const float CAMERA_HEIGHT = 3.0f;   // Height of the camera above the airplane
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
            glm::mat4 projectionMatrix = glm::perspective(currentFov, Ar, 1.f, 500.f);
            projectionMatrix[1][1] *= -1;
            viewMatrix = glm::lookAt(cameraPos, cameraLookAt, glm::vec3(0.0f, 1.0f, 0.0f));

            glm::mat4 airplaneGlobal =
                                glm::translate(glm::mat4(1.0f), airplanePosition) *
                                glm::mat4_cast(airplaneOrientation) *
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

            txt.print(0.f, 0.f, "PREMI P PER INIZIARE", INSTRUCTIONS_TEXT, "CO", true, false, true, TAL_CENTER, TRH_CENTER, TRV_MIDDLE, {1, 1, 1, 1}, {0, 0, 0, 1}, {0, 0, 0, 0}, 2, 2);
            if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS)
            {
                txt.removeText(INSTRUCTIONS_TEXT);
                gameState = PLAYING;
            }
        }
        // PLAYING
        else if (gameState == PLAYING)
        {
            handleKeyboardInput();

            float targetFov = baseFov;
            if (isBoosting && currentCameraMode == THIRD_PERSON && isEngineOn)
            {
                targetFov += boostFovIncrease;
            }
            float fovInterpSpeed = 5.0f;
            currentFov = glm::mix(currentFov, targetFov, fovInterpSpeed * deltaT);

            // User pressed H to start the game, timer countdown
            if (activateCountdownTimer)
            {
                timerCountdown -= deltaT;

                if (timerCountdown > 0.0f)
                {
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(0) << timerCountdown;
                    txt.print(0.f, -0.f, oss.str(), COUNTDOWN_TEXT, "CO", true, false, true, TAL_CENTER, TRH_CENTER, TRV_MIDDLE,
                              {1, 1, 1, 1}, {0, 0, 0, 1}, {0, 0, 0, 1}, 2, 2);
                }
                else
                {
                    // Countdown finished, position the gems
                    activateCountdownTimer = false;
                    txt.removeText(COUNTDOWN_TEXT);
                    gameTimerActive = true;
                    initGems();
                }
            }

            // Countdown finished, we begin the game
            if (gameTimerActive)
            {
                timer += deltaT;
                if (timer < gameTime && gemsCollected < gemsToCollect)
                {
                    // Formatting time string
                    int minutes = static_cast<int>(gameTime - timer) / 60;
                    int seconds = static_cast<int>(gameTime - timer) % 60;
                    std::ostringstream oss;
                    oss << std::setw(2) << std::setfill('0') << minutes << ":"
                        << std::setw(2) << std::setfill('0') << seconds;

                    txt.print(0.5f, 0.5f, oss.str(), TIMER_TEXT, "CO", true, false, true, TAL_CENTER, TRH_CENTER, TRV_MIDDLE,{1, 1, 1, 1}, {0, 0, 0, 1}, {0, 0, 0, 1}, 1, 1);
                }
                else
                {
                    // Game over condition when timer expires or all gems are collected
                    gameTimerActive = false;
                    txt.removeText(TIMER_TEXT); // Remove the timer text
                    txt.removeText(COLLECTED_GEMS_TEXT); // Remove the collected gems text
                    gameState = GAME_OVER;
                    glm::vec3 cameraOffset = cameraPos - airplanePosition;
                    // Calculate the camera angle for game over view
                    // using atan2 to obtain the angle in radians
                    gameOverCameraAngle = atan2(cameraOffset.x, cameraOffset.z);
                }
            }

            if (airplaneInitialized)
            {
                // gather airplane position and orientation to update it
                const dReal* velocity = dBodyGetLinearVel(odeAirplaneBody);
                const dReal* pos = dBodyGetPosition(odeAirplaneBody);
                glm::vec3 globalVel{ velocity[0], velocity[1], velocity[2] };
                float magSpeed = glm::length(globalVel);

                const float basePitchAccel = 25.f;
                const float baseYawAccel = 20.f;
                const float baseRollAccel = 100.f;

                // setting rotation acceleration properties
                float a_pitch = basePitchAccel * (magSpeed / maxSpeed);
                float a_yaw   = baseYawAccel   * (magSpeed / maxSpeed);
                float a_roll  = baseRollAccel * (magSpeed / maxSpeed);

                // retrieving inertial properties
                const float inertiaScale = 1.f;
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
                const float takeoffSpeed = 15.0f;

                bool keysPressed = false;

                // Allow controls when airplane is on
                if (isEngineOn)
                {
                    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
                    {
                        keysPressed = true;
                        // turn the plane with a torque over roll and yaw axis (local)
                        dBodyAddRelTorque(odeAirplaneBody,
                                          +Izz * a_roll,
                                          0,
                                          0);
                        dBodyAddRelTorque(odeAirplaneBody,
                                          0,
                                          +Iyy * a_yaw,
                                          0);

                        const dReal* q = dBodyGetQuaternion(odeAirplaneBody);
                        glm::quat Q{
                            static_cast<float>(q[0]), static_cast<float>(q[1]), static_cast<float>(q[2]),
                            static_cast<float>(q[3])
                        };

                        // Direction of steering
                        glm::vec3 leftB = glm::normalize(glm::vec3(-0.5, 0, 1));
                        glm::vec3 leftW = Q * leftB;
                        float lateralForceMag = 500.0f;

                        glm::vec3 F = leftW * lateralForceMag;
                        dBodyAddForce(odeAirplaneBody, F.x, F.y, F.z);
                    }
                    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
                    {
                        keysPressed = true;
                        // turn the plane with a torque over roll and yaw axis (local)
                        dBodyAddRelTorque(odeAirplaneBody,
                                          -Izz * a_roll, // body‑x axis roll
                                          0,
                                          0);

                        dBodyAddRelTorque(odeAirplaneBody,
                                          0,
                                          -Iyy * a_yaw,
                                          0);

                        const dReal* q = dBodyGetQuaternion(odeAirplaneBody);
                        glm::quat Q{
                            static_cast<float>(q[0]), static_cast<float>(q[1]), static_cast<float>(q[2]),
                            static_cast<float>(q[3])
                        };

                        // Direction of steering
                        glm::vec3 rightB = glm::normalize(glm::vec3(-0.5, 0, -1));
                        glm::vec3 rightW = Q * rightB;
                        float lateralForceMag = 500.0f;

                        glm::vec3 F = rightW * lateralForceMag;
                        dBodyAddForce(odeAirplaneBody, F.x, F.y, F.z);
                    }

                    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
                    {
                        keysPressed = true;
                        // negative pitch (nose down)
                        dBodyAddRelTorque(odeAirplaneBody,
                                          0,
                                          0, +Ixx * a_pitch);

                        const float rho = 1.225f;
                        const float wingArea = 10.0f;
                        const float CL = 1.0f;
                        // Lift force is always perpendicular to the wing (small trick here is to push down the airplane)
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

                        const float rho = 1.225f;
                        const float wingArea = 10.0f;
                        const float CL = 1.0f;
                        // Lift force is always perpendicular to the wing (here we push up the airplane)
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

                // if no keys are pressed, apply a roll stabilizer
                if (!keysPressed) {
                    const dReal* q = dBodyGetQuaternion(odeAirplaneBody);
                    glm::quat currentOrientation(q[0], q[1], q[2], q[3]);

                    glm::vec3 worldRight = currentOrientation * glm::vec3(0, 0, 1);
                    glm::vec3 projectedRight = glm::normalize(glm::vec3(worldRight.x, 0.0f, worldRight.z));

                    float rollAngle = -worldRight.y;

                    glm::vec3 worldForward = currentOrientation * glm::vec3(-1, 0, 0);
                    glm::vec3 rollTorque = worldForward * rollAngle * 4000.0f;

                    dBodyAddTorque(odeAirplaneBody, rollTorque.x, rollTorque.y, rollTorque.z);
                }
                // When flying turn off gravity for easier flight commands
                if (isEngineOn && magSpeed > takeoffSpeed) {
                    dWorldSetGravity(odeWorld, 0, 0.f, 0);
                }else {
                    dWorldSetGravity(odeWorld, 0, -9.81, 0);
                }

                // accelerate airplane when engine is on
                if (isEngineOn) {
                    const float thrustMagnitude = thrustCoefficient * speed; // tune this
                    dReal fx = thrustMagnitude;
                    dReal fy = 0;
                    dReal fz = 0;
                    dBodyAddRelForce(odeAirplaneBody, -fx, fy, fz);
                }

                // to avoid plane going too fast, we limit the speed
                if (magSpeed > maxSpeed) {
                    dBodySetForce(odeAirplaneBody, 0, 0, 0);
                }

                // check collision
                dSpaceCollide(odeSpace, this, &nearCallback);
                const dReal stepSize = deltaT;
                dWorldStep(odeWorld, stepSize);

                dJointGroupEmpty(contactgroup);

                // gather airplane position and orientation from ODE to update it to the scene
                pos = dBodyGetPosition(odeAirplaneBody);
                const dReal* rot = dBodyGetQuaternion(odeAirplaneBody);
                const dReal* linVel = dBodyGetLinearVel(odeAirplaneBody);
                airplanePosition = glm::vec3(pos[0], pos[1], pos[2]);
                airplaneOrientation = glm::quat(rot[0], rot[1], rot[2], rot[3]);
                airplaneVelocity = glm::vec3(linVel[0], linVel[1], linVel[2]);

                glm::quat finalOrientation = airplaneOrientation * airplaneModelCorrection;

                SC.TI[airplaneTechIdx].I[airplaneInstIdx].Wm =
                    glm::translate(glm::mat4(1.0f), airplanePosition) *
                    glm::mat4_cast(finalOrientation) *
                    glm::scale(glm::mat4(1.0f), airplaneScale);

                glm::mat4 airplaneGlobal =
                                    glm::translate(glm::mat4(1.0f), airplanePosition) *
                                    glm::mat4_cast( airplaneOrientation ) *
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


                // Different views of the camera
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
                    cameraOffset = glm::vec3(-2.5f, 2.0f, 0.0f);
                    glm::vec3 forwardDirection = airplaneOrientation * glm::vec3(-1.0f, 0.0f, 0.0f);
                    //targetCameraLookAt = airplanePosition + (airplaneOrientation * cameraOffset) + forwardDirection;
                    targetCameraLookAt = airplanePosition + airplaneOrientation * cameraOffset + forwardDirection;
                    cameraLookAt = targetCameraLookAt;
                }
                else
                {
                    // THIRD_PERSON (default)
                    cameraOffset = glm::vec3(15.0f, 1.5f, 0.0f);
                }

                targetCameraPos = airplanePosition + (airplaneOrientation * cameraOffset);

                const float CAMERA_SMOOTHING = 10.0f;
                float cameraInterpFactor = 1.0f - glm::exp(-CAMERA_SMOOTHING * deltaT);
                cameraPos = glm::mix(cameraPos, targetCameraPos, cameraInterpFactor);
                cameraLookAt = glm::mix(cameraLookAt, targetCameraLookAt, cameraInterpFactor);

                glm::vec3 targetShakeOffset = glm::vec3(0.0f);
                if (isBoosting && isEngineOn)
                {
                    noiseOffset += deltaT * shakeSpeed;
                    glm::vec3 localShake = glm::vec3(
                        0.0f,
                        noise.GetNoise(noiseOffset, 10.0f) * shakeIntensity,
                        noise.GetNoise(noiseOffset, 20.0f) * shakeIntensity
                    );
                    targetShakeOffset = finalOrientation * localShake;
                    const float thrustMagnitude = thrustCoefficient * speed * 3.0f; // tune this
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

                glm::vec3 cameraUp = glm::normalize(airplaneOrientation * glm::vec3(0.0f, 1.0f, 0.0f));

                //Projection modalities, a perspective and 2 orthographic
                glm::mat4 projectionMatrix;
                switch (currentProjectionMode) {
                case ORTHOGRAPHIC:
                    {
                        float halfWidth  = 5.0f * orthoZoom * Ar;
                        float halfHeight = 5.0f * orthoZoom;
                        projectionMatrix = glm::ortho(-halfWidth, halfWidth, -halfHeight, halfHeight, 1.f, 300.f);
                        break;
                    }
                case ISOMETRIC:
                    {
                        // Same projection as orthographic, but with a different view matrix
                        float halfWidth = orthoZoom * Ar;
                        float halfHeight = orthoZoom;
                        projectionMatrix = glm::ortho(-halfWidth, halfWidth, -halfHeight, halfHeight, -100.f, 300.f);
                        break;
                    }
                case PERSPECTIVE:
                default:
                    {
                        // Here we use the currentFov for perspective projection
                        projectionMatrix = glm::perspective(currentFov, Ar, 1.f, 500.f);
                        break;
                    }
                }

                projectionMatrix[1][1] *= -1; // Vulkan convention, flip Y-axis

                if (currentProjectionMode == ISOMETRIC) {
                    glm::mat4 identity = glm::mat4(1.0f);

                    // Rotate 45° around Y-axis
                    glm::mat4 rotationY = glm::rotate(identity, glm::radians(45.0f), glm::vec3(0.0f, 1.0f, 0.0f));

                    // Rotate -35.26° around X-axis
                    glm::mat4 rotationX = glm::rotate(identity, glm::radians(-35.264f), glm::vec3(1.0f, 0.0f, 0.0f));

                    // Combine rotations and reposition with respect to camera and airplane
                    glm::mat4 isoViewRotation = rotationX * rotationY;
                    glm::vec3 cameraOffset = glm::vec3(0.0f, 0.0f, 40.0f); // distance along Z, adjust as needed
                    glm::vec4 rotatedOffset = isoViewRotation * glm::vec4(cameraOffset, 1.0f);
                    glm::vec3 cameraPos = airplanePosition + glm::vec3(rotatedOffset);

                    // Build view matrix looking at the airplane
                    viewMatrix = glm::lookAt(cameraPos, airplanePosition, glm::vec3(0.0f, 1.0f, 0.0f));
                } else if (currentProjectionMode == ORTHOGRAPHIC) {
                    const float topDistance = 50.0f;

                    // Place the camera directly above the airplane
                    glm::vec3 cameraPos = airplanePosition + glm::vec3(0.0f, topDistance, 0.0f);

                    // Look straight down at the airplane
                    glm::vec3 target = airplanePosition;

                    // 'Up' vector should be something perpendicular to the view direction.
                    // Since we're looking straight down (negative Y), we can use +Z or -Z as up.
                    glm::vec3 up = glm::vec3(-1.0f, 0.0f, 0.0f); // or (0, 0, 1), depending on your coordinate system

                    viewMatrix = glm::lookAt(cameraPos, target, up);
                } else {
                    viewMatrix = glm::lookAt(finalCameraPos, cameraLookAt, cameraUp);
                }

                ViewPrj = projectionMatrix * viewMatrix;

                updateTreePositions();
                // If collision happens, or in water, turn off engine and GAME OVER
                collisionDetected();
                insideWater();

                for (auto &jf : jointFeedbacks) {
                    delete jf;
                }
                jointFeedbacks.clear();
            }
            GameLogic();
        }
        // GAME OVER
        else if (gameState == GAME_OVER)
        {
            if (glfwGetKey(window, GLFW_KEY_ESCAPE))
            {
                glfwSetWindowShouldClose(window, GL_TRUE);
            }

            if (airplaneInitialized)
            {
                // The camera will stay fixed in the last position, wil not follow the airplane
                glm::mat4 projectionMatrix = glm::perspective(currentFov, Ar, 1.f, 500.f);
                projectionMatrix[1][1] *= -1;
                viewMatrix = glm::lookAt(cameraPos, cameraLookAt, glm::vec3(0.0f, 1.0f, 0.0f));
                ViewPrj = projectionMatrix * viewMatrix;

                dSpaceCollide(odeSpace, this, &nearCallback);
                dWorldStep(odeWorld, deltaT);
                dJointGroupEmpty(contactgroup);
                for (auto &jf : jointFeedbacks) {
                    delete jf;
                }
                jointFeedbacks.clear();

                // keep the plane going where it has heading, either with engine on or off
                const dReal* pos = dBodyGetPosition(odeAirplaneBody);
                const dReal* rot = dBodyGetQuaternion(odeAirplaneBody);
                airplanePosition = glm::vec3(pos[0], pos[1], pos[2]);
                airplaneOrientation = glm::quat(rot[0], rot[1], rot[2], rot[3]);
                if (isEngineOn)
                {
                    const float thrustMagnitude = thrustCoefficient * speed; // tune this
                    dReal fx = thrustMagnitude;
                    dReal fy = 0;
                    dReal fz = 0;
                    dBodyAddRelForce(odeAirplaneBody, -fx, fy, fz);
                    dWorldSetGravity(odeWorld, 0, 0.f, 0);
                }
                else
                {
                    dWorldSetGravity(odeWorld, 0, -9.81, 0);
                }

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

            // end game message
            std::ostringstream oss;
            if (inWater) {
                oss << "Sei entrato in acqua!";
            }
            else if (hardImpact) {
                oss << "Ti sei schiantato!";
            }
            else {
                if (gemsCollected == gemsToCollect) {
                    oss << "Hai raccolto tutte le gemme!";
                }
                else {
                    oss << "Hai raccolto solo " << gemsCollected << " gemme su " << gemsToCollect << ". Hai perso!";
                }
            }

            oss << "\nFine del gioco! Premi esc per uscire";

            txt.print(0.f, 0.f, oss.str(), GAME_OVER_TEXT, "SS", false, true, true, TAL_CENTER,
                      TRH_CENTER, TRV_MIDDLE, {1, 1, 1, 1}, {0, 0, 0, 1}, {0, 0, 0, 1}, 1, 1);
        }

        updateUniforms(currentImage, deltaT);

        // Update the OpenAL listener and sources
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


        // move the ground plane to follow the airplane (not in game over)
        glm::mat4 groundXzFollow = glm::translate(
            glm::mat4(1.0f),
            glm::vec3(airplanePosition.x, 0, airplanePosition.z)
        );

        if (groundTechIdx >= 0 && groundInstIdx >= 0 && gameState != GAME_OVER)
        {
            SC.TI[groundTechIdx].I[groundInstIdx].Wm = groundXzFollow * groundBaseWm;
        }

        // update the gems rotation and engine audio
        updateGemsRotation(deltaT);
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
                          {0.0f, 0.0f, 0.0f, 1.0f}, {1.f, 1.f, 1.f, 1.0f}, {0.0f, 0.0f, 0.0f, 1.0f});
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
            auto yRandom = distY(rng) + sampleHeight(xRandom, zRandom);
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
                          {1.0f, 1.0f, 1.0f, 1.0f}, {0.f, 0.f, 0.f, 1.0f}, {0, 0, 0, 1});
    }
    void toggleEngineState() {
        isEngineOn = !isEngineOn;
        if (isEngineOn) targetSpinVelocity = maxSpinVelocity;
        else targetSpinVelocity = minSpinVelocity;
        std::cout << "Engine state: " << (isEngineOn ? "ON" : "OFF") << "\n";
    }
    bool collisionDetected()
    {
        bool collision = false;
        // after dWorldStep(odeWorld,...);
        for (auto fb : jointFeedbacks) {
            // fb->f1 is the force applied to body1 in world coords
            // fb->f2 is the opposite force on body2
            dVector3& F = fb->f1;
            float magnitude = std::sqrt(F[0]*F[0] + F[1]*F[1] + F[2]*F[2]);

            if (magnitude > crashThreshold) {
                if (isEngineOn) toggleEngineState();
                std::cout << "Hard impact! force = " << magnitude << "\n";
                hardImpact = true;
                gameState = GAME_OVER;
            }
            // reset for next frame
            fb->f1[0]=fb->f1[1]=fb->f1[2]=0;
            fb->t1[0]=fb->t1[1]=fb->t1[2]=0;
        }
        return collision;
    }
    bool insideWater() {
        inWater = false;
        if (airplanePosition.y < waterLevel - 1.f) {
            if (isEngineOn) toggleEngineState();
            std::cout << "Inside water\n";
            inWater = true;
            gameState = GAME_OVER;
        }
        return inWater;
    }

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

    float sampleHeight(float x, float z)
    {
        // Sample the terrain height at (x, z) using the noise function
        return noiseGround.GetNoise(x * 0.004f, z * 0.004f) * 0.05f * 500;
    }

    void updateTreePositions()
    {
        float distance = 500.f;
        for (auto & M : treeWorld)
        {
            // extract translation from the matrix
            float x = M[3][0];
            float z = M[3][2];
            bool movedX = false;
            bool movedZ = false;

            if (x < airplanePosition.x - distance) {
                x += distance * 2.f;
                movedX = true;
            }
            if (x > airplanePosition.x + distance) {
                x -= distance * 2.f;
                movedX = true;
            }
            if (z < airplanePosition.z - distance) {
                z += distance * 2.f;
                movedZ = true;
            }
            if (z > airplanePosition.z + distance) {
                z -= distance * 2.f;
                movedZ = true;
            }

            float yNew = sampleHeight(x, z);
            while (yNew < -0.5f)
            {
                if (movedX && !movedZ) {
                    z = airplanePosition.z + treeZ(rng);
                } else if (!movedX && movedZ) {
                    x = airplanePosition.x + treeX(rng);
                } else {
                    x = airplanePosition.x + treeX(rng);
                    z = airplanePosition.z + treeZ(rng);
                }
                yNew = sampleHeight(x, z);
            }
            M = glm::translate(glm::mat4(1.0f), {x, yNew, z});
        }
    }

    void audioCleanUp()
    {
        // Clean up OpenAL resources
        alDeleteSources(1, &audio_source);
        for (unsigned int engineSource : engineSources) {
            alDeleteSources(1, &engineSource);
        }
        for (unsigned int gemSource : gemSources) {
            alDeleteSources(1, &gemSource);
        }
        alDeleteSources(1, &gemCollectedSource);
        alDeleteBuffers(1, &audio_buffer);
        for (unsigned int engineBuffer : engineBuffers) {
            alDeleteBuffers(1, &engineBuffer);
        }
        for (unsigned int gemBuffer : gemBuffers) {
            alDeleteBuffers(1, &gemBuffer);
        }
        alDeleteBuffers(1, &gemCollectedBuffer);
        // Clean up OpenAL context and device
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

private:
    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
    {
        CG_Exam* app = (CG_Exam*)glfwGetWindowUserPointer(window);
        if (app)
        {
            app->handleMouseScroll(yoffset);
        }
    }
};

// This is the main: probably you do not need to touch this!
int main()
{
    CG_Exam app;

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
