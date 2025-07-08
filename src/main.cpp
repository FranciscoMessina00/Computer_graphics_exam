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

// The uniform buffer object used in this example
struct VertexChar
{
    glm::vec3 pos;
    glm::vec3 norm;
    glm::vec2 UV;
    glm::uvec4 jointIndices;
    glm::vec4 weights;
};

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
};

struct UniformBufferObjectChar
{
    alignas(16) glm::vec4 debug1;
    alignas(16) glm::mat4 mvpMat[65];
    alignas(16) glm::mat4 mMat[65];
    alignas(16) glm::mat4 nMat[65];
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
    DescriptorSetLayout DSLlocalChar, DSLlocalSimp, DSLlocalGem, DSLlocalPBR, DSLglobal, DSLglobalGround, DSLskyBox,
                        DSLground;

    // Vertex formants, Pipelines [Shader couples] and Render passes
    VertexDescriptor VDchar;
    VertexDescriptor VDsimp;
    VertexDescriptor VDgem;
    VertexDescriptor VDskyBox;
    VertexDescriptor VDtan;
    VertexDescriptor VDground;
    RenderPass RP;
    Pipeline Pchar, PsimpObj, PskyBox, P_PBR, Pgem, Pground;
    //*DBG*/Pipeline PDebug;

    // Models, textures and Descriptors (values assigned to the uniforms)
    Scene SC;
    std::vector<VertexDescriptorRef> VDRs;
    std::vector<TechniqueRef> PRs;
    //*DBG*/Model MS;
    //*DBG*/DescriptorSet SSD;

    // To support animation
#define N_ANIMATIONS 5

    AnimBlender AB;
    Animations Anim[N_ANIMATIONS];
    SkeletalAnimation SKA;

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

    glm::vec4 debug1 = glm::vec4(0);

    std::vector<glm::mat4> gemWorlds; // world transforms for each spawned gem
    std::vector<bool> gemsCatched = {false, false, false, false, false, false, false, false, false, false};
    float gemScale = 0.20f; // scale of the gem model
    float catchRadius = 2.5f;
    float timer = 0.f;
    bool timerDone = false;
    float gemAngle = 0.0f;
    float menuCameraAngle = 0.0f;
    enum GameState { START_MENU, PLAYING };
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

    // Variabili per il controllo dell'aereo
    glm::vec3 airplanePosition = {};
    glm::vec3 airplaneVelocity = {};
    glm::quat airplaneOrientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    const glm::quat airplaneModelCorrection = glm::angleAxis(glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    glm::vec3 airplaneScale = glm::vec3(1.0f);
    bool airplaneInitialized = false;
    bool isEngineOn = false;
    bool isAirplaneOnGround = true;
    float visualRollAngle = 0.0f;
    float thrustCoefficient = 200.0f; // Coefficiente di spinta
    float speed = 5.0f;
    float dragCoefficient = 1.0f;
    glm::vec3 currentShakeOffset = glm::vec3(0.0f);

    dWorldID odeWorld;
    dSpaceID odeSpace;
    dBodyID odeAirplaneBody;
    dGeomID odeAirplaneGeom;
    dMass odeAirplaneMass;
    dGeomID odeGroundPlane;
    dJointGroupID contactgroup;

    ALCdevice* device = nullptr;
    ALCcontext* context = nullptr;
    ALuint audio_source = -1;
    ALuint audio_buffer = -1;


    // Indici per il pavimento
    int groundTechIdx = -1;
    int groundInstIdx = -1;
    // fix the ground’s Y (height) to whatever you want—say groundY = 0.0f:
    const float groundY = 0.1f;
    glm::mat4 groundBaseWm = glm::mat4(1.f);
    Model* ground = nullptr;

    // Here you set the main application parameters
    void setWindowParameters()
    {
        // window size, titile and initial background
        windowWidth = 800;
        windowHeight = 600;
        windowTitle = "E09 - Showing animations";
        windowResizable = GLFW_TRUE;

        // Initial aspect ratio
        Ar = 4.0f / 3.0f;
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

    // Here you load and setup all your Vulkan Models and Texutures.
    // Here you also create your Descriptor set layouts and load the shaders for the pipelines
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

        DSLlocalChar.init(this, {
                              // this array contains the binding:
                              // first  element : the binding number
                              // second element : the type of element (buffer or texture)
                              // third  element : the pipeline stage where it will be used
                              {
                                  0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT,
                                  sizeof(UniformBufferObjectChar), 1
                              },
                              {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 1}
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

        DSLground.init(this, {
                           // this array contains the binding:
                           // first  element : the binding number
                           // second element : the type of element (buffer or texture)
                           // third  element : the pipeline stage where it will be used
                           {
                               0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT,
                               sizeof(UniformBufferObjectGround), 1
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

        VDchar.init(this, {
                        {0, sizeof(VertexChar), VK_VERTEX_INPUT_RATE_VERTEX}
                    }, {
                        {
                            0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexChar, pos),
                            sizeof(glm::vec3), POSITION
                        },
                        {
                            0, 1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexChar, norm),
                            sizeof(glm::vec3), NORMAL
                        },
                        {
                            0, 2, VK_FORMAT_R32G32_SFLOAT, offsetof(VertexChar, UV),
                            sizeof(glm::vec2), UV
                        },
                        {
                            0, 3, VK_FORMAT_R32G32B32A32_UINT, offsetof(VertexChar, jointIndices),
                            sizeof(glm::uvec4), JOINTINDEX
                        },
                        {
                            0, 4, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(VertexChar, weights),
                            sizeof(glm::vec4), JOINTWEIGHT
                        }
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

        VDground.init(this, {
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

        VDRs.resize(6);
        VDRs[0].init("VDchar", &VDchar);
        VDRs[1].init("VDsimp", &VDsimp);
        VDRs[2].init("VDskybox", &VDskyBox);
        VDRs[3].init("VDtan", &VDtan);
        VDRs[4].init("VDgem", &VDgem);
        VDRs[5].init("VDground", &VDground);

        // initializes the render passes
        RP.init(this);
        // sets the blue sky
        RP.properties[0].clearValue = {0.0f, 0.9f, 1.0f, 1.0f};


        // Pipelines [Shader couples]
        // The last array, is a vector of pointer to the layouts of the sets that will
        // be used in this pipeline. The first element will be set 0, and so on..
        Pchar.init(this, &VDchar, "shaders/PosNormUvTanWeights.vert.spv", "shaders/CookTorranceForCharacter.frag.spv",
                   {&DSLglobal, &DSLlocalChar});

        PsimpObj.init(this, &VDsimp, "shaders/SimplePosNormUV.vert.spv", "shaders/CookTorrance.frag.spv",
                      {&DSLglobal, &DSLlocalSimp});
        Pgem.init(this, &VDgem, "shaders/Gem.vert.spv", "shaders/CookTorranceGem.frag.spv",
                  {&DSLglobal, &DSLlocalGem});
        Pground.init(this, &VDground, "shaders/GroundVertex.vert.spv", "shaders/CookTorranceGround.frag.spv",
                     {&DSLglobalGround, &DSLground});

        PskyBox.init(this, &VDskyBox, "shaders/SkyBoxShader.vert.spv", "shaders/SkyBoxShader.frag.spv", {&DSLskyBox});
        PskyBox.setCompareOp(VK_COMPARE_OP_LESS_OR_EQUAL);
        PskyBox.setCullMode(VK_CULL_MODE_BACK_BIT);
        PskyBox.setPolygonMode(VK_POLYGON_MODE_FILL);

        P_PBR.init(this, &VDtan, "shaders/SimplePosNormUvTan.vert.spv", "shaders/PBR.frag.spv",
                   {&DSLglobalGround, &DSLlocalPBR});

        PRs.resize(6);
        PRs[0].init("CookTorranceChar", {
                        {
                            &Pchar, {
                                //Pipeline and DSL for the first pass
                                /*DSLglobal*/{},
                                /*DSLlocalChar*/{
                                    /*t0*/{true, 0, {}} // index 0 of the "texture" field in the json file
                                }
                            }
                        }
                    }, /*TotalNtextures*/1, &VDchar);
        PRs[1].init("CookTorranceNoiseSimp", {
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
        PRs[2].init("CookTorranceGem", {
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
                    }, /*TotalNtextures*/2, &VDgem);
        PRs[3].init("SkyBox", {
                        {
                            &PskyBox, {
                                //Pipeline and DSL for the first pass
                                /*DSLskyBox*/{
                                    /*t0*/{true, 0, {}} // index 0 of the "texture" field in the json file
                                }
                            }
                        }
                    }, /*TotalNtextures*/1, &VDskyBox);
        PRs[4].init("GroundShader", {
                        {
                            &Pground, {
                                //Pipeline and DSL for the first pass
                                /*DSLglobal*/{},
                                /*DSLlocalSimp*/{
                                    /*t0*/{true, 0, {}}, // index 0 of the "texture" field in the json file
                                    /*t1*/{true, 1, {}} // index 1 of the "texture" field in the json file
                                }
                            }
                        }
                    }, /*TotalNtextures*/2, &VDground);
        PRs[5].init("PBR", {
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
            odeGroundPlane = dCreatePlane(odeSpace, 0, 1, 0, groundY);
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


        // initializes animations
        for (int ian = 0; ian < N_ANIMATIONS; ian++)
        {
            Anim[ian].init(*SC.As[ian]);
        }
        AB.init({{0, 32, 0.0f, 0}, {0, 16, 0.0f, 1}, {0, 263, 0.0f, 2}, {0, 83, 0.0f, 3}, {0, 16, 0.0f, 4}});
        //AB.init({{0,31,0.0f}});
        SKA.init(Anim, 5, "Armature|mixamo.com|Layer0", 0);

        // initializes the textual output
        txt.init(this, windowWidth, windowHeight);

        // submits the main command buffer
        submitCommandBuffer("main", 0, populateCommandBufferAccess, this);

        // Prepares for showing the FPS count
        txt.print(1.0f, 1.0f, "FPS:", 1, "CO", false, false, true, TAL_RIGHT, TRH_RIGHT, TRV_BOTTOM,
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
        noiseGround.SetFrequency( 2.f );
        noiseGround.SetNoiseType(FastNoise::Perlin);
        noiseGround.SetFractalOctaves(2);
        noiseGround.SetFractalGain(0.8f);

        gemWorlds.resize(10);
        for (auto& M : gemWorlds)
        {
            M =
                glm::translate(glm::mat4(1.0f),
                               glm::vec3(distX(rng), distY(rng), distZ(rng)))
                * glm::scale(glm::mat4(1.0f), glm::vec3(gemScale));
        }

        audioInit();

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
        }

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

        assert(groundTechIdx >= 0 && groundInstIdx >= 0);

        std::cout << "Init done!\n";
    }

    // Here you create your pipelines and Descriptor Sets!
    void pipelinesAndDescriptorSetsInit()
    {
        // creates the render pass
        RP.create();

        // This creates a new pipeline (with the current surface), using its shaders for the provided render pass
        Pchar.create(&RP);
        PsimpObj.create(&RP);
        PskyBox.create(&RP);
        P_PBR.create(&RP);
        Pgem.create(&RP);
        Pground.create(&RP);

        SC.pipelinesAndDescriptorSetsInit();
        txt.pipelinesAndDescriptorSetsInit();
    }

    // Here you destroy your pipelines and Descriptor Sets!
    void pipelinesAndDescriptorSetsCleanup()
    {
        Pchar.cleanup();
        PsimpObj.cleanup();
        PskyBox.cleanup();
        P_PBR.cleanup();
        RP.cleanup();
        Pgem.cleanup();
        Pground.cleanup();

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

        DSLlocalChar.cleanup();
        DSLlocalSimp.cleanup();
        DSLlocalPBR.cleanup();
        DSLskyBox.cleanup();
        DSLglobal.cleanup();
        DSLlocalGem.cleanup();
        DSLground.cleanup();

        Pchar.destroy();
        PsimpObj.destroy();
        PskyBox.destroy();
        P_PBR.destroy();
        Pgem.destroy();
        Pground.destroy();

        RP.destroy();

        SC.localCleanup();
        txt.localCleanup();

        for (int ian = 0; ian < N_ANIMATIONS; ian++)
        {
            Anim[ian].cleanup();
        }

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
        counter++;
        ground->vertices = rawVB_original;
        std::vector<unsigned char>& rawVB = ground->vertices;

        // 3) figure out your stride and the offset of POSITION in it:
        size_t stride    = ground->VD->Bindings[0].stride;
        size_t posOffset = ground->VD->Position.offset;  // byte‑offset in each vertex

        // 1) compute that same world‐space offset
        glm::vec3 worldOffset = airplanePosition;
        // std::cout << "Ground world pos: " << worldOffset.x << ", " << worldOffset.z << "\n";
        glm::vec3 scale;
        scale.x = glm::length(glm::vec3(groundBaseWm[0]));
        scale.y = glm::length(glm::vec3(groundBaseWm[1]));
        scale.z = glm::length(glm::vec3(groundBaseWm[2]));

        // 2) walk vertices
        for (size_t i = 0; i < rawVB.size(); i += stride) {
            glm::vec3* p =
                reinterpret_cast<glm::vec3*>(&rawVB[i + posOffset]);

            // local XZ:
            float lx = p->x * scale.x , lz = p->z * scale.z;

            // // world XZ = local + plane‐translation
            worldOffset.x = std::floor(worldOffset.x * 100.0f) / 100.0f;
            worldOffset.z = std::floor(worldOffset.z * 100.0f) / 100.0f;

            float wx = lx + worldOffset.x;
            float wz = lz + worldOffset.z;

            // float wz = 0.f;
            // now sample noise in world coordinates
            const float NOISE_SCALE  = 0.004f;  // try 0.1…1.0
            const float HEIGHT_SCALE = 0.05f;  // how tall your hills are

            float h = noiseGround.GetNoise(wx * NOISE_SCALE,
                                           wz * NOISE_SCALE)
                    * HEIGHT_SCALE;
            if (i==0 * stride) {
                // std::cout << "Ground local pos: " << lx << ", " << lz << "\n";
                // std::cout << "Ground world pos: " << wx << ", " << wz << "\n";
                std::cout << "Ground height: " << h << "\n";
            }
            p->y = h;
        }
        // 3) reupload:
        ground->updateVertexBuffer();

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

        const int MAX_CONTACTS = 4; // Massimo numero di punti di contatto
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
        //if (handleDebouncedKeyPress(GLFW_KEY_1)) debug1.x = 1.0f - debug1.x;
        if (handleDebouncedKeyPress(GLFW_KEY_1)) currentCameraMode = FIRST_PERSON;
        //if (handleDebouncedKeyPress(GLFW_KEY_2)) debug1.y = 1.0f - debug1.y;
        if (handleDebouncedKeyPress(GLFW_KEY_2)) currentCameraMode = THIRD_PERSON;
        if (handleDebouncedKeyPress(GLFW_KEY_P))
        {
            debug1.z = (float)(((int)debug1.z + 1) % 65);
            std::cout << "Showing bone index: " << debug1.z << "\n";
        }
        if (handleDebouncedKeyPress(GLFW_KEY_O))
        {
            debug1.z = (float)(((int)debug1.z + 64) % 65);
            std::cout << "Showing bone index: " << debug1.z << "\n";
        }
        if (handleDebouncedKeyPress(GLFW_KEY_SPACE))
        {
            static int curAnim = 0;
            curAnim = (curAnim + 1) % 5;
            AB.Start(curAnim, 0.5);
            std::cout << "Playing anim: " << curAnim << "\n";
        }
        if (handleDebouncedKeyPress(GLFW_KEY_H))
        {
            for (auto& M : gemWorlds)
            {
                M = glm::translate(glm::mat4(1.0f), {distX(rng), distY(rng), distZ(rng)}) * glm::scale(
                    glm::mat4(1.0f), glm::vec3(gemScale));
            }
        }

        if (handleDebouncedKeyPress(GLFW_KEY_F))
        {
            isEngineOn = !isEngineOn;
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

        // Calcola il moltiplicatore di velocità target
        float targetSpeedMultiplier = isBoosting ? BOOST_MULTIPLIER : 1.0f;

        // Interpola gradualmente il moltiplicatore di velocità corrente verso il target
        float deltaSpeedMultiplier = targetSpeedMultiplier - currentSpeedMultiplier;
        currentSpeedMultiplier += deltaSpeedMultiplier * ACCELERATION_RATE * deltaT;

        // Applica il moltiplicatore alla velocità di movimento
        float currentForwardSpeed = AIRPLANE_FORWARD_SPEED * currentSpeedMultiplier;

        airplanePosition += localForward * (currentForwardSpeed * deltaT);
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

        // Animazione del personaggio
        const float ANIMATION_SPEED_FACTOR = 0.85f;
        AB.Advance(deltaT * ANIMATION_SPEED_FACTOR);
    }

    // --- Aggiorna tutti gli Uniform Buffer per il frame corrente ---
    void updateUniforms(uint32_t currentImage, float deltaT)
    {
        shift2Dplane();
        const int CHAR_TECH_INDEX = 0, SIMP_TECH_INDEX = 1, GEM_TECH_INDEX = 2, SKY_TECH_INDEX = 3, GROUND_TECH_INDEX =
                      4, PBR_TECH_INDEX = 5;

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
        guboground.eyePosNoSmooth = airplanePosition;

        UniformBufferObjectChar uboc{};
        uboc.debug1 = debug1;
        SKA.Sample(AB);
        std::vector<glm::mat4>* TMsp = SKA.getTransformMatrices();
        glm::mat4 AdaptMat = glm::scale(glm::mat4(1.0f), glm::vec3(0.01f)) * glm::rotate(
            glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
        for (size_t i = 0; i < TMsp->size(); ++i)
        {
            uboc.mMat[i] = AdaptMat * (*TMsp)[i];
            uboc.mvpMat[i] = ViewPrj * uboc.mMat[i];
            uboc.nMat[i] = glm::inverse(glm::transpose(uboc.mMat[i]));
        }
        for (int inst_idx = 0; inst_idx < SC.TI[CHAR_TECH_INDEX].InstanceCount; ++inst_idx)
        {
            SC.TI[CHAR_TECH_INDEX].I[inst_idx].DS[0][0]->map(currentImage, &gubo, 0);
            SC.TI[CHAR_TECH_INDEX].I[inst_idx].DS[0][1]->map(currentImage, &uboc, 0);
        }

        UniformBufferObjectSimp ubos{};
        for (int inst_idx = 0; inst_idx < SC.TI[SIMP_TECH_INDEX].InstanceCount; ++inst_idx)
        {
            ubos.mMat = SC.TI[SIMP_TECH_INDEX].I[inst_idx].Wm;
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

        if (SC.TI[GROUND_TECH_INDEX].InstanceCount > 0)
        {
            UniformBufferObjectGround uboground{};
            uboground.mMat = SC.TI[GROUND_TECH_INDEX].I[0].Wm;
            uboground.mvpMat = ViewPrj * uboground.mMat;
            uboground.nMat = glm::inverse(glm::transpose(uboground.mMat));
            uboground.worldMat = groundBaseWm;
            SC.TI[GROUND_TECH_INDEX].I[0].DS[0][0]->map(currentImage, &guboground, 0);
            SC.TI[GROUND_TECH_INDEX].I[0].DS[0][1]->map(currentImage, &uboground, 0);
        }

        // Aggiornamento HUD (precedentemente in una funzione separata)
        static float elapsedT = 0.0f;
        static int countedFrames = 0;

        countedFrames++;
        elapsedT += deltaT;
        timer += deltaT;
        if (elapsedT > 1.0f)
        {
            float fps = (float)countedFrames / elapsedT;
            std::ostringstream oss;
            oss << "FPS: " << std::fixed << std::setprecision(1) << fps;
            txt.print(1.0f, 1.0f, oss.str(), 1, "CO", false, false, true, TAL_RIGHT, TRH_RIGHT, TRV_BOTTOM,
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
            airplaneOrientation = glm::normalize(glm::quat_cast(rotationPart));
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
            ViewPrj = glm::perspective(currentFov, Ar, 0.1f, 100.0f);
            glm::mat4 projectionMatrix = glm::perspective(currentFov, Ar, 0.1f, 300.f);
            projectionMatrix[1][1] *= -1;
            viewMatrix = glm::lookAt(cameraPos, cameraLookAt, glm::vec3(0.0f, 1.0f, 0.0f));

            ViewPrj = projectionMatrix * viewMatrix;

            updateUniforms(currentImage, deltaT);

            // 4) Stampa il testo “Premi SPAZIO per iniziare”
            txt.print(0.f, 0.f, "PREMI P PER INIZIARE", 2, "CO", true, false, true, TAL_CENTER, TRH_CENTER, TRV_TOP, {1, 1, 1, 1}, {0, 0, 0, 1}, {0, 0, 0, 0}, 2, 2);
            txt.updateCommandBuffer();
            // 5) Controlla SPAZIO
            if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS)
            {
                txt.removeText(2);
                gameState = PLAYING;
            }
        }
        else if (gameState == PLAYING)
        {
            // 2. Gestisci l'input da tastiera (ESC, debug, etc.)
            handleKeyboardInput();

            // 3. Aggiorna lo stato delle animazioni (gemme, personaggio)
            updateState(deltaT);

            bool isBoosting = (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS);
            float targetFov = baseFov;
            if (isBoosting && currentCameraMode == THIRD_PERSON)
            {
                targetFov += boostFovIncrease;
            }
            float fovInterpSpeed = 5.0f;
            currentFov = glm::mix(currentFov, targetFov, fovInterpSpeed * deltaT);


            if (airplaneInitialized)
            {
                const dReal* velocity = dBodyGetLinearVel(odeAirplaneBody);
                const dReal* pos = dBodyGetPosition(odeAirplaneBody);
                glm::vec3 globalVel{ velocity[0], velocity[1], velocity[2] };
                float magSpeed = glm::length(globalVel);
                constexpr float maxSpeed = 20.0f; // m/s, tune to your liking

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

                isAirplaneOnGround = (pos[1] <= groundY + 0.2f);
                bool keysPressed = false;
                const dReal* q = dBodyGetQuaternion(odeAirplaneBody);
                glm::quat Q{ static_cast<float>(q[0]), static_cast<float>(q[1]), static_cast<float>(q[2]), static_cast<float>(q[3]) };

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
                        + Iyy * a_yaw,
                        0);
                        dBodyAddRelTorque(odeAirplaneBody,
                            + Izz * a_roll,
                            0,
                            0);

                        // --- 3) Lateral “skid” force ---
                        //  a) get the body → world rotation quaternion
                        const dReal* q = dBodyGetQuaternion(odeAirplaneBody);
                        glm::quat Q{ static_cast<float>(q[0]), static_cast<float>(q[1]), static_cast<float>(q[2]), static_cast<float>(q[3]) };

                        // b) compute body‑space right axis ( +Z or +Y depending on convention;
                        //    here we assume body +Z is right wing, adjust if yours is different )
                        glm::vec3 leftB = glm::normalize(glm::vec3(-0.5, 0, 1));

                        // c) rotate it into world space:
                        glm::vec3 leftW = Q * leftB;

                        // d) pick a lateral force magnitude (tune this!)
                        float lateralForceMag = 500.0f;  // e.g. 500 N

                        // e) apply that force sideways at the CG
                        glm::vec3 F = leftW * lateralForceMag;
                        dBodyAddForce(odeAirplaneBody, F.x, F.y, F.z);
                    }
                    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
                    {
                        keysPressed = true;
                        // --- 1) Roll torque (roll to the right) ---
                        dBodyAddRelTorque(odeAirplaneBody,
                                          - Izz * a_roll,  // body‑x axis roll
                                          0,
                                          0);

                        // --- 2) Yaw torque (turn nose right) ---
                        dBodyAddRelTorque(odeAirplaneBody,
                                          0,
                                          - Iyy * a_yaw,
                                          0);

                        // --- 3) Lateral “skid” force ---
                        //  a) get the body → world rotation quaternion
                        const dReal* q = dBodyGetQuaternion(odeAirplaneBody);
                        glm::quat Q{ static_cast<float>(q[0]), static_cast<float>(q[1]), static_cast<float>(q[2]), static_cast<float>(q[3]) };

                        // b) compute body‑space right axis ( +Z or +Y depending on convention;
                        //    here we assume body +Z is right wing, adjust if yours is different )
                        glm::vec3 rightB = glm::normalize(glm::vec3(-0.5, 0, -1));

                        // c) rotate it into world space:
                        glm::vec3 rightW = Q * rightB;

                        // d) pick a lateral force magnitude (tune this!)
                        float lateralForceMag = 500.0f;  // e.g. 500 N

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
                                          0, + Ixx * a_pitch);

                        const float rho = 1.225f; // kg/m^3, density of air at sea level
                        const float wingArea = 10.0f; // m^2, tune to your liking
                        const float CL = 1.0f; // lift coefficient, tune to your liking
                        float liftMag = 0.5f * rho * magSpeed * magSpeed * wingArea * CL;
                        const dReal* q = dBodyGetQuaternion(odeAirplaneBody);
                        glm::quat orient(q[0], q[1], q[2], q[3]);
                        glm::vec3 localUp = orient * glm::vec3(0,-1,0);
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
                                          0, - Ixx * a_pitch);

                        const float rho = 1.225f; // kg/m^3, density of air at sea level
                        const float wingArea = 10.0f; // m^2, tune to your liking
                        const float CL = 1.0f; // lift coefficient, tune to your liking
                        float liftMag = 0.5f * rho * magSpeed * magSpeed * wingArea * CL;
                        const dReal* q = dBodyGetQuaternion(odeAirplaneBody);
                        glm::quat orient(q[0], q[1], q[2], q[3]);
                        glm::vec3 localUp = orient * glm::vec3(0,1,0);
                        dBodyAddForce(odeAirplaneBody,
                                      liftMag * localUp.x,
                                      liftMag * localUp.y,
                                      liftMag * localUp.z);
                    }
                }

                if (!keysPressed) {
                    // roll stabilizer (world torque)
                    std::cout << "Roll stabilizer\n";
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

                glm::quat finalOrientation = airplaneOrientation * glm::angleAxis(
                    glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));

                SC.TI[airplaneTechIdx].I[airplaneInstIdx].Wm =
                    glm::translate(glm::mat4(1.0f), airplanePosition) *
                    glm::mat4_cast(finalOrientation) *
                    glm::scale(glm::mat4(1.0f), airplaneScale);

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
                    cameraOffset = glm::vec3(15.0f, 5.0f, 0.0f);
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
                }
                const float SHAKE_INTERP_SPEED = 10.0f;
                float shakeInterpFactor = 1.0f - glm::exp(-SHAKE_INTERP_SPEED * deltaT);
                currentShakeOffset = glm::mix(currentShakeOffset, targetShakeOffset, shakeInterpFactor);
                glm::vec3 finalCameraPos = cameraPos + currentShakeOffset;

                // FIX: L'asse "up" della camera deve seguire il rollio dell'aereo per evitare scatti
                glm::vec3 cameraUp = glm::normalize(airplaneOrientation * glm::vec3(0.0f, 1.0f, 0.0f));
                viewMatrix = glm::lookAt(finalCameraPos, cameraLookAt, cameraUp);
            }
        }

        glm::mat4 projectionMatrix = glm::perspective(currentFov, Ar, 1.f, 500.f);
        projectionMatrix[1][1] *= -1;

        ViewPrj = projectionMatrix * viewMatrix;

        updateUniforms(currentImage, deltaT);

        alListener3f(AL_POSITION, cameraPos.x, cameraPos.y, cameraPos.z);
        alListener3f(AL_VELOCITY, airplaneVelocity.x, airplaneVelocity.y, airplaneVelocity.z);

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
            glm::vec3(airplanePosition.x, groundY, airplanePosition.z)
        );

        if (groundTechIdx >= 0 && groundInstIdx >= 0)
        {
            SC.TI[groundTechIdx].I[groundInstIdx].Wm = groundXzFollow * groundBaseWm;
        }

        GameLogic();
    }


    void GameLogic()
    {
        for (int i = 0; i < gemWorlds.size(); i++)
        {
            // 1) Get gem position (translation column of the mat4)
            auto gemPos = glm::vec3(gemWorlds[i][3]);

            // 2) If within radius and not yet caught...
            if (const float dist = glm::distance(gemPos, airplanePosition); dist < catchRadius && !gemsCatched[i])
            {
                gemsCatched[i] = true;
                gemWorlds[i] = glm::translate(glm::mat4(1.0f), gemPos)
                    * glm::scale(glm::mat4(1.0f), glm::vec3(0.0f));
            }
        }
        if (timer > 120.f && !timerDone)
        {
            std::cout << "Time's up!" << std::endl;
            timerDone = true;
        }
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
        alSource3f(audio_source, AL_POSITION, 0, 0, 0);
        alSource3f(audio_source, AL_VELOCITY, 0, 0, 0);
        alDistanceModel(AL_INVERSE_DISTANCE_CLAMPED);

        loadWavToBuffer("assets/audios/audio_mono.wav");

        alSourcei(audio_source, AL_BUFFER, audio_buffer);
        alSourcei(audio_source, AL_LOOPING, AL_TRUE);
        alSourcef(audio_source, AL_REFERENCE_DISTANCE, 1.0f);
        alSourcef(audio_source, AL_ROLLOFF_FACTOR, 1.0f);
        alSourcef(audio_source, AL_MAX_DISTANCE, 500.0f);
        alSourcei(audio_source, AL_SOURCE_RELATIVE, AL_FALSE);
        alSourcePlay(audio_source);
    }

    void audioCleanUp()
    {
        alDeleteSources(1, &audio_source);
        alDeleteBuffers(1, &audio_buffer);
        alcMakeContextCurrent(nullptr);
        alcDestroyContext(context);
        alcCloseDevice(device);
    }

    void loadWavToBuffer(const char* fileName)
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

        alGenBuffers(1, &audio_buffer);
        alBufferData(audio_buffer, format, pcmData,
                     (ALsizei)(totalSamples * sizeof(int16_t)),
                     wav.sampleRate);
        free(pcmData);
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
