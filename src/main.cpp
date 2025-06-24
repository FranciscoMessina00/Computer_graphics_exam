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

struct skyBoxUniformBufferObject
{
    alignas(16) glm::mat4 mvpMat;
};


// MAIN !
class E09 : public BaseProject
{
protected:
    enum CameraMode {FIRST_PERSON, THIRD_PERSON};
    enum AirplaneState {STATE_ON_GROUND, STATE_TAXIING, STATE_FLYING};
    AirplaneState airplaneState = STATE_ON_GROUND;
    bool engineOn = false;
    float currentSpeed = 0.0f;

    const float GROUND_LEVEL = 0.5f;       // Altezza del terreno
    const float ACCELERATION = 4.0f;       // Tasso di accelerazione
    const float DECELERATION = 2.0f;       // Tasso di decelerazione (resistenza)
    const float MAX_GROUND_SPEED = 20.0f;  // Velocità massima a terra
    const float MAX_AIR_SPEED = 30.0f;     // Velocità massima in aria
    const float TAKEOFF_SPEED = 18.0f;     // Velocità minima per decollare
    const float SINK_RATE = 3.0f;

    CameraMode currentCameraMode = THIRD_PERSON;
    // Here you list all the Vulkan objects you need:

    // Descriptor Layouts [what will be passed to the shaders]
    DescriptorSetLayout DSLlocalChar, DSLlocalSimp, DSLlocalGem, DSLlocalPBR, DSLglobal, DSLskyBox;

    // Vertex formants, Pipelines [Shader couples] and Render passes
    VertexDescriptor VDchar;
    VertexDescriptor VDsimp;
    VertexDescriptor VDgem;
    VertexDescriptor VDskyBox;
    VertexDescriptor VDtan;
    RenderPass RP;
    Pipeline Pchar, PsimpObj, PskyBox, P_PBR, Pgem;
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
    float Ar; // Aspect ratio

    glm::mat4 ViewPrj;
    glm::mat4 World;
    glm::vec3 Pos = glm::vec3(0, 0, 5);
    glm::vec3 cameraPos;
    glm::vec3 cameraLookAt = glm::vec3(0.0f);
    float Yaw = glm::radians(0.0f);
    float Pitch = glm::radians(0.0f);
    float Roll = glm::radians(0.0f);

    glm::vec4 debug1 = glm::vec4(0);

    std::vector<glm::mat4> gemWorlds; // world transforms for each spawned gem
    std::vector<bool> gemsCatched = {false, false, false, false, false, false, false, false, false, false};
    float catchRadius = 2.f;
    float timer = 0.f;
    bool timerDone = false;
    float gemAngle = 0.0f;

    float minFov = glm::radians(30.0f);
    float maxFov = glm::radians(80.0f);
    float baseFov = glm::radians(45.0f);
    float boostFovIncrease = glm::radians(15.0f);
    float currentFov;
    FastNoise noise;
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
    glm::vec3 airplanePosition;
    glm::vec3 airplaneVelocity;
    glm::quat airplaneOrientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    glm::vec3 airplaneScale = glm::vec3(1.0f);
    bool airplaneInitialized = false;
    float visualRollAngle = 0.0f;
    glm::vec3 currentShakeOffset = glm::vec3(0.0f);

	ALCdevice*  device;
	ALCcontext* context;
	ALuint audio_source;
	ALuint audio_buffer;


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

        VDRs.resize(5);
        VDRs[0].init("VDchar", &VDchar);
        VDRs[1].init("VDsimp", &VDsimp);
        VDRs[2].init("VDskybox", &VDskyBox);
        VDRs[3].init("VDtan", &VDtan);
        VDRs[4].init("VDgem", &VDgem);

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
        Pgem.init(this, &VDgem, "shaders/SimplePosNormUV.vert.spv", "shaders/CookTorrance.frag.spv",
                  {&DSLglobal, &DSLlocalGem});

        PskyBox.init(this, &VDskyBox, "shaders/SkyBoxShader.vert.spv", "shaders/SkyBoxShader.frag.spv", {&DSLskyBox});
        PskyBox.setCompareOp(VK_COMPARE_OP_LESS_OR_EQUAL);
        PskyBox.setCullMode(VK_CULL_MODE_BACK_BIT);
        PskyBox.setPolygonMode(VK_POLYGON_MODE_FILL);

        P_PBR.init(this, &VDtan, "shaders/SimplePosNormUvTan.vert.spv", "shaders/PBR.frag.spv",
                   {&DSLglobal, &DSLlocalPBR});

        PRs.resize(5);
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
        PRs[4].init("PBR", {
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
        DPSZs.uniformBlocksInPool = 10;
        DPSZs.texturesInPool = 10;
        DPSZs.setsInPool = 10;

        std::cout << "\nLoading the scene\n\n";
        if (SC.init(this, /*Npasses*/1, VDRs, PRs, "assets/models/scene.json") != 0)
        {
            std::cout << "ERROR LOADING THE SCENE\n";
            exit(0);
        }

        // Cerca l'indice della tecnica e dell'istanza dell'aereo
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
        distX = std::uniform_real_distribution<float>(-40.0f, 40.0f);
        distY = std::uniform_real_distribution<float>(0.0f, 40.0f);
        distZ = std::uniform_real_distribution<float>(-40.0f, 40.0f);

        noise.SetSeed(1337);
        noise.SetNoiseType(FastNoise::Perlin);

		gemWorlds.resize(10);
		for (auto& M : gemWorlds) {
			M =
				glm::translate(glm::mat4(1.0f),
							 glm::vec3(distX(rng), distY(rng), distZ(rng)))
				* glm::scale(glm::mat4(1.0f), glm::vec3(0.05f));
		}

		audioInit();


		std::cout << "Init done!\n";
	}
	
	// Here you create your pipelines and Descriptor Sets!
	void pipelinesAndDescriptorSetsInit() {
		// creates the render pass
		RP.create();
		
		// This creates a new pipeline (with the current surface), using its shaders for the provided render pass
		Pchar.create(&RP);
		PsimpObj.create(&RP);
		PskyBox.create(&RP);
		P_PBR.create(&RP);
		
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

        SC.pipelinesAndDescriptorSetsCleanup();
        txt.pipelinesAndDescriptorSetsCleanup();
    }

    // Here you destroy all the Models, Texture and Desc. Set Layouts you created!
    // You also have to destroy the pipelines
    void localCleanup()
    {
        DSLlocalChar.cleanup();
        DSLlocalSimp.cleanup();
        DSLlocalPBR.cleanup();
        DSLskyBox.cleanup();
        DSLglobal.cleanup();

        Pchar.destroy();
        PsimpObj.destroy();
        PskyBox.destroy();
        P_PBR.destroy();

        RP.destroy();

		SC.localCleanup();	
		txt.localCleanup();
		
		for(int ian = 0; ian < N_ANIMATIONS; ian++) {
			Anim[ian].cleanup();
		}

		audioCleanUp();
	}
	
	// Here it is the creation of the command buffer:
	// You send to the GPU all the objects you want to draw,
	// with their buffers and textures
	static void populateCommandBufferAccess(VkCommandBuffer commandBuffer, int currentImage, void *Params) {
		// Simple trick to avoid having always 'T->'
		// in che code that populates the command buffer!
		std::cout << "Populating command buffer for " << currentImage << "\n";
		E09 *T = (E09 *)Params;
		T->populateCommandBuffer(commandBuffer, currentImage);
	}
	// This is the real place where the Command Buffer is written
	void populateCommandBuffer(VkCommandBuffer commandBuffer, int currentImage) {
		std::cout << "Let's command buffer!";
		// begin standard pass
		RP.begin(commandBuffer, currentImage);

        SC.populateCommandBuffer(commandBuffer, 0, currentImage);

        RP.end(commandBuffer);
    }

    // =================================================================================
    // Funzioni Helper Modulari
    // =================================================================================
    void handleMouseScroll(double yoffset)
    {
        const float FOV_SENSITIVITY = glm::radians(2.5f);
        baseFov -= yoffset * FOV_SENSITIVITY;
        baseFov = glm::clamp(baseFov, minFov, maxFov);
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
                    glm::mat4(1.0f), glm::vec3(0.025f));
            }
        }
    }

    // --- Prototipo della funzione per gestire l'accelerazione ---
    void handleAirplaneBoost(GLFWwindow* window, float deltaT, float& currentSpeedMultiplier, float AIRPLANE_FORWARD_SPEED, glm::vec3& localForward, glm::vec3& airplanePosition) {
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
        const int CHAR_TECH_INDEX = 0, SIMP_TECH_INDEX = 1, GEM_TECH_INDEX = 2, SKY_TECH_INDEX = 3, PBR_TECH_INDEX = 4;

        const glm::mat4 lightView = glm::rotate(glm::mat4(1), glm::radians(-30.0f), glm::vec3(0.0f, 1.0f, 0.0f)) *
            glm::rotate(glm::mat4(1), glm::radians(-45.0f), glm::vec3(1.0f, 0.0f, 0.0f));
        GlobalUniformBufferObject gubo{};
        gubo.lightDir = glm::vec3(lightView * glm::vec4(0.0f, 0.0f, 1.0f, 1.0f));
        gubo.lightColor = glm::vec4(1.0f);
        gubo.eyePos = cameraPos;

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

        for (int inst_idx = 0; inst_idx < SC.TI[PBR_TECH_INDEX].InstanceCount; ++inst_idx)
        {
            ubos.mMat = SC.TI[PBR_TECH_INDEX].I[inst_idx].Wm;
            ubos.mvpMat = ViewPrj * ubos.mMat;
            ubos.nMat = glm::inverse(glm::transpose(ubos.mMat));
            SC.TI[PBR_TECH_INDEX].I[inst_idx].DS[0][0]->map(currentImage, &gubo, 0);
            SC.TI[PBR_TECH_INDEX].I[inst_idx].DS[0][1]->map(currentImage, &ubos, 0);
        }

        UniformBufferObjectSimp uboGem{};
        glm::mat4 spinY = glm::rotate(glm::mat4(1.0f), gemAngle, glm::vec3(0, 1, 0));
        for (int inst_idx = 0; inst_idx < SC.TI[GEM_TECH_INDEX].InstanceCount; ++inst_idx)
        {
            uboGem.mMat = gemWorlds[inst_idx] * spinY * glm::rotate(glm::mat4(1.0f), glm::radians(90.0f),
                                                                    glm::vec3(1, 0, 0)) * glm::scale(
                glm::mat4(1.0f), glm::vec3(0.5f));
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
        // 1. Ottieni il deltaT e gli input del controller
        float deltaT;
        glm::vec3 m, r;
        bool fire;
        getSixAxis(deltaT, m, r, fire);

        // 2. Gestisci l'input da tastiera (ESC, debug, etc.)
        handleKeyboardInput();

        // 3. Aggiorna lo stato delle animazioni (gemme, personaggio)
        updateState(deltaT);

        // 4. Esegui la logica principale a doppia modalità (Aereo o Personaggio)
        glm::mat4 viewMatrix;

        bool isBoosting = (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS);
        float targetFov = baseFov;
        if (isBoosting && currentCameraMode == THIRD_PERSON)
        {
            targetFov += boostFovIncrease;
        }
        float fovInterpSpeed = 5.0f;
        currentFov = glm::mix(currentFov, targetFov, fovInterpSpeed * deltaT);

        if (airplaneTechIdx != -1 && !airplaneInitialized)
        {
            const glm::mat4& initialWm = SC.TI[airplaneTechIdx].I[airplaneInstIdx].Wm;
            airplanePosition = glm::vec3(initialWm[3]);
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
        }


        if (airplaneInitialized)
        {
            // ========== MODO AEREO ==========
            const float AIRPLANE_FORWARD_SPEED = 7.0f;
            const float PITCH_RATE = glm::radians(45.0f);
            const float YAW_RATE = glm::radians(100.0f);
            const float MAX_VISUAL_ROLL_ANGLE = glm::radians(35.0f); // Inclinazione massima
            const float ROLL_INTERP_SPEED = 5.0f; // Velocità di inclinazione e auto-livellamento
            static float currentSpeedMultiplier = 1.f; // Moltiplicatore di velocità per il boost
            // --- Lettura Input ---
            float pitchInput = 0.0f;
            float yawInput = 0.0f;
            float rollDirection = 0.0f; // Direzione del rollio: -1 (sinistra), 0 (neutro), 1 (destra)

            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) pitchInput -= PITCH_RATE;
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) pitchInput += PITCH_RATE;
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            {
                yawInput += YAW_RATE;
                rollDirection -= 1.f;
            }
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            {
                yawInput -= YAW_RATE;
                rollDirection += 1.f;
            }

            glm::vec3 localForward = airplaneOrientation * glm::vec3(-1.0f, 0.0f, 0.0f);
            glm::vec3 globalUp = glm::vec3(0.0f, 1.0f, 0.0f);
            // Equivalent to cross product of the previous two vectors, without the risk of null vector, because
            // we are using quaternions (as a pro)! :)
            glm::vec3 localRight = airplaneOrientation * globalUp;

            glm::quat pitchRotation = glm::angleAxis(pitchInput * deltaT, localRight);
            glm::quat yawRotation = glm::angleAxis(yawInput * deltaT, globalUp);

            airplaneOrientation = glm::normalize(yawRotation * pitchRotation * airplaneOrientation);
            //airplanePosition += localForward * (AIRPLANE_FORWARD_SPEED * deltaT);
            //airplanePosition += localForward * (AIRPLANE_FORWARD_SPEED * deltaT);
            handleAirplaneBoost(window, deltaT, currentSpeedMultiplier, AIRPLANE_FORWARD_SPEED, localForward, airplanePosition);

            airplaneVelocity = localForward * AIRPLANE_FORWARD_SPEED * currentSpeedMultiplier;
            // --- Logica del Rollio VISIVO (Nuova versione) ---
            // 1. Determina l'angolo di inclinazione target in base all'input
            float targetRollAngle = MAX_VISUAL_ROLL_ANGLE * rollDirection;

            // 2. Interpola dolcemente l'angolo attuale verso il target.
            // Questa singola riga gestisce sia l'inclinazione che l'auto-livellamento.
            float interpFactor = 1.0f - glm::exp(-ROLL_INTERP_SPEED * deltaT);
            visualRollAngle = glm::mix(visualRollAngle, targetRollAngle, interpFactor);

            // --- Aggiornamento Matrice e Camera ---
            glm::quat visualRollQuat = glm::angleAxis(visualRollAngle, glm::vec3(-1.0f, 0.0f, 0.0f));
            glm::quat finalOrientation = airplaneOrientation * visualRollQuat;

            SC.TI[airplaneTechIdx].I[airplaneInstIdx].Wm = glm::translate(glm::mat4(1.0f), airplanePosition) *
                glm::mat4_cast(finalOrientation) * glm::scale(glm::mat4(1.0f), airplaneScale);

            glm::vec3 targetCameraPos;
            glm::vec3 targetCameraLookAt;

            if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
            {
                glm::vec3 leftOffset = glm::vec3(2.0f, 8.0f, 2.0f);
                targetCameraPos = airplanePosition + (airplaneOrientation * leftOffset);
                targetCameraLookAt = airplanePosition;
            }
            else if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
            {
                glm::vec3 rightOffset = glm::vec3(2.0f, -8.0f, 2.0f);
                targetCameraPos = airplanePosition + (airplaneOrientation * rightOffset);
                targetCameraLookAt = airplanePosition;
            }
            else if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS)
            {
                glm::vec3 frontOffset = glm::vec3(-12.0f, 0.0f, 3.0f);
                targetCameraPos = airplanePosition + (airplaneOrientation * frontOffset);
                targetCameraLookAt = airplanePosition;
            }
            else
            {
                if (currentCameraMode == THIRD_PERSON)
                {
                    targetCameraPos = airplanePosition + (airplaneOrientation * glm::vec3(10.0f, 0.0f, 5.5f));
                    targetCameraLookAt = airplanePosition;
                }
                else // FIRST_PERSON
                {
                    glm::vec3 noseOffset = glm::vec3(-2.15f, 0.0f, 1.2f);
                    targetCameraPos = airplanePosition + (finalOrientation * noseOffset);
                    targetCameraLookAt = targetCameraPos + (finalOrientation * glm::vec3(-1.0f, 0.0f, 0.0f));
                }
            }

            const float CAMERA_SMOOTHING = 5.0f;
            float cameraInterpFactor = 1.0f - glm::exp(-CAMERA_SMOOTHING * deltaT);
            cameraPos = glm::mix(cameraPos, targetCameraPos, cameraInterpFactor);
            cameraLookAt = glm::mix(cameraLookAt, targetCameraLookAt, cameraInterpFactor);

            //glm::vec3 finalCameraPos = cameraPos;
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

            }
            const float SHAKE_INTERP_SPEED = 10.0f;
            float shakeInterpFactor = 1.0f - glm::exp(-SHAKE_INTERP_SPEED * deltaT);
            currentShakeOffset = glm::mix(currentShakeOffset, targetShakeOffset, shakeInterpFactor);
            glm::vec3 finalCameraPos = cameraPos + currentShakeOffset;
            glm::vec3 cameraUp = glm::normalize(finalOrientation * glm::vec3(0.0f, 0.0f, 1.f));
            viewMatrix = glm::lookAt(finalCameraPos, cameraLookAt, cameraUp);
        }
        else
        {
            // ========== MODO PERSONAGGIO (Fallback) ==========
            // Questa è la logica che mancava, presa dalla vecchia funzione GameLogic
            const float ROT_SPEED = glm::radians(120.0f);
            const float MOVE_SPEED = 2.0f;

            Yaw -= ROT_SPEED * deltaT * r.y;
            Pitch -= ROT_SPEED * deltaT * r.x;
            Pitch = glm::clamp(Pitch, glm::radians(-8.75f), glm::radians(60.0f));

            glm::vec3 ux = glm::rotate(glm::mat4(1.0f), Yaw, glm::vec3(0, 1, 0)) * glm::vec4(1, 0, 0, 1);
            glm::vec3 uz = glm::rotate(glm::mat4(1.0f), Yaw, glm::vec3(0, 1, 0)) * glm::vec4(0, 0, -1, 1);

            Pos += m.x * ux * MOVE_SPEED * deltaT;
            Pos -= m.z * uz * MOVE_SPEED * deltaT;
            Pos.y += m.y * MOVE_SPEED * deltaT;

            cameraPos = Pos; // In prima persona, la camera è il personaggio
            viewMatrix = glm::rotate(glm::mat4(1.0), -Pitch, glm::vec3(1, 0, 0)) *
                glm::rotate(glm::mat4(1.0), -Yaw, glm::vec3(0, 1, 0)) *
                glm::translate(glm::mat4(1.0), -Pos);
        }

        // 5. Calcola la matrice di proiezione e la View-Projection finale
        glm::mat4 projectionMatrix = glm::perspective(currentFov, Ar, 0.1f, 100.f);
        projectionMatrix[1][1] *= -1; // Adatta a Vulkan

        ViewPrj = projectionMatrix * viewMatrix;

        // 6. Aggiorna tutti gli uniform buffer con le matrici finali
        updateUniforms(currentImage, deltaT);

        alListener3f(AL_POSITION, cameraPos.x, cameraPos.y, cameraPos.z);
        alListener3f(AL_VELOCITY, airplaneVelocity.x, airplaneVelocity.y, airplaneVelocity.z);

        // 1) Compute your “forward” (aka “at”) vector:
        glm::vec3 forward = glm::normalize(cameraLookAt - cameraPos);

        // 2) Choose a “world up” (usually +Y):
        glm::vec3 worldUp(0.0f, 1.0f, 0.0f);

        // 3) Build a true camera “up” vector that’s perpendicular to forward:
        //    this handles roll if you ever introduce it.
        //    (right × forward gives an up that’s perpendicular to both)
        glm::vec3 right   = glm::normalize(glm::cross(forward, worldUp));
        glm::vec3 up      = glm::cross(right, forward);
        float ori[6] = {
            forward.x, forward.y, forward.z,
            up.x,      up.y,      up.z
        };
        alListenerfv(AL_ORIENTATION, ori);
        GameLogic();
    }


    void GameLogic(){
        for (int i = 0; i < gemWorlds.size(); i++)
        {
            // 1) Get gem position (translation column of the mat4)
            auto gemPos = glm::vec3(gemWorlds[i][3]);

            // 2) If within radius and not yet caught...
            if (const float dist = glm::distance(gemPos, airplanePosition); dist < catchRadius && !gemsCatched[i])
            {
                gemsCatched[i] = true;
                gemWorlds[i] = glm::translate(glm::mat4(1.0f), gemPos)
                             * glm::scale    (glm::mat4(1.0f), glm::vec3(0.0f));
            }
        }
        if (timer > 10.f && !timerDone) {
            std::cout << "Time's up!" << std::endl;
            timerDone = true;
        }
    }

    void audioInit() {
		// 1) Open default device & create context
		device  = alcOpenDevice(nullptr);
		if (!device) { std::cerr<<"Failed to open audio device\n"; return ; }
		context = alcCreateContext(device, nullptr);
		if (!context || !alcMakeContextCurrent(context)) {
			std::cerr<<"Failed to create/make context\n";
			if (context) alcDestroyContext(context);
			alcCloseDevice(device);
			return ;
		}

		// 3) Set up the listener (camera) defaults
		//    Position at origin, no velocity
		alListener3f(AL_POSITION, 0.0f, 0.0f, 0.0f);
		alListener3f(AL_VELOCITY, 0.0f, 0.0f, 0.0f);

		//    Orientation: facing down −Z, with +Y as up
		float listenerOri[] = {
			0.0f, 0.0f, -1.0f,   // “forward” vector
			0.0f, 1.0f,  0.0f    // “up” vector
		};
		alListenerfv(AL_ORIENTATION, listenerOri);

		alGenSources(1, &audio_source);
		alSource3f(audio_source, AL_POSITION, 0, 0, 0);
		alSource3f(audio_source, AL_VELOCITY, 0, 0, 0);
		alDistanceModel(AL_INVERSE_DISTANCE_CLAMPED);

		loadWavToBuffer("assets/audios/audio_mono.wav");

		alSourcei(audio_source, AL_BUFFER,  audio_buffer);
		alSourcei(audio_source, AL_LOOPING, AL_TRUE);
		alSourcef(audio_source, AL_REFERENCE_DISTANCE, 1.0f);
		alSourcef(audio_source, AL_ROLLOFF_FACTOR, 1.0f);
		alSourcef(audio_source, AL_MAX_DISTANCE, 500.0f);
		alSourcei(audio_source, AL_SOURCE_RELATIVE, AL_FALSE);
		alSourcePlay(audio_source);
	}
	void audioCleanUp() {

		alDeleteSources(1, &audio_source);
		alDeleteBuffers(1, &audio_buffer);
		alcMakeContextCurrent(nullptr);
		alcDestroyContext(context);
		alcCloseDevice(device);
	}
	void loadWavToBuffer(const char* fileName) {
		// 2) Load WAV into an OpenAL buffer
		drwav wav;
		if (!drwav_init_file(&wav, fileName, nullptr)) {
			std::cerr<<"Could not open audio.wav\n";
			return ;
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