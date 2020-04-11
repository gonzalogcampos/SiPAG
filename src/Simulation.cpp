//MIT License
//Copyright (c) 2019 Gonzalo G Campos


#include <Simulation.h>
#include <Console.h>
#include <CudaControler.h>
#include <CPUControler.h>
#include <Render.h>
#include <OClock.h>
#include <GUI.h>
#include <Values.h>

#include <imGUI/imgui.h>
#include <imGUI/imgui_impl_glfw.h>
#include <imGUI/imgui_impl_opengl3.h>

#include <GLFW/glfw3.h>
#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif



bool GPU_Computing = true;
bool CUDA = true;
bool paused = false;



CudaControler *cudaControler;
CPUControler *cpuControler;
Render* render;
OClock oclock;
GLFWwindow* window;



void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}


int start(int argv, char **argc)
{
    cPrint("Start\n", 3);
    cPrint("SiPAG | Cuda & OpenGL Particle simulatior\nBuild: v1.0 2020\nMIT License Copyright (c) Gonzalo G. Campos 2020\n",1);


    cudaControler   = CudaControler::getInstance();
    cpuControler    = CPUControler::getInstance();
    render          = Render::getInstance();


    /*=================================================*/
    /*==================  CUDA INIT ===================*/
    /*=================================================*/
    if(cudaControler->testDevices()!=0)
        CUDA = false;
    /*=================================================*/
    /*=================================================*/



    /*=================================================*/
    /*==================  GLFW INIT ===================*/
    /*=================================================*/
    if (!glfwInit())
        return 1;
    
    std::string title;
    if(CUDA)title = "SiPAG | " + cudaControler->getDevice();
    else title =  "SiPAG | No available CUDA devices";
    window = glfwCreateWindow(1080, 720, title.c_str(), NULL, NULL);
	if (!window) exit(EXIT_FAILURE);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);
    /*=================================================*/
    /*=================================================*/


    /*=================================================*/
    /*=================  IMGUI INIT ===================*/
    /*=================================================*/
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    // Setup Platform/Renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 450");
    /*=================================================*/
    /*=================================================*/
  
    //Start objects
    cudaControler->start();
    cpuControler->start();
    oclock.start();
    render->start();

    return 0;
}

void loop(void)
{
    double dt;
    while (!glfwWindowShouldClose(window)) {
        dt = oclock.step();

        if(CUDA && GPU_Computing && !paused)
            cudaControler->step(dt);
        else if(!paused)
            cpuControler->step(dt);
            
        render->draw(dt);
	    GUIupdate();
        glfwPollEvents();        
	    glfwSwapBuffers(window);
	}
}

void close(void)
{
    cudaControler->close();
    cpuControler->close();
    render->close();
}