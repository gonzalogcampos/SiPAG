//MIT License
//Copyright (c) 2019 Gonzalo G Campos


#include <Simulation.h>
#include <Console.h>
#include <CudaControler.h>
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



bool GPU = true;



CudaControler *cudaControler;
Render* render;
OClock oclock;
GLFWwindow* window;


int start(int argv, char **argc)
{
    cPrint("Start\n", 3);
    cPrint("SiPAG | Cuda & OpenGL Particle simulatior\nBuild: v1.0 2020\nMIT License Copyright (c) Gonzalo G. Campos 2020\n",1);


    cudaControler   = CudaControler::getInstance();
    render          = Render::getInstance();

    if(cudaControler->testDevices()!=0)
        return 1;

    if (!glfwInit())
        return 1;

    std::string title = "SiPAG | " + cudaControler->getDevice();
    GLFWwindow* window = glfwCreateWindow(1080, 720, title.c_str(), NULL, NULL);
	if (!window) exit(EXIT_FAILURE);

	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);


    /*=================================================*/
    /*===============  IMGUI OPTIONS ==================*/
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
  
    cudaControler->start();
    oclock.start();
    render->start();

    while (!glfwWindowShouldClose(window)) {
        
        step();
        glfwPollEvents();        
	    glfwSwapBuffers(window);
	}


    return 0;
}

void step(void)
{
    double dt = oclock.step();
    cudaControler->step(dt);
    render->draw(dt);
	GUIupdate();
}

void close(void)
{
    cudaControler->close();
    render->close();
}