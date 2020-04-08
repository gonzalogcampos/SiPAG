//MIT License
//Copyright (c) 2019 Gonzalo G Campos


#include <Simulation.h>
#include <Console.h>
#include <CudaControler.h>
#include <Render.h>
#include <OClock.h>
#include <GUI.h>
//#include <GL/freeglut.h>
#include <Values.h>

#include <GLFW/glfw3.h>
#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

CudaControler *cudaControler;
Render *render;
OClock oclock;
GUI gui;
int window;


int start(int argv, char **argc)
{
    cPrint("Start\n", 3);
    cPrint("SiPAG | Cuda & OpenGL Particle simulatior\nBuild: v1.0 2020\nMIT License Copyright (c) Gonzalo G. Campos 2020\n",1);


    cudaControler = CudaControler::getInstance();
    render = Render::getInstance();

    if(cudaControler->testDevices()!=0)
    { 
        return 1;
    }


    if (!glfwInit())
        return 1;

    std::string title = "SiPAG | " + cudaControler->getDevice();
    GLFWwindow* window;
	window = glfwCreateWindow(720, 720, title.c_str(), NULL, NULL);
	if (!window) exit(EXIT_FAILURE);

	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

    //Calculate needed memory on device
    int particles_bytes = values::e_MaxParticles * 8 * 4;
    int perlin_bytes = values::g_Size*values::g_Size*values::g_Size * 3 * 2 * 4;
    int bytes = particles_bytes + perlin_bytes;
    cPrint("Memory allocated in device: " + cString(bytes/1048576) + " Mb\n", 2);


    
    cudaControler->start();
    oclock.start();
    render->start();

    if(createMenu()!=0)
        return 1;

	glfwMakeContextCurrent(window);

    while (!glfwWindowShouldClose(window)) {
        
        step();
        glfwPollEvents();

	    gui.update();
        
        glfwMakeContextCurrent(window);

	    glfwSwapBuffers(window);

	}


    return 0;
}

void step(void)
{
    double dt = oclock.step();
    cudaControler->step(dt);
    render->draw(dt);
}

void close(void)
{
    cudaControler->close();
    render->close();
    gui.close();
}

int createMenu()
{
    gui.init();
    return 0;
}


void processNormalKeys(unsigned char key, int x, int y)
{
    switch(key)
    {
        case 'w':
            Render::getInstance()->moveCamera(Render::Direction::FRONT);
            break;
        case 's':
            Render::getInstance()->moveCamera(Render::Direction::BACK);
            break;
        case 'a':
            Render::getInstance()->moveCamera(Render::Direction::LEFT);
            break;
        case 'd':
            Render::getInstance()->moveCamera(Render::Direction::RIGHT);
            break;
        case 'r':
            Render::getInstance()->moveCamera(Render::Direction::UP);
            break;
        case 'f':
            Render::getInstance()->moveCamera(Render::Direction::DOWN);
            break;
        case 'q':
            Render::getInstance()->changeShader();
            break;
    }


}