//MIT License
//Copyright (c) 2019 Gonzalo G Campos


#include <Simulation.h>
#include <Console.h>
#include <CudaControler.h>
#include <Render.h>
#include <OClock.h>
#include <GUI.h>
#include <GL/freeglut.h>
#include <Values.h>



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

	glutInit(&argv, argc);
	glutInitWindowSize(720, 720);
    glutInitDisplayMode(GLUT_RGB | GLUT_STENCIL | GLUT_DOUBLE | GLUT_DEPTH);
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);


    std::string title = "SiPAG | " + cudaControler->getDevice();
  	window = glutCreateWindow(title.c_str());
    glutDisplayFunc(step);
    glutKeyboardFunc(processNormalKeys);

    //Calculate needed memory on device
    int particles_bytes = values::e_MaxParticles * 8 * 4;
    int perlin_bytes = values::g_Size*values::g_Size*values::g_Size * 3 * 2 * 4;
    int bytes = particles_bytes + perlin_bytes;
    cPrint("Memory allocated in device: " + cString(bytes/1048576) + " Mb\n", 2);

    if(createMenu()!=0)
        return 1;
    
    cudaControler->start();
    oclock.start();
    render->start();



    glutMainLoop();


    return 0;
}

void step(void)
{
    double dt = oclock.step();
    cudaControler->step(dt);
    render->draw(dt);

    glutSwapBuffers();
    glutPostRedisplay();
}

void close(void)
{
    cudaControler->close();
    render->close();
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