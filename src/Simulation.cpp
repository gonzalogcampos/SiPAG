#include <Simulation.h>
#include <Console.h>
#include <CudaControler.h>
#include <Render.h>
#include <GL/freeglut.h>
#include <iostream>

CudaControler *cudaControler;
Render *render;

int window, subwindow;


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
    
    cudaControler->setKernel();
    render->start();

    glutDisplayFunc(step);


    glutMainLoop();


    return 0;
}

void step(void)
{
    cudaControler->step(0.1);
    render->draw();

    glutSwapBuffers();
    glutPostRedisplay();
}

void close(void)
{
    cudaControler->closeKernel();
    render->close();
}