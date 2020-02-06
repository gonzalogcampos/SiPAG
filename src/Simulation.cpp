//MIT License
//Copyright (c) 2019 Gonzalo G Campos


#include <Simulation.h>
#include <Console.h>
#include <CudaControler.h>
#include <Render.h>
#include <OClock.h>
#include <GL/freeglut.h>

//#include <GL/glui.h>

CudaControler *cudaControler;
Render *render;
OClock oclock;
int window;
//GLUI *subwindow;


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
	glutInitWindowSize(300, 720);
    glutInitDisplayMode(GLUT_RGB | GLUT_STENCIL | GLUT_DOUBLE | GLUT_DEPTH);
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);


    std::string title = "SiPAG | " + cudaControler->getDevice();
  	window = glutCreateWindow(title.c_str());

    if(createMenu()!=0)
        return 1;
    
    cudaControler->setKernel();
    oclock.start();
    render->start();

    glutDisplayFunc(step);


    glutMainLoop();


    return 0;
}

void step(void)
{
    cudaControler->step(oclock.step());
    render->draw();

    glutSwapBuffers();
    glutPostRedisplay();
}

void close(void)
{
    cudaControler->closeKernel();
    render->close();
}

int createMenu()
{
    //subwindow = GLUI_Master.create_glui_subwindow (window, GLUI_SUBWINDOW_LEFT);
    return 0;
}