#include <Simulation.h>
#include <Console.h>
#include <CudaControler.h>
#include <Render.h>
#include <GL/glut.h>

CudaControler *cudaControler;
Render *render;

int window, subwindow;


int start(int argv, char **argc)
{
    cPrint("Start\n", 2);
    cPrint("SiPAG | Cuda & OpenGL Particle simulatior\nBuilt version:1.0 2020\nMIT License Copyright (c) Gonzalo G. Campos 2020\n",1);


    cudaControler = CudaControler::getInstance();
    render = Render::getInstance();

    if(cudaControler->testDevices()!=0)
    {
        return 1;
    }

	glutInit(&argv, argc);
	glutInitWindowSize(720, 720);
    glutInitDisplayMode(GLUT_RGB | GLUT_STENCIL | GLUT_DOUBLE | GLUT_DEPTH);

  	window = glutCreateWindow("SiPAG | ");
    
    cudaControler->setKernel();
    cPrint("Kernel seted\n", 3);
    render->start();
    cPrint("Started render\n", 3);

    glutDisplayFunc(step);


    glutMainLoop();

    return 0;
}

void step(void)
{
    cPrint("Step\n", 3);


    cudaControler->step(true);
    render->draw();

    glutSwapBuffers();
    glutPostRedisplay();
}

void close(void)
{
    cPrint("Close\n", 3);

    cudaControler->closeKernel();
    render->close();

    glutDestroyWindow(window);
}