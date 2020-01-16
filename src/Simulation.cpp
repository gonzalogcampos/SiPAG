#include <Simulation.h>
#include <Console.h>
#include <CudaControler.h>
#include <Render.h>
#include <GL/glut.h>

CudaControler *cudaControler;
Render *render;

int window;


void start(int argv, char **argc)
{
    cPrint("Start\n", 1);

    cudaControler = new CudaControler();
    render = new Render();

	glutInit(&argv, argc);
	glutInitWindowSize(720, 720);
    glutInitDisplayMode(GLUT_RGB | GLUT_STENCIL | GLUT_DOUBLE | GLUT_DEPTH);

  	window = glutCreateWindow("SiPAG");
    

    cudaControler->setKernel();
    render->start();

    glutDisplayFunc(step);


    glutMainLoop();
}

void step(void)
{
    cPrint("Step\n", 1);


    cudaControler->step(true);
    render->draw();

    glutSwapBuffers();
    glutPostRedisplay();
}

void close(void)
{
    cPrint("Close\n", 1);

    cudaControler->closeKernel();
    render->close();

    delete cudaControler;
    delete render;

    glutDestroyWindow(window);
}