#include <Simulation_old.h>
#include <Console.h>
#include <CudaControler.h>
#include <Render.h>
#include <GL/glut.h>

Simulation::Simulation()
{
    cudaControler = new CudaControler();
    render = new Render();
}
Simulation::~Simulation()
{
    delete cudaControler;
    delete render;
}

void Simulation::step()
{
    cPrint("Step\n", 1);

    cudaControler->step(true);
    render->draw();
}

void Simulation::start(int argv, char **argc)
{
    cPrint("Start\n", 1);

	glutInit(&argv, argc);
	glutInitWindowSize(720, 720);
  	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  	window = glutCreateWindow("SiPAG");
    //glutDisplayFunc();

    cudaControler->setKernel();
    render->start();

    //glutMainLoop();
}

void Simulation::close()
{
    cPrint("Close\n", 1);

    cudaControler->closeKernel();
    render->close();
    glutDestroyWindow(window);
}