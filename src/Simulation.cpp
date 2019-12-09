#include <Simulation.h>
#include <Console.h>
#include <CudaControler.h>

Simulation::Simulation()
{
    cudaControler = new CudaControler();
}
Simulation::~Simulation()
{
    delete cudaControler;
}
void Simulation::run()
{
}

void Simulation::start()
{
    cPrint("Start\n", 1);
    cudaControler->setKernel();
    for(int i=0; i<2000; i++)
        cudaControler->step(0.1);

    cudaControler->closeKernel();
}

void Simulation::close()
{
    cPrint("Close\n", 1);
}

void Simulation::pause()
{
    cPrint("Stop\n", 1);
}