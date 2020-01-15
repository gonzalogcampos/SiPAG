#include <Simulation.h>
#include <Console.h>
#include <CudaControler.h>
//#include <Clock.h>
Simulation::Simulation()
{
    cudaControler = new CudaControler();
    //clock = new Clock();
}
Simulation::~Simulation()
{
    delete cudaControler;
    //delete clock;
}
void Simulation::run()
{
    int i = 0;
    while(true/*clock->canContinue()*/)
    {
        cudaControler->step(/*clock->getElapsedTime()*/true);
        i++;
        if(i>100)
            return;
    }
}

void Simulation::start()
{
    cPrint("Start\n", 1);
    //clock->start();
    cudaControler->setKernel();
}

void Simulation::close()
{
    cPrint("Close\n", 1);
    cudaControler->closeKernel();
}

void Simulation::pause()
{
    cPrint("Stop\n", 1);
}