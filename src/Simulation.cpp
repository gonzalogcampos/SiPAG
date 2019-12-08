#include <Simulation.h>
#include <Console.h>
#include <CudaControler.h>

void Simulation::run()
{
}

void Simulation::start()
{
    cPrint("Start\n", 1);
    CudaControler::doSomething();
}

void Simulation::close()
{
    cPrint("Close\n", 1);
}

void Simulation::pause()
{
    cPrint("Stop\n", 1);
}