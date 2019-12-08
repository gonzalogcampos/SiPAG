#include <Simulation.h>
#include <Console.h>
#include <CudaControler.h>

void Simulation::run()
{
    print("Run\n", 1);
    CudaControler::doSomething();
    int i = 0;
    while(true)
    {
        i++;
    }
}

void Simulation::start()
{
    print("Start\n", 1);
}

void Simulation::close()
{
    print("Close\n", 1);
}

void Simulation::pause()
{
    print("Stop\n", 1);
}