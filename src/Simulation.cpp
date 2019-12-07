#include <Simulation.h>
#include <Console.h>

void Simulation::run()
{
    Console::print("Run\n", 1);
}

void Simulation::start()
{
    Console::print("Start\n", 1);
}

void Simulation::close()
{
    Console::print("Close\n", 1);
}

void Simulation::pause()
{
    Console::print("Stop\n", 1);
}