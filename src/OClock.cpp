//MIT License
//Copyright (c) 2019 Gonzalo G Campos

#include <OClock.h>
#include <Values.h>
#include <Console.h>

void OClock::start()
{
    lastTime = std::chrono::high_resolution_clock::now();
};

double OClock::getElapsedTime()
{
    return elapsedTime.count();
};

double OClock::step()
{
    std::chrono::time_point<std::chrono::high_resolution_clock> currentTime = std::chrono::high_resolution_clock::now();
    elapsedTime = currentTime - lastTime;
    lastTime = currentTime;
    fps = 1/elapsedTime.count();
    if(elapsedTime.count()>1.f/30)
        return 1.f/30;

    return elapsedTime.count();
}