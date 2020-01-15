#pragma once
#include <chrono>
#include "Values.h"

class Clock
{
    private:
        std::chrono::duration<double>  elapsedTime; //The elapsed time on a single iteration
        std::chrono::time_point<std::chrono::high_resolution_clock> lastTime; //The elapsed time amounted
    public:
        Clock();
        float getElapsedTime();
        bool canContinue();
        void start();


};

Clock::Clock()
{
    lastTime = std::chrono::high_resolution_clock::now();
};

void Clock::start()
{
    lastTime = std::chrono::high_resolution_clock::now();
};

float Clock::getElapsedTime()
{
    return elapsedTime.count();
};

bool Clock::canContinue()
{
    std::chrono::time_point<std::chrono::high_resolution_clock> currentTime = std::chrono::high_resolution_clock::now();

    elapsedTime = currentTime - lastTime;

    double minTime = 1.0/values::sys_FPS;
    if(elapsedTime.count() > minTime)
    {
        lastTime = currentTime;
        return true;
    }
    return false;
};