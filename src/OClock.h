#pragma once

#include <chrono>

class OClock
{
    private:
        std::chrono::duration<double>  elapsedTime; //Elapsed time on a single iteration
        std::chrono::time_point<std::chrono::high_resolution_clock> lastTime; //Elapsed time amounted
    public:
        float getElapsedTime();
        bool canContinue();
        void start();
        float step();
};