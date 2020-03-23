//MIT License
//Copyright (c) 2019 Gonzalo G Campos

#pragma once

#include <chrono>

class OClock
{
    private:
        std::chrono::duration<double>  elapsedTime; //Elapsed time on a single iteration
        std::chrono::time_point<std::chrono::high_resolution_clock> lastTime; //Elapsed time amounted
        float fps;
    public:
        float getFPS(){return fps;}
        double getElapsedTime();
        void start();
        double step();
};