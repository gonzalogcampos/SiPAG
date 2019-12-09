//MIT License
//Copyright (c) 2019 Gonzalo G Campos

#include <GUI.h>
#pragma once
class CudaControler;
class Simulation 
{
    public:
        Simulation();
        ~Simulation();

        void run();
        void start();
        void close();
        void pause();

    private:
        GUI gui;
        CudaControler* cudaControler;
};