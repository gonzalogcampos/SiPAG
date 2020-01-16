//MIT License
//Copyright (c) 2019 Gonzalo G Campos

#pragma once
class CudaControler;
class Render;

class Simulation 
{
    public:
        Simulation();
        ~Simulation();

        void start(int argc, char **argv);
        void step();
        void close();

    private:
        int window;
        CudaControler *cudaControler;
        Render *render;
};