//MIT License
//Copyright (c) 2019 Gonzalo G Campos

#include <Simulation.h>
#include <iostream>
#include <CudaControler.h>
#include <Render.h>


int main(){
    Render* render = new Render();
    render->start();

    Simulation* s = new  Simulation();
    s->start();
    s->run();
    s->close();
    render->close();
    delete s;
    delete render;
    
    return 0;
}