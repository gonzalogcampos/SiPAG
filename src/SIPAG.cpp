//MIT License
//Copyright (c) 2019 Gonzalo G Campos

#include <Simulation.h>
#include <iostream>
#include <CudaControler.h>


int main(){
    Simulation* s = new  Simulation();
    s->start();
    return 0;
}