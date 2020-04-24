//MIT License
//Copyright (c) 2019 Gonzalo G Campos
#include <iostream>

#include <Simulation.h>

int main(int argv, char **argc)
{
    if(start(argv, argc)!=0)
        return 1;
    
    loop();
    
    close();

    return 0;
}