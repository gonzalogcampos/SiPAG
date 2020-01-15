//MIT License
//Copyright (c) 2019 Gonzalo G Campos

#include <Console.h>
#include <iostream>

const int print_priority = 1;

void cPrint(std::string text, int priority)
{
    if(priority<=print_priority)
    {
        std::cout<<text;
    }
}