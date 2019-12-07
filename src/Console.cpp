//MIT License
//Copyright (c) 2019 Gonzalo G Campos

#include <Console.h>
#include <iostream>
#include <Values.h>

void print(char* text, int priority = 1)
{
    if(priority>=values::print_priority)
    {
        std::cout<<text;
    }
}