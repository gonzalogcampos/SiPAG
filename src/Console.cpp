//MIT License
//Copyright (c) 2019 Gonzalo G Campos

#include <Console.h>
#include <iostream>
#include <Values.h>

void print(std::string text, int priority)
{
    if(priority>=values::print_priority)
    {
        std::cout<<text;
    }
}