//MIT License
//Copyright (c) 2019 Gonzalo G Campos

#include <Values.h>
#include <iostream>


class Printer
{
    public:
    static void print(char* text, int priority = 1)
    {
        if(priority>=values::print_priority)
        {
            System.out.ptint(text);
        }
    }

    static void println(char* text, int priority = 1)
    {
        if(priority>=values::print_priority)
        {
            System.out.ptintln(text);
        }
    }
};