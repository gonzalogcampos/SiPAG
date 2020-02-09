//MIT License
//Copyright (c) 2019 Gonzalo G Campos

#include <Console.h>

#include <iostream>

#include <Values.h>


void cPrint(std::string text, int priority)
{
    if(priority<=values::print_priority)
    {
        std::cout<<text;
    }
}


std::string cString( float n )
{
    std::ostringstream ss;
    ss << n;
    return ss.str();
}

std::string cString( double n )
{
    std::ostringstream ss;
    ss << n;
    return ss.str();
}

std::string cString( int n )
{
    std::ostringstream ss;
    ss << n;
    return ss.str();
}

std::string cString( size_t n )
{
    std::ostringstream ss;
    ss << n;
    return ss.str();
}

std::string cString( char* n )
{
    return (char*)n;
}
