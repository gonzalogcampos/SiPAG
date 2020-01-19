#pragma once

#include <iostream>
class Render
{
    public:
    
        static Render* getInstance(){
            static Render only_instance;
            return &only_instance;
        }

        void start();
        void draw();
        void close();
        void createBuffers();

    private:
        Render(){}
        ~Render(){}
        std::string loadShader(char* path);
        int compileShaders();
};