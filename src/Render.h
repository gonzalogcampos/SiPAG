#pragma once

#include <iostream>
#include <GL/glew.h>

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

    private:
        Render(){}
        ~Render(){}
        std::string loadShader(char* path);
        GLuint compileShaders();
        void createBuffers();
        void enableAtrib();
        void diableAtrib();




        GLuint rendering_program;
        GLuint vertex_array_object;


        //Buffers
        GLuint bufferX, bufferY, bufferZ, bufferLT, bufferLR;
};