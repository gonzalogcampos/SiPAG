//MIT License
//Copyright (c) 2019 Gonzalo G Campos

#pragma once

#include <iostream>
#include <GL/glew.h>

class Render
{
    public:
        //Singleton
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
        void compileShaders();
        void createBuffers();
        void enableAtrib();
        void disableAtrib();
        bool setTexture(char* file);



        //Render programs
        GLuint default_program, dots_program;

        //Buffers
        GLuint bufferX, bufferY, bufferZ, bufferLT, bufferLR;

        GLuint texture;
};