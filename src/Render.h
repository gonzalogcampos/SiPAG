//MIT License
//Copyright (c) 2019 Gonzalo G Campos

#pragma once

#include <GL/glew.h>
#include <string>

class Render
{
    public:
        //Singleton
        static Render* getInstance(){
            static Render only_instance;
            return &only_instance;
        }
        enum Direction{UP,DOWN,LEFT,RIGHT,FRONT,BACK};
        void moveCamera(Direction dir);
        void changeShader();
        void start();
        void draw(float dt);

        void close();

    private:
        Render(){}
        ~Render(){}
        std::string loadShader(char* path);
        void compileShaders();
        void createBuffers();
        void enableAtrib();
        void disableAtrib();
        void setTexture(char* file);
        void paseUniforms();

        float time = 0.f;
        float dt = 0.f;
        
        float camH=0.f;
        float camD=1.f;
        float camR=0.f;

        bool defaultShader = true;

        //Render programs
        GLuint default_program, dots_program;
        //Uniforms
        GLuint defaultVP, dotsVP;

        //Buffers
        GLuint bufferX, bufferY, bufferZ, bufferVX, bufferVY, bufferVZ, bufferLT, bufferLR;

        GLuint texture;
};