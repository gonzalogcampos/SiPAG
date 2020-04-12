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
        void changeShader();
        void start();
        void draw(float dt);
        void close();
        void resize();
        void pasateBuffers(float* x, float*  y, float* z, float* vx, float* vy, float* vz, float* lt, float* lr);


    private:
        Render(){}
        ~Render(){}
        std::string loadShader(char* path);
        void compileShaders();
        void createBuffers();
        void deleteBuffers();
        void changeSize();
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
        GLuint defaultVP, defaultIncSize, defaultMinSize, defaultRasAlpha, 
                    defaultColor, defaultTimeOpacityGrowing, defaultTimeOpacityDecreasing, 
                    defaultMaxOpacity, dotsVP, dotsColor;

        //Buffers
        GLuint bufferX, bufferY, bufferZ, bufferVX, bufferVY, bufferVZ, bufferLT, bufferLR;

        GLuint texture;
};