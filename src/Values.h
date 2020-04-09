//MIT License
//Copyright (c) 2019 Gonzalo G Campos


#pragma once

/*
*
* -------- CAMERA --------
* 
*/
extern float c_Rotation;
extern float c_Distance;
extern float c_Height;

/*
*
* -------- EMITTER --------
* 
*/
extern float e_Length;                         //Emitter length
extern int e_Type;
extern int e_EmissionFrec;                      //In 1/1000
extern unsigned int e_MaxParticles;             //Max Particles

/*
*
* -------- PARTICLE --------
* 
*/
extern float p_LifeTime;                           //Life of the particle in seconds
extern float p_RLifeTime;                          //% of random in life
extern float p_Size;                               //Size of the particle
extern float p_SizeEvolution;                      //%per second size improves
extern float p_Opacity;                            //Opacity of the particle
extern float p_OpacityEvolution;                   //% per second opacity decays
extern float p_InitVelocityX;                      //X init velocity
extern float p_InitVelocityY;                      //Y init velocity
extern float p_InitVelocityZ;                      //Z init velocity
extern float p_InitVelocity[3];                      //Z init velocity
extern float p_RInitVelocity[3];                      //Z init velocity
extern float p_RInitVelocityX;                     //X random in init velocity
extern float p_RInitVelocityY;                     //Y random in init velocity
extern float p_RInitVelocityZ;                     //Z random in init velocity
extern float p_VelocityDecay;                      //% per second velocity decays



namespace values
{


        const bool run = false;
        const int print_priority = 2 ;
        const int cu_BlockSize = 1024;
        const int sys_FPS = 60;
        const bool sys_Double = false;
        const unsigned int render_program = 0; //0=default 1=dots

        /*
            *
            * -------- WIND --------
            * 
            */

        const unsigned int g_Size = 256;           //Grid Size

        const float w_ConstantX = 0.f;             //Constant velocity X
        const float w_ConstantY = 0.03f;           //Constant velocity Y
        const float w_ConstantZ = 0.f;             //Constant velocity Z
        //Wind 1
        const unsigned int w_Seed = 1;             //Wind 1 initial seed
        const unsigned int w_SeedVariation = 0;    //Wind 1 seed valiarion per second
        const float w_TurbulenceSize = 1.f;        //Wind 1 turbulence size
        const float w_Velocity = 1.f;              //Wind 1 velocity




}