//MIT License
//Copyright (c) 2019 Gonzalo G Campos


#pragma once

namespace values
{


        const bool run = false;

        const int print_priority = 2 ;

        const int cu_BlockSize = 1024;

        const int sys_FPS = 60;

        const unsigned int render_program = 1; //0=default 1=dots


            /*
            *
            * -------- EMITTER --------
            * 
            */
        const float e_Radious = 1.f;                        //Emitter radious
        const unsigned int e_ParticlesSecond = 2;           //Not used
        const unsigned int e_EmissionFrec = 100;            //In 1/1000
        const unsigned int e_MaxParticles = 2000;           //Max Particles

            /*
            *
            * -------- PARTICLE --------
            * 
            */
        const float p_LifeTime = 2.f;                       //Life of the particle in seconds
        const float p_RLifeTime = 0.2f;                     //% of random in life
        const float p_Size = 1.f;                           //Size of the particle
        const float p_SizeEvolution = .05f;                 //%per second size improves
        const float p_Opacity = 1.f;                        //Opacity of the particle
        const float p_OpacityEvolution = .05f;              //% per second opacity decays
        const float p_InitVelocityX = 0.0f;                 //X init velocity
        const float p_InitVelocityY = 1.0f;                 //Y init velocity
        const float p_InitVelocityZ = 0.0f;                 //Z init velocity
        const float p_RInitVelocityX = 0.5f;                //X random in init velocity
        const float p_RInitVelocityY = 0.5f;                //Y random in init velocity
        const float p_RInitVelocityZ = 0.5f;                //Z random in init velocity
        const float p_VelocityDecay = .3f;                  //% per second velocity decays

        /*
            *
            * -------- WIND --------
            * 
            */

        const unsigned int g_Size = 256;           //Grid Size

        const float w_ConstantX = 0.f;             //Constant velocity X
        const float w_ConstantY = -0.01f;           //Constant velocity Y
        const float w_ConstantZ = 0.f;             //Constant velocity Z
        //Wind 1
        const unsigned int w_Seed = 1;             //Wind 1 initial seed
        const unsigned int w_SeedVariation = 0;    //Wind 1 seed valiarion per second
        const float w_TurbulenceSize = 1.f;        //Wind 1 turbulence size
        const float w_Velocity = 1.f;              //Wind 1 velocity




}