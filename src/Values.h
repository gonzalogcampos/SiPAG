#pragma once

namespace values
{


            const bool run = false;

            const int print_priority = 2 ;

            const int cu_BlockSize = 1024;

            const int sys_FPS = 60;


            /*
            *
            * -------- EMITTER --------
            * 
            */
        const float e_Radious = 1.f;                   //Emitter radious
        const unsigned int e_ParticlesSecond = 2;      //Not used
        const unsigned int e_EmissionFrec = 1000;      //In 1/1000
        const unsigned int e_MaxParticles = 50000;     //Max Particles

            /*
            *
            * -------- PARTICLE --------
            * 
            */
        const float p_LifeTime = 10.f;             //Life of the particle in seconds
        const float p_RLifeTime = 0.2f;            //% of random in life
        const float p_Size = 1.f;                  //Size of the particle
        const float p_SizeEvolution = .05f;        //%per second size improves
        const float p_Opacity = 1.f;               //Opacity of the particle
        const float p_OpacityEvolution = .05f;     //% per second opacity decays
        const float p_InitVelocity = 1.f;          //XYZ init velocity
        const float p_RInitVelocity = .1f;         //% of random in XYZ init velocity
        const float p_VelocityDecay = .2f;         //% per second velocity decays

        /*
            *
            * -------- WIND --------
            * 
            */

        const unsigned int g_Size = 256;           //Grid Size

        const float w_ConstantX = 1.f;             //Constant velocity X
        const float w_ConstantY = 1.f;             //Constant velocity Y
        const float w_ConstantZ = 1.f;             //Constant velocity Z
        //Wind 1
        const unsigned int w_Seed = 1;             //Wind 1 initial seed
        const unsigned int w_SeedVariation = 0;    //Wind 1 seed valiarion per second
        const float w_TurbulenceSize = 1.f;        //Wind 1 turbulence size
        const float w_Velocity = 1.f;              //Wind 1 velocity




}