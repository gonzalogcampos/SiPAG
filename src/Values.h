#pragma once

namespace values
{
    const bool run = false;

    const int print_priority = 1 ;

    const int cu_BlockSize = 1024;


    /*
    *
    * -------- EMITTER --------
    * 
    */
   float e_Radious = 1.f;
   unsigned int e_ParticlesSecond = 2;
   unsigned int e_MaxParticles = 300;

    /*
    *
    * -------- PARTICLE --------
    * 
    */
   float p_LifeTime = 5.f;            //Life of the particle in seconds
   float p_RLifeTime = 0.2f;         //% of random in life
   float p_Size = 1.f;                //Size of the particle
   float p_SizeEvolution = .05f;      //%per second size improves
   float p_Opacity = 1.f;             //Opacity of the particle
   float p_OpacityEvolution = .05f;   //% per second opacity decays
   float p_InitVelocity = 1.f;        //XYZ init velocity
   float p_RInitVelocity = .1f;      //% of random in XYZ init velocity
   float p_VelocityDecay = .2f;       //% per second velocity decays

   /*
    *
    * -------- WIND --------
    * 
    */

   unsigned int g_Size = 256;           //Grid Size

   float w_ConstantX = 1.f;             //Constant velocity X
   float w_ConstantY = 1.f;             //Constant velocity Y
   float w_ConstantZ = 1.f;             //Constant velocity Z
   //Wind 1
   unsigned int w_Seed = 1;             //Wind 1 initial seed
   unsigned int w_SeedVariation = 0;    //Wind 1 seed valiarion per second
   float w_TurbulenceSize = 1.f;        //Wind 1 turbulence size
   float w_Velocity = 1.f;              //Wind 1 velocity



}
