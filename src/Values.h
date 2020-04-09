//MIT License
//Copyright (c) 2019 Gonzalo G Campos


#pragma once

/*
*
* -------- SYSTEM --------
* 
*/
extern int cu_BlockSize;
extern int print_priority;
extern bool GPU;


/*
*
* -------- RENDER --------
* 
*/
extern float c_Rotation;
extern float c_Distance;
extern float c_Height;
extern char* r_Texture;
extern float r_BackgroundColor[3];
extern float r_DotsColor[4];
extern float r_WiresColor[4];
extern float r_DefaultColor[3];
extern bool r_RasAlpha;
extern float r_TimeOpacityGrowing;
extern float r_TimeOpacityDecreasing;
extern float r_MaxOpacity;

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
extern float p_incSize;                            //Size of the particle
extern float p_minSize;                            //%per second size improves
extern float p_InitVelocity[3];                    //Init velocity
extern float p_RInitVelocity[3];                   //Random init velocity
extern float p_VelocityDecay;                      //% per second velocity decays

/*
*
* -------- WIND --------
* 
*/

extern int w_VoxelNum;           //Grid Size
extern float w_Constant[3];
//Wind 1
const unsigned int w_Seed = 1;             //Wind 1 initial seed
const unsigned int w_SeedVariation = 0;    //Wind 1 seed valiarion per second
const float w_TurbulenceSize = 1.f;        //Wind 1 turbulence size
const float w_Velocity = 1.f;              //Wind 1 velocity