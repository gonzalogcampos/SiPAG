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
extern bool GPU_Computing;
extern bool	cu_CopyConstants;
extern bool cu_UpdateRandomKernel;
extern bool paused;


/*
*
* -------- RENDER --------
* 
*/
extern bool r_enable;
extern bool c_autoRotation;
extern float c_autoRotationV;
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
extern float e_Length;                              //Emitter length
extern int e_Type;
extern int e_EmissionFrec;                          //In 1/1000
extern int e_MaxParticles;                          //Max Particles

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
extern float 	timeEv;

extern float    w_Constant[3];
//Wind 1
extern bool     w_1;
extern int 	    w_1n;
extern float 	w_1Amp[3];
extern float 	w_1Size;
extern float 	w_1lacunarity;
extern float 	w_1decay;

extern bool     w_2;
extern int 	    w_2n;
extern float 	w_2Amp[3];
extern float 	w_2Size;
extern float 	w_2lacunarity;
extern float 	w_2decay;