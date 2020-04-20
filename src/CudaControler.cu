//MIT License
//Copyright (c) 2019 Gonzalo G Campos


//Include headers
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <Windows.h>

//Includes for cuda
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_gl_interop.h>

//Include cuda header with noise functions
#include <CudaNoiseFunc.cuh>

//Include our headers
#include <CudaControler.h>
#include <Values.h>
#include <Console.h>










/*===============================================================*/
/*======================    VALUES    ===========================*/
/*===============================================================*/

//SYSTEM
int 	cu_BlockSize 			= 1024;
bool	cu_CopyConstants 		= true;
bool	cu_UpdateRandomKernel 	= true;

//EMITTER
float 	e_Length 				= .8f;                  //Emitter radious
int 	e_Type 					= 0;                    //Emitter Type
int 	e_EmissionFrec 			= 100;            		//In 1/1000
int 	e_MaxParticles 			= 30000;           		//Max Particles

//PARTICLES
float 	p_LifeTime 				= 3.f;                  //Life of the particle in seconds
float 	p_RLifeTime 			= 0.5f;                 //% of random in life
float 	p_InitVelocity[3]		= {0.0f, 1.0f, 0.0f};	//Z init velocity
float 	p_RInitVelocity[3] 		= {0.5f, 0.5f, 0.5f}; 	//Z init velocity
float 	p_VelocityDecay 		= 1.0f;                 //% per second velocity decays

//WIND
double 	currentTime 			= 0.f;
float 	timeEv 					= 1.f;
float 	w_Constant[3] 			= {0.2f, 0.2f, 0.f};

bool	w_1 					= true;
int 	w_1n 					= 4;
float 	w_1Amp[3] 				= {1.f, 1.f, 1.f};
float 	w_1Size 				= .2f;
float 	w_1lacunarity 			= .25f;
float 	w_1decay 				= 0.2f;

bool	w_2 					= true;
int 	w_2n 					= 3;
float 	w_2Amp[3] 				= {.4f, .4f, .4f};
float 	w_2Size 				= 1.f;
float 	w_2lacunarity 			= 1.f;
float 	w_2decay 				= 1.f;

enum Data
{
	PARTICLE_X,
	PARTICLE_Y,
	PARTICLE_Z,
	PARTICLE_VX,
	PARTICLE_VY,
	PARTICLE_VZ,
	PARTICLE_LT,
	PARTICLE_LR
};

/*===============================================================*/
/*===============================================================*/










/*===============================================================*/
/*===============    CUDA Constant Memory    ====================*/
/*===============================================================*/

//Time
__constant__ float d_timeEv[1];
//Life
__constant__ float d_rLife[1], d_life[1];
//Velocity
__constant__ float d_initVelocity[3], d_rInitVelocity[3];
__constant__ float d_vDecay[1];
//Constants for emitter
__constant__ int d_maxParticles[1];
__constant__ int d_emitterFrec[1];
__constant__ float d_emitterLength[1];
__constant__ int d_emitterType[1];
//Constans for wind
__constant__ float d_constant[3];
__constant__ bool d_1[1];
__constant__ int d_1n[1];
__constant__ float d_1Size[1], d_1lacunarity[1], d_1decay[1], d_1Amp[3];
__constant__ bool d_2[1];
__constant__ int d_2n[1];
__constant__ float d_2Size[1], d_2lacunarity[1], d_2decay[1], d_2Amp[3];

/*===============================================================*/
/*===============================================================*/











/*===============================================================*/
/*===================    CUDA Kernels    ========================*/
/*===============================================================*/

__global__ void setupRandomParticle( curandState * state, unsigned long seed)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx<d_maxParticles[0])
    curand_init ( seed, idx, 0, &state[idx] );
} 


__global__ void kernelParticle(float *x, float *y, float *z,
							float *vx, float *vy, float *vz,
							float *lt, float *lr,
							curandState* state, double dt, double t)
{

	// Get our global thread ID
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if (id < d_maxParticles[0])
	{
		/* curand works like rand - except that it takes a state as a parameter */
		if(lr[id]<0.f)
		{
			//Space to create a particle
			int r = curand(&state[id])%1000;
			if(r<d_emitterFrec[0])
			{
				//Fist of all choose the position
				if(d_emitterType[0]==2)
				{
					r=curand(&state[id])%1000;
					float theta =  (r/1000.f) * 2.0 * 3.14159265359;
					float phi = acos(2.0 *  (r/1000.f) - 1.0) - (3.14159265359/2);
					float sinTheta = sin(theta);
					float cosTheta = cos(theta);
					float sinPhi = sin(phi);
					float cosPhi = cos(phi);
					x[id] = d_emitterLength[0] * cosPhi * cosTheta;
					y[id] = d_emitterLength[0] * cosPhi * sinTheta;
					z[id] = d_emitterLength[0] * sinPhi;
				}
				else if(d_emitterType[0]==1)
				{
					r=curand(&state[id])%1000;
					x[id] = d_emitterLength[0]*((r/1000.f)-.5f);
					y[id] = 0.f;
					z[id] = 0.f;
				}else
				{
					r=curand(&state[id])%1000;
					float theta = (r/1000.f) * 2.0 * 3.14159265359;
					r=curand(&state[id])%1000;
					float phi = acos(2.0 * (r/1000.f) - 1.0) - (3.14159265359/2);
					float sinTheta = sin(theta);
					float cosTheta = cos(theta);
					float sinPhi = sin(phi);
					float cosPhi = cos(phi);
					x[id] = d_emitterLength[0] * cosPhi * cosTheta;
					y[id] = d_emitterLength[0] * cosPhi * sinTheta;
					z[id] = d_emitterLength[0] * sinPhi;
				}
			

				//Then calculate de init velocity
				r=curand(&state[id])%1000;
				vx[id] = d_initVelocity[0] + d_rInitVelocity[0]*2.f*((r/1000.f)-0.5f);

				r=curand(&state[id])%1000;
				vy[id] = d_initVelocity[1] + d_rInitVelocity[1]*2.f*((r/1000.f)-0.5f);

				r=curand(&state[id])%1000;
				vz[id] = d_initVelocity[2] + d_rInitVelocity[2]*2.f*((r/1000.f)-0.5f);


				//And last, calculate the life
				r=curand(&state[id])%1000;
				lt[id] = 0.f;
				lr[id] = d_life[0] + d_rLife[0]*(0.5*(r/1000.f));
			}
		}else
		{
			//Velocity Decay
			vx[id] = vx[id] - vx[id]*d_vDecay[0]*dt;
			vy[id] = vy[id] - vy[id]*d_vDecay[0]*dt;
			vz[id] = vz[id] - vz[id]*d_vDecay[0]*dt;

			//Wind constant velocity
			vx[id] = vx[id] + d_constant[0];
			vy[id] = vy[id] + d_constant[1];
			vz[id] = vz[id] + d_constant[2];

			
			//Wind perlin Big
			if(d_1[0])
			{
				float3 pos = make_float3(x[id], y[id], z[id]);
				float time = t*d_timeEv[0];
				float pbx = d_1Amp[0]*repeaterPerlin(pos, time, d_1Size[0], 2989,   d_1n[0], d_1lacunarity[0], d_1decay[0]);
				float pby = d_1Amp[1]*repeaterPerlin(pos, time, d_1Size[0], 841126, d_1n[0], d_1lacunarity[0], d_1decay[0]);
				float pbz = d_1Amp[2]*repeaterPerlin(pos, time, d_1Size[0], 189277, d_1n[0], d_1lacunarity[0], d_1decay[0]);

				vx[id] = vx[id] + pbx;
				vy[id] = vy[id] + pby;
				vz[id] = vz[id] + pbz;
			}
			
			if(d_2[0])
			{
				float3 pos = make_float3(x[id], y[id], z[id]);
				float time = t*d_timeEv[0];
				float pbx = d_2Amp[0]*repeaterPerlin(pos, time, d_2Size[0], 2989,   d_2n[0], d_2lacunarity[0], d_2decay[0]);
				float pby = d_2Amp[1]*repeaterPerlin(pos, time, d_2Size[0], 841126, d_2n[0], d_2lacunarity[0], d_2decay[0]);
				float pbz = d_2Amp[2]*repeaterPerlin(pos, time, d_2Size[0], 189277, d_2n[0], d_2lacunarity[0], d_2decay[0]);

				vx[id] = vx[id] + pbx;
				vy[id] = vy[id] + pby;
				vz[id] = vz[id] + pbz;
			}

			//Position addition
			x[id] += vx[id]*dt;
			y[id] += vy[id]*dt;
			z[id] += vz[id]*dt;

			//Life set
			lr[id] = lr[id] - dt;
			lt[id] = lt[id] + dt;

		}
	}
}

/*===============================================================*/
/*===============================================================*/










/*===============================================================*/
/*=====================    Functions   ==========================*/
/*===============================================================*/

int CudaControler::testDevices()
{
	int devCount;
		cudaGetDeviceCount(&devCount);
		if(devCount<1)
		{
			cPrint("Error: No devices found\n", 1);
			return 1;
		}
			cPrint("Devices found: " + std::to_string(devCount) + "\nDevice using: ", 1);
			cudaDeviceProp devProp;
			cudaGetDeviceProperties(&devProp, 0);
			cPrint(devProp.name, 1);
			cPrint("\n",1);
			cudaGLSetGLDevice(0);

			cPrint("  Device Properties:\n", 2);
			cPrint("    >Total mem: " + cString(devProp.totalGlobalMem/(1024*1024))+" Mb\n" ,2);
			cPrint("    >Multiprocessor count: " + cString(devProp.multiProcessorCount)+"\n" ,2);
			cPrint("    >Max thread per Multiprocessor: " + cString(devProp.maxThreadsPerMultiProcessor)+"\n", 2);
			cPrint("      >Total: " + cString(devProp.multiProcessorCount*devProp.maxThreadsPerMultiProcessor) + "\n", 2);


		return 0;
}


std::string CudaControler::getDevice()
{
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);
	return devProp.name;
}


void CudaControler::start()
{
	// Size, in bytes, of Particles vector host
	size_t bytes = e_MaxParticles*sizeof(float);

	//Allocate memory for resource vector in host
	h_resource = (float*)malloc(bytes);

	cudaMalloc ( &devStates, e_MaxParticles*sizeof( curandState ) );
	int particles_blockSize = cu_BlockSize;
	int particles_gridSize = (int)ceil((float)e_MaxParticles/particles_blockSize);
	setupRandomParticle<<<particles_gridSize, particles_blockSize>>> ( (curandState*)devStates, rand()%10000);
}


void CudaControler::close()
{
	// Release host memory
	free(h_resource);
	cudaFree(devStates);
}


void CudaControler::step(double dt)
{
	currentTime += dt;

	//Copy constant data to device
	if(cu_CopyConstants)
		copyConstants();

	//Maping the OpenGL buffers for CUDA
	if(r_enable)
	{
		size_t bytes = e_MaxParticles*sizeof(float);
		cudaSafeCall( cudaGraphicsMapResources(1, &resource_x) );
		cudaSafeCall( cudaGraphicsMapResources(1, &resource_y) );
		cudaSafeCall( cudaGraphicsMapResources(1, &resource_z) );	
		cudaSafeCall( cudaGraphicsMapResources(1, &resource_vx) );
		cudaSafeCall( cudaGraphicsMapResources(1, &resource_vy) );
		cudaSafeCall( cudaGraphicsMapResources(1, &resource_vz) );
		cudaSafeCall( cudaGraphicsMapResources(1, &resource_lt) );
		cudaSafeCall( cudaGraphicsMapResources(1, &resource_lr) );
		cudaSafeCall( cudaGraphicsResourceGetMappedPointer((void **)&d_x, &bytes, resource_x) );
		cudaSafeCall( cudaGraphicsResourceGetMappedPointer((void **)&d_y, &bytes, resource_y) );
		cudaSafeCall( cudaGraphicsResourceGetMappedPointer((void **)&d_z, &bytes, resource_z) );	
		cudaSafeCall( cudaGraphicsResourceGetMappedPointer((void **)&d_vx, &bytes, resource_vx) );
		cudaSafeCall( cudaGraphicsResourceGetMappedPointer((void **)&d_vy, &bytes, resource_vy) );
		cudaSafeCall( cudaGraphicsResourceGetMappedPointer((void **)&d_vz, &bytes, resource_vz) );
		cudaSafeCall( cudaGraphicsResourceGetMappedPointer((void **)&d_lt, &bytes, resource_lt) );
		cudaSafeCall( cudaGraphicsResourceGetMappedPointer((void **)&d_lr, &bytes, resource_lr) );
	}

	//Execute Kernel
	// Number of threads in each block
	int particles_blockSize = cu_BlockSize;
	// Number of blocks in grid
	int particles_gridSize = (int)ceil((float)e_MaxParticles/particles_blockSize);

	//Random device States
	if(cu_UpdateRandomKernel)
		setupRandomParticle<<<particles_gridSize, particles_blockSize>>> ( (curandState*)devStates, rand()%10000);

	//Kernel particle
	kernelParticle<<<particles_gridSize, particles_blockSize>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz, d_lt, d_lr, (curandState*)devStates, dt, currentTime);
	
	//Reset the buffers
	if(r_enable)
	{
		cudaSafeCall( cudaGraphicsUnmapResources(1, &resource_x) );
		cudaSafeCall( cudaGraphicsUnmapResources(1, &resource_y) );
		cudaSafeCall( cudaGraphicsUnmapResources(1, &resource_z) );
		cudaSafeCall( cudaGraphicsUnmapResources(1, &resource_vx) );
		cudaSafeCall( cudaGraphicsUnmapResources(1, &resource_vy) );
		cudaSafeCall( cudaGraphicsUnmapResources(1, &resource_vz) );
		cudaSafeCall( cudaGraphicsUnmapResources(1, &resource_lt) );
		cudaSafeCall( cudaGraphicsUnmapResources(1, &resource_lr) );
	}
}


void CudaControler::resize()
{
	close();
	start();
	copyConstants();
}


void CudaControler::conectBuffers(unsigned int bufferX,unsigned int bufferY, unsigned int bufferZ,
									unsigned int bufferVX,unsigned int bufferVY, unsigned int bufferVZ, 
									unsigned int bufferLT, unsigned int bufferLR)
{	
	cudaSafeCall( cudaGraphicsGLRegisterBuffer(&resource_x, (GLuint)bufferX, cudaGraphicsRegisterFlagsNone) );
	cudaSafeCall( cudaGraphicsGLRegisterBuffer(&resource_y, (GLuint)bufferY, cudaGraphicsRegisterFlagsNone) );
	cudaSafeCall( cudaGraphicsGLRegisterBuffer(&resource_z, (GLuint)bufferZ, cudaGraphicsRegisterFlagsNone) );
	cudaSafeCall( cudaGraphicsGLRegisterBuffer(&resource_vx, (GLuint)bufferVX, cudaGraphicsRegisterFlagsNone) );
	cudaSafeCall( cudaGraphicsGLRegisterBuffer(&resource_vy, (GLuint)bufferVY, cudaGraphicsRegisterFlagsNone) );
	cudaSafeCall( cudaGraphicsGLRegisterBuffer(&resource_vz, (GLuint)bufferVZ, cudaGraphicsRegisterFlagsNone) );
	cudaSafeCall( cudaGraphicsGLRegisterBuffer(&resource_lt, (GLuint)bufferLT, cudaGraphicsRegisterFlagsNone) );
	cudaSafeCall( cudaGraphicsGLRegisterBuffer(&resource_lr, (GLuint)bufferLR, cudaGraphicsRegisterFlagsNone) );
}


void CudaControler::cudaSafeCall(cudaError err)
{
  if(cudaSuccess != err)
  {
	std::string m = cudaGetErrorString(err);
	cPrint("Error in CUDA: " + m + "\n", 1);
  }
}


void CudaControler::printData(Data d)
{
	size_t bytes = e_MaxParticles*sizeof(float);

	cPrint("Cuda:   ", 2);

	// Copy array back to host
	switch(d)
	{
		case PARTICLE_X:
			cPrint("X:   ", 2);
			cudaMemcpy( h_resource, d_x, bytes, cudaMemcpyDeviceToHost );
		break;		
		case PARTICLE_Y:
			cPrint("Y:   ", 2);
			cudaMemcpy( h_resource, d_y, bytes, cudaMemcpyDeviceToHost );
		break;
		case PARTICLE_Z:
			cPrint("Z:   ", 2);
			cudaMemcpy( h_resource, d_z, bytes, cudaMemcpyDeviceToHost );
		break;
		case PARTICLE_VX:
			cPrint("VX:   ", 2);
			cudaMemcpy( h_resource, d_vx, bytes, cudaMemcpyDeviceToHost );
		break;
		case PARTICLE_VY:
			cPrint("VY:   ", 2);
			cudaMemcpy( h_resource, d_vy, bytes, cudaMemcpyDeviceToHost );
		break;
		case PARTICLE_VZ:
			cPrint("VZ:   ", 2);
			cudaMemcpy( h_resource, d_vz, bytes, cudaMemcpyDeviceToHost );
		break;
		case PARTICLE_LT:
			cPrint("LT:   ", 2);
			cudaMemcpy( h_resource, d_lt, bytes, cudaMemcpyDeviceToHost );
		break;
		case PARTICLE_LR:
			cPrint("LR:   ", 2);
			cudaMemcpy( h_resource, d_lr, bytes, cudaMemcpyDeviceToHost );
		break;
		default:
			cPrint("X:   ", 2);
			cudaMemcpy( h_resource, d_x, bytes, cudaMemcpyDeviceToHost );
		break;
	}

	for(int i = 0; i<e_MaxParticles; i++)
	{
		cPrint(cString(h_resource[i]) + " ", 2);
	}
	cPrint("\n", 1);
}


void CudaControler::expData(float* x, float*  y, float* z, float* vx, float* vy, float* vz, float* lt, float* lr)
{
	size_t bytes = e_MaxParticles*sizeof(float);

	cudaSafeCall(cudaMemcpy( x, d_x, bytes, cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaMemcpy( y, d_y, bytes, cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaMemcpy( z, d_z, bytes, cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaMemcpy( vx, d_vx, bytes, cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaMemcpy( vy, d_vy, bytes, cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaMemcpy( vz, d_vz, bytes, cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaMemcpy( lt, d_lt, bytes, cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaMemcpy( lr, d_lr, bytes, cudaMemcpyDeviceToHost));


}


void CudaControler::impData(float* x, float*  y, float* z, float* vx, float* vy, float* vz, float* lt, float* lr)
{
	size_t bytes = e_MaxParticles*sizeof(float);

	cudaSafeCall(cudaMemcpy( d_x, x, bytes, cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMemcpy( d_y, y, bytes, cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMemcpy( d_z, z, bytes, cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMemcpy( d_vx, vx, bytes, cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMemcpy( d_vy, vy, bytes, cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMemcpy( d_vz, vz, bytes, cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMemcpy( d_lt, lt, bytes, cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMemcpy( d_lr, lr, bytes, cudaMemcpyHostToDevice));
}


void CudaControler::copyConstants()
{
		cudaSafeCall(cudaMemcpyToSymbol(d_rLife, 			&(p_RLifeTime),			sizeof(float)));
		cudaSafeCall(cudaMemcpyToSymbol(d_life, 			&(p_LifeTime), 			sizeof(float)));
		cudaSafeCall(cudaMemcpyToSymbol(d_vDecay, 			&(p_VelocityDecay), 	sizeof(float)));
		cudaSafeCall(cudaMemcpyToSymbol(d_initVelocity, 	&(p_InitVelocity), 		3*sizeof(float)));
		cudaSafeCall(cudaMemcpyToSymbol(d_rInitVelocity, 	&(p_RInitVelocity), 	3*sizeof(float)));

		cudaSafeCall(cudaMemcpyToSymbol(d_maxParticles, 	&(e_MaxParticles), 		sizeof(int)));
		cudaSafeCall(cudaMemcpyToSymbol(d_emitterFrec, 		&(e_EmissionFrec), 		sizeof(int)));
		cudaSafeCall(cudaMemcpyToSymbol(d_emitterLength, 	&(e_Length), 			sizeof(float)));
		cudaSafeCall(cudaMemcpyToSymbol(d_emitterType, 		&(e_Type), 				sizeof(int)));

		cudaSafeCall(cudaMemcpyToSymbol(d_timeEv, 			&(timeEv), 				sizeof(float)));

		cudaSafeCall(cudaMemcpyToSymbol(d_constant,			&(w_Constant), 			3*sizeof(float)));
		cudaSafeCall(cudaMemcpyToSymbol(d_1,				&(w_1), 				sizeof(bool)));
		cudaSafeCall(cudaMemcpyToSymbol(d_2,				&(w_2), 				sizeof(bool)));
		cudaSafeCall(cudaMemcpyToSymbol(d_1Amp,				&(w_1Amp), 				3*sizeof(float)));
		cudaSafeCall(cudaMemcpyToSymbol(d_2Amp,				&(w_2Amp), 				3*sizeof(float)));
		cudaSafeCall(cudaMemcpyToSymbol(d_1n,				&(w_1n), 				sizeof(int)));
		cudaSafeCall(cudaMemcpyToSymbol(d_2n,				&(w_2n), 				sizeof(int)));
		cudaSafeCall(cudaMemcpyToSymbol(d_1Size,			&(w_1Size), 			sizeof(float)));
		cudaSafeCall(cudaMemcpyToSymbol(d_2Size,			&(w_2Size), 			sizeof(float)));
		cudaSafeCall(cudaMemcpyToSymbol(d_1lacunarity,		&(w_1lacunarity), 		sizeof(float)));
		cudaSafeCall(cudaMemcpyToSymbol(d_2lacunarity,		&(w_2lacunarity), 		sizeof(float)));
		cudaSafeCall(cudaMemcpyToSymbol(d_1decay,			&(w_1decay), 			sizeof(float)));
		cudaSafeCall(cudaMemcpyToSymbol(d_2decay,			&(w_2decay), 			sizeof(float)));
}

/*===============================================================*/
/*===============================================================*/
