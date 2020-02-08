//MIT License
//Copyright (c) 2019 Gonzalo G Campos



#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <Windows.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <curand.h>
#include <curand_kernel.h>
#include <cuda_gl_interop.h>


#include <CudaControler.h>
#include <Values.h>
#include <Console.h>

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

//Constants for particle
	//Life
	__constant__ float d_rLife[1], d_life[1];
	//Velocity
	__constant__ float d_initVelocityX[1], d_initVelocityY[1], d_initVelocityZ[1];
	__constant__ float d_rInitVelocityX[1], d_rInitVelocityY[1], d_rInitVelocityZ[1];
	__constant__ float d_vDecay[1];

//Constants for emitter
__constant__ unsigned int d_maxParticles[1], d_emitterFrec[1];


//Constans for wind
__constant__ float d_constantX[1], d_constantY[1], d_constantZ[1];
__constant__ unsigned int d_gridSize[1], d_perlinSize[1];



__global__ void setupRandomParticle( curandState * state, unsigned long seed)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx<d_maxParticles[0])
    curand_init ( seed, idx, 0, &state[idx] );
} 


__global__ void kernelParticle(float *x, float *y, float *z,
							float *vx, float *vy, float *vz,
							float *lt, float *lr,
							curandState* state, double dt)
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
				x[id] = 0.f;
				y[id] = 0.f;
				z[id] = 0.f;

				r=curand(&state[id])%1000;
				vx[id] = d_initVelocityX[0] + d_rInitVelocityX[0]*2.f*((r/1000.f)-0.5f);

				r=curand(&state[id])%1000;
				vy[id] = d_initVelocityY[0] + d_rInitVelocityY[0]*2.f*((r/1000.f)-0.5f);

				r=curand(&state[id])%1000;
				vz[id] = d_initVelocityZ[0] + d_rInitVelocityZ[0]*2.f*((r/1000.f)-0.5f);

				r=curand(&state[id])%1000;
				lt[id] = 0.f;
				lr[id] = d_life[0] - d_life[0]*d_rLife[0] + 2*d_life[0]*d_rLife[0]*r/1000.f;
			}
		}else
		{
			//Velocity Decay
			vx[id] = vx[id] - vx[id]*d_vDecay[0]*dt;
			vy[id] = vy[id] - vy[id]*d_vDecay[0]*dt;
			vz[id] = vz[id] - vz[id]*d_vDecay[0]*dt;

			//Wind constant velocity
			vx[id] = vx[id] + d_constantX[0];
			vy[id] = vy[id] + d_constantY[0];
			vz[id] = vz[id] + d_constantZ[0];

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

__global__ void setupRandomPerlin(curandState * state, unsigned long seed)
{
	unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int idy = blockIdx.y*blockDim.y+threadIdx.y;
	unsigned int idz = blockIdx.z*blockDim.z+threadIdx.z;

	if(idx>d_gridSize[0] || idy>d_gridSize[0] || idz>d_gridSize[0])
		return;

	unsigned int id = idx + idy*d_gridSize[0] + idz*d_gridSize[0]*d_gridSize[0];

	curand_init ( seed, id, 0, &state[idx]);
}

__global__ void addRandomPerlin(float *d_perlin_x, float *d_perlin_y, float *d_perlin_z, curandState* state)
{
	unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int idy = blockIdx.y*blockDim.y+threadIdx.y;
	unsigned int idz = blockIdx.z*blockDim.z+threadIdx.z;

	if(idx>d_gridSize[0] || idy>d_gridSize[0] || idz>d_gridSize[0])
		return;

	unsigned int id = idx + idy*d_gridSize[0] + idz*d_gridSize[0]*d_gridSize[0];


}

__global__ void kernelPerlin(float *d_perlin_x, float *d_perlin_y, float *d_perlin_z)
{
	unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int idy = blockIdx.y*blockDim.y+threadIdx.y;
	unsigned int idz = blockIdx.z*blockDim.z+threadIdx.z;

	if(idx>d_gridSize[0] || idy>d_gridSize[0] || idz>d_gridSize[0])
		return;

	unsigned int id = idx + idy*d_gridSize[0] + idz*d_gridSize[0]*d_gridSize[0];


}

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
	size_t bytes = values::e_MaxParticles*sizeof(float);
	if(values::sys_Double)
		bytes = values::e_MaxParticles*sizeof(double);

	//Allocate memory for resource vector in host
	h_resource = (float*)malloc(bytes);


	// Size, in bytes, of each 3D perlin matrix in device
	bytes = values::g_Size*values::g_Size*values::g_Size*sizeof(float);
	if(values::sys_Double)
		bytes = values::g_Size*values::g_Size*values::g_Size*sizeof(double);

	//Allocate memory for perlin noise grids in device
	cudaSafeCall(cudaMalloc(&d_perlin_x, bytes));
	cudaSafeCall(cudaMalloc(&d_perlin_y, bytes));
	cudaSafeCall(cudaMalloc(&d_perlin_z, bytes));

	// Number of threads in each block
	particles_blockSize = values::cu_BlockSize;
	perlin_blockSize = values::cu_BlockSize;

	// Number of blocks in grid
	if(values::sys_Double)
	{
		particles_gridSize = (int)ceil((double)values::e_MaxParticles/particles_blockSize);
		perlin_gridSize = (int)ceil((double)(values::g_Size*values::g_Size*values::g_Size)/perlin_blockSize);
	}else
	{
		particles_gridSize = (int)ceil((float)values::e_MaxParticles/particles_blockSize);
		perlin_gridSize = (int)ceil((float)(values::g_Size*values::g_Size*values::g_Size)/perlin_blockSize);	
	}

	//calculatePerlin();

}

void CudaControler::close()
{
		// Release host memory
		free(h_resource);

		cudaSafeCall(cudaFree(d_perlin_x));
		cudaSafeCall(cudaFree(d_perlin_y));
		cudaSafeCall(cudaFree(d_perlin_z));
}

void CudaControler::step(double dt)
{
	//Copy constant data to device
	copyConstants();

	//Maping the OpenGL buffers for CUDA
	size_t bytes = values::e_MaxParticles*sizeof(float);

	cudaSafeCall( cudaGraphicsMapResources(1, &resource_x) );
	cudaSafeCall( cudaGraphicsMapResources(1, &resource_y) );
	cudaSafeCall( cudaGraphicsMapResources(1, &resource_z) );	
	cudaSafeCall( cudaGraphicsMapResources(1, &resource_vx) );
	cudaSafeCall( cudaGraphicsMapResources(1, &resource_vy) );
	cudaSafeCall( cudaGraphicsMapResources(1, &resource_vz) );
	cudaSafeCall( cudaGraphicsMapResources(1, &resource_lt) );
	cudaSafeCall( cudaGraphicsMapResources(1, &resource_lr) );
	cudaSafeCall( cudaGraphicsResourceGetMappedPointer((void **)&d_x_s, &bytes, resource_x) );
	cudaSafeCall( cudaGraphicsResourceGetMappedPointer((void **)&d_y_s, &bytes, resource_y) );
	cudaSafeCall( cudaGraphicsResourceGetMappedPointer((void **)&d_z_s, &bytes, resource_z) );	
	cudaSafeCall( cudaGraphicsResourceGetMappedPointer((void **)&d_vx_s, &bytes, resource_vx) );
	cudaSafeCall( cudaGraphicsResourceGetMappedPointer((void **)&d_vy_s, &bytes, resource_vy) );
	cudaSafeCall( cudaGraphicsResourceGetMappedPointer((void **)&d_vz_s, &bytes, resource_vz) );
	cudaSafeCall( cudaGraphicsResourceGetMappedPointer((void **)&d_lt_s, &bytes, resource_lt) );
	cudaSafeCall( cudaGraphicsResourceGetMappedPointer((void **)&d_lr_s, &bytes, resource_lr) );

	//Execute Kernel
	//Random device States
	curandState* devStates;
	cudaMalloc ( &devStates, values::e_MaxParticles*sizeof( curandState ) );
	
	setupRandomParticle<<<particles_gridSize, particles_blockSize>>> ( devStates, rand()%10000);

	kernelParticle<<<particles_gridSize, particles_blockSize>>>(d_x_s, d_y_s, d_z_s, d_vx_s, d_vy_s, d_vz_s, d_lt_s, d_lr_s, devStates, dt);
	
	cudaFree(devStates);

	//printData(PARTICLE_X);
	//printData(PARTICLE_VX);
	//printData(PARTICLE_Y);
	//printData(PARTICLE_VY);
	//printData(PARTICLE_Z);
	//printData(PARTICLE_VZ);
	//printData(PARTICLE_LT);
	//printData(PARTICLE_LR);


	//Reset the buffers
	cudaGraphicsUnmapResources(1, &resource_x);
	cudaGraphicsUnmapResources(1, &resource_y);
	cudaGraphicsUnmapResources(1, &resource_z);
	cudaGraphicsUnmapResources(1, &resource_vx);
	cudaGraphicsUnmapResources(1, &resource_vy);
	cudaGraphicsUnmapResources(1, &resource_vz);
	cudaGraphicsUnmapResources(1, &resource_lt);
	cudaGraphicsUnmapResources(1, &resource_lr);
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

	/*Unecesary
	// Initialize vectors on host
	for (size_t i = 0; i< sizeof(h_resource)/sizeof(*h_resource); i++)
		h_resource[i] = -1.f;

	// Copy host vectors to device
	size_t bytes = values::e_MaxParticles*sizeof(float);
	cudaMemcpy( d_lr_s, h_resource, bytes, cudaMemcpyHostToDevice);
	*/

}


void CudaControler::cudaSafeCall(cudaError err){
  if(cudaSuccess != err) {
	  std::string m = cudaGetErrorString(err);
	  cPrint("Error in CUDA: " + m + "\n", 1);
  }
}


void CudaControler::printData(Data d)
{
	size_t bytes = values::e_MaxParticles*sizeof(float);
	if(values::sys_Double)
		bytes = values::e_MaxParticles*sizeof(double);

	cPrint("Cuda:   ", 2);

	// Copy array back to host
	switch(d)
	{
		case PARTICLE_X:
			cPrint("X:   ", 2);
			cudaMemcpy( h_resource, d_x_s, bytes, cudaMemcpyDeviceToHost );
		break;		
		case PARTICLE_Y:
			cPrint("Y:   ", 2);
			cudaMemcpy( h_resource, d_y_s, bytes, cudaMemcpyDeviceToHost );
		break;
		case PARTICLE_Z:
			cPrint("Z:   ", 2);
			cudaMemcpy( h_resource, d_z_s, bytes, cudaMemcpyDeviceToHost );
		break;
		case PARTICLE_VX:
			cPrint("VX:   ", 2);
			cudaMemcpy( h_resource, d_vx_s, bytes, cudaMemcpyDeviceToHost );
		break;
		case PARTICLE_VY:
			cPrint("VY:   ", 2);
			cudaMemcpy( h_resource, d_vy_s, bytes, cudaMemcpyDeviceToHost );
		break;
		case PARTICLE_VZ:
			cPrint("VZ:   ", 2);
			cudaMemcpy( h_resource, d_vz_s, bytes, cudaMemcpyDeviceToHost );
		break;
		case PARTICLE_LT:
			cPrint("LT:   ", 2);
			cudaMemcpy( h_resource, d_lt_s, bytes, cudaMemcpyDeviceToHost );
		break;
		case PARTICLE_LR:
			cPrint("LR:   ", 2);
			cudaMemcpy( h_resource, d_lr_s, bytes, cudaMemcpyDeviceToHost );
		break;
		default:
			cPrint("X:   ", 2);
			cudaMemcpy( h_resource, d_x_s, bytes, cudaMemcpyDeviceToHost );
		break;
	}

	for(int i = 0; i<values::e_MaxParticles; i++)
	{
		cPrint(cString(h_resource[i]) + " ", 2);
	}
	cPrint("\n", 1);
}


void CudaControler::copyConstants()
{
	cudaSafeCall(cudaMemcpyToSymbol(d_rLife, 			&(values::p_RLifeTime),			sizeof(const float)));
	cudaSafeCall(cudaMemcpyToSymbol(d_life, 			&(values::p_LifeTime), 			sizeof(const float)));
	cudaSafeCall(cudaMemcpyToSymbol(d_vDecay, 			&(values::p_VelocityDecay), 	sizeof(const float)));
	cudaSafeCall(cudaMemcpyToSymbol(d_initVelocityX, 	&(values::p_InitVelocityX), 	sizeof(const float)));
	cudaSafeCall(cudaMemcpyToSymbol(d_initVelocityY, 	&(values::p_InitVelocityY), 	sizeof(const float)));
	cudaSafeCall(cudaMemcpyToSymbol(d_initVelocityZ, 	&(values::p_InitVelocityZ), 	sizeof(const float)));
	cudaSafeCall(cudaMemcpyToSymbol(d_rInitVelocityX,	&(values::p_RInitVelocityX), 	sizeof(const float)));
	cudaSafeCall(cudaMemcpyToSymbol(d_rInitVelocityY,	&(values::p_RInitVelocityY), 	sizeof(const float)));
	cudaSafeCall(cudaMemcpyToSymbol(d_rInitVelocityZ,	&(values::p_RInitVelocityZ), 	sizeof(const float)));
	cudaSafeCall(cudaMemcpyToSymbol(d_maxParticles, 	&(values::e_MaxParticles), 		sizeof(const unsigned int)));
	cudaSafeCall(cudaMemcpyToSymbol(d_emitterFrec, 		&(values::e_EmissionFrec), 		sizeof(const unsigned int)));


	cudaSafeCall(cudaMemcpyToSymbol(d_constantX,		&(values::w_ConstantX), 		sizeof(const float)));
	cudaSafeCall(cudaMemcpyToSymbol(d_constantY,		&(values::w_ConstantY), 		sizeof(const float)));
	cudaSafeCall(cudaMemcpyToSymbol(d_constantZ,		&(values::w_ConstantZ), 		sizeof(const float)));
	cudaSafeCall(cudaMemcpyToSymbol(d_gridSize,			&(values::g_Size), 				sizeof(const unsigned int)));

}


void CudaControler::calculatePerlin()
{
	curandState* devStates;
	cudaMalloc ( &devStates, values::g_Size*values::g_Size*values::g_Size*sizeof( curandState ) );
	
	setupRandomPerlin<<<perlin_gridSize, perlin_blockSize>>> ( devStates, rand()%10000);

	addRandomPerlin<<<perlin_gridSize, perlin_blockSize>>>(d_perlin_x, d_perlin_y, d_perlin_z, devStates);

	kernelPerlin<<<perlin_gridSize, perlin_blockSize>>>(d_perlin_x, d_perlin_y, d_perlin_z);
	
	cudaFree(devStates);
}