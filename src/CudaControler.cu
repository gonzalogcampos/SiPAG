#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <Windows.h>

// Cabecera necesaria para las rutinas del runtime, es decir, todas
// aquellas que empiezan con cudaXXXXX.
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
__constant__ float d_rLife[1], d_life[1], d_vDecay[1], d_dt[1], d_initVelocity[1], d_rInitVelocity[1];

//Constants for emitter
__constant__ unsigned int d_maxParticles[1], d_emitterFrec[1];


//Constans for wind
__constant__ float d_constantX[1], d_constantY[1], d_constantZ[1];



__global__ void setup_kernel ( curandState * state, unsigned long seed, unsigned int maxParticles)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx<maxParticles)
    curand_init ( seed, idx, 0, &state[idx] );
} 


__global__ void kernel(float *x, float *y, float *z,
							float *vx, float *vy, float *vz,
							float *lt, float *lr,
							curandState* state, float dt)
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

				r=curand(&state[id])%100;

				vx[id] = d_initVelocity[0] - d_initVelocity[0]*d_rInitVelocity[0] + 2*d_initVelocity[0]*d_rInitVelocity[0]*r/100;

				r=curand(&state[id])%100;

				vy[id] = d_initVelocity[0] - d_initVelocity[0]*d_rInitVelocity[0] + 2*d_initVelocity[0]*d_rInitVelocity[0]*r/100;

				r=curand(&state[id])%100;

				vz[id] = d_initVelocity[0] - d_initVelocity[0]*d_rInitVelocity[0] + 2*d_initVelocity[0]*d_rInitVelocity[0]*r/100;

				r=curand(&state[id])%100;

				lt[id] = 0.f;
				lr[id] = d_life[0] - d_life[0]*d_rLife[0] + 2*d_life[0]*d_rLife[0]*r/100;
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

		return 0;
}

std::string CudaControler::getDevice()
{
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);
	return devProp.name;
}


void CudaControler::setKernel()
{

	// Size, in bytes, of each vector
	size_t bytes = values::e_MaxParticles*sizeof(float);

	//Allocate memory for resource vector in host
	h_resource = (float*)malloc(bytes);
	
	//Allocate memory for each vector in device
	cudaMalloc(&d_vx, bytes);
	cudaMalloc(&d_vy, bytes);
	cudaMalloc(&d_vz, bytes);

	// Number of threads in each block
	blockSize = values::cu_BlockSize;

	// Number of blocks in grid
	gridSize = (int)ceil((float)values::e_MaxParticles/blockSize);
}

void CudaControler::closeKernel()
{
		// Release device memory
		cudaFree(d_vx);
		cudaFree(d_vy);
		cudaFree(d_vz);

		// Release host memory
		free(h_resource);
}

void CudaControler::step(float dt)
{
	//Copy constant data to device
	copyConstants();

	//Maping the OpenGL buffers for CUDA
	size_t bytes = values::e_MaxParticles*sizeof(float);
	cudaSafeCall( cudaGraphicsMapResources(1, &resource_x) );
	cudaSafeCall( cudaGraphicsMapResources(1, &resource_y) );
	cudaSafeCall( cudaGraphicsMapResources(1, &resource_z) );
	cudaSafeCall( cudaGraphicsMapResources(1, &resource_lt) );
	cudaSafeCall( cudaGraphicsMapResources(1, &resource_lr) );
	cudaSafeCall( cudaGraphicsResourceGetMappedPointer((void **)&d_x, &bytes, resource_x) );
	cudaSafeCall( cudaGraphicsResourceGetMappedPointer((void **)&d_y, &bytes, resource_y) );
	cudaSafeCall( cudaGraphicsResourceGetMappedPointer((void **)&d_z, &bytes, resource_z) );
	cudaSafeCall( cudaGraphicsResourceGetMappedPointer((void **)&d_lt, &bytes, resource_lt) );
	cudaSafeCall( cudaGraphicsResourceGetMappedPointer((void **)&d_lr, &bytes, resource_lr) );

	//Execute Kernel
	//Random device States
	curandState* devStates;
	cudaMalloc ( &devStates, values::e_MaxParticles*sizeof( curandState ) );
	
	setup_kernel<<<gridSize, blockSize>>> ( devStates, rand()%10000, values::e_MaxParticles);

	kernel<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz, d_lt, d_lr, devStates, dt);
	
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
	cudaGraphicsUnmapResources(1, &resource_lt);
	cudaGraphicsUnmapResources(1, &resource_lr);
}

void CudaControler::conectBuffers(unsigned int bufferX,unsigned int bufferY, unsigned int bufferZ, unsigned int bufferLT, unsigned int bufferLR)
{
	
	cudaSafeCall( cudaGraphicsGLRegisterBuffer(&resource_x, (GLuint)bufferX, cudaGraphicsRegisterFlagsNone) );
	cudaSafeCall( cudaGraphicsGLRegisterBuffer(&resource_y, (GLuint)bufferY, cudaGraphicsRegisterFlagsNone) );
	cudaSafeCall( cudaGraphicsGLRegisterBuffer(&resource_z, (GLuint)bufferZ, cudaGraphicsRegisterFlagsNone) );
	cudaSafeCall( cudaGraphicsGLRegisterBuffer(&resource_lt, (GLuint)bufferLT, cudaGraphicsRegisterFlagsNone) );
	cudaSafeCall( cudaGraphicsGLRegisterBuffer(&resource_lr, (GLuint)bufferLR, cudaGraphicsRegisterFlagsNone) );

	// Initialize vectors on host
	for (size_t i = 0; i< sizeof(h_resource)/sizeof(*h_resource); i++)
		h_resource[i] = -1.f;

	// Copy host vectors to device
	size_t bytes = values::e_MaxParticles*sizeof(float);
	cudaMemcpy( d_lr, h_resource, bytes, cudaMemcpyHostToDevice);

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

	for(int i = 0; i<values::e_MaxParticles; i++)
	{
		cPrint(std::to_string(h_resource[i]) + " ", 2);
	}
	cPrint("\n", 1);
}


void CudaControler::copyConstants()
{
	cudaSafeCall(cudaMemcpyToSymbol(d_rLife, 		&(values::p_RLifeTime),		sizeof(const float)));
	cudaSafeCall(cudaMemcpyToSymbol(d_life, 		&(values::p_LifeTime), 		sizeof(const float)));
	cudaSafeCall(cudaMemcpyToSymbol(d_vDecay, 		&(values::p_VelocityDecay), sizeof(const float)));
	cudaSafeCall(cudaMemcpyToSymbol(d_initVelocity, &(values::p_InitVelocity), 	sizeof(const float)));
	cudaSafeCall(cudaMemcpyToSymbol(d_rInitVelocity,&(values::p_RInitVelocity), sizeof(const float)));
	cudaSafeCall(cudaMemcpyToSymbol(d_maxParticles, &(values::e_MaxParticles), 	sizeof(const unsigned int)));
	cudaSafeCall(cudaMemcpyToSymbol(d_emitterFrec, 	&(values::e_EmissionFrec), 	sizeof(const unsigned int)));


	cudaSafeCall(cudaMemcpyToSymbol(d_constantX,&(values::w_ConstantX), sizeof(const float)));
	cudaSafeCall(cudaMemcpyToSymbol(d_constantY,&(values::w_ConstantY), sizeof(const float)));
	cudaSafeCall(cudaMemcpyToSymbol(d_constantZ,&(values::w_ConstantZ), sizeof(const float)));

}