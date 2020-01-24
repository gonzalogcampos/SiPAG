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



__global__ void setup_kernel ( curandState * state, unsigned long seed, unsigned int maxParticles)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx<maxParticles)
    curand_init ( seed, idx, 0, &state[idx] );
} 

// CUDA kernel. Each thread takes care of one element of c
__global__ void kernel(float *x, float *y, float *z,
							float *vx, float *vy, float *vz,
							float *lt, float *lr, 
							float rLife, 
							float life,
							curandState* state,
							float dt,
							unsigned int maxParticles, 
							unsigned int emitterFrec,
							float vDecay)
{

	// Get our global thread ID
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if (id < maxParticles)
	{
		/* curand works like rand - except that it takes a state as a parameter */
		if(lr[id]<0.f)
		{
			//Space to create a particle
			int r = curand(&state[id])%1000;
			if(r<emitterFrec)
			{
				r=curand(&state[id])%100;
				//x[id] = 0.f;
				y[id] = 0.f;
				z[id] = 0.f;

				vx[id] = 10.f;
				vy[id] = 100.f;
				vz[id] = 1000.f;

				lt[id] = 0.f;
				lr[id] = life -life*rLife + 2*life*rLife*r/100;
			}
		}else
		{
			//Velocity Decay
			vx[id] = vx[id] - vx[id]*vDecay*dt;
			vy[id] = vy[id] - vy[id]*vDecay*dt;
			vz[id] = vz[id] - vz[id]*vDecay*dt;

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

	//Allocate memory for each vector in host
	h_resource = (float*)malloc(bytes);
	
	//Allocate memory for each vector in device
	//cudaMalloc(&d_x, bytes);
	cudaMalloc(&d_y, bytes);
	cudaMalloc(&d_z, bytes);

	cudaMalloc(&d_vx, bytes);
	cudaMalloc(&d_vy, bytes);
	cudaMalloc(&d_vz, bytes);

	cudaMalloc(&d_lt, bytes);
	cudaMalloc(&d_lr, bytes);

	// Initialize vectors on host
	for (size_t i = 0; i< sizeof(h_resource)/sizeof(*h_resource); i++)
		h_resource[i] = -1.f;

	// Copy host vectors to device
	cudaMemcpy( d_lr, h_resource, bytes, cudaMemcpyHostToDevice);

	// Number of threads in each block
	blockSize = values::cu_BlockSize;

	// Number of blocks in grid
	gridSize = (int)ceil((float)values::e_MaxParticles/blockSize);
}

void CudaControler::closeKernel()
{
		// Release device memory
		//cudaFree(d_x);
		//cudaFree(d_y);
		//cudaFree(d_z);
		cudaFree(d_vx);
		cudaFree(d_vy);
		cudaFree(d_vz);
		//cudaFree(d_lt);
		cudaFree(d_lr);

		// Release host memory
		free(h_resource);
}

void CudaControler::step(float dt)
{
	//Set the buffers
	size_t bytes = values::e_MaxParticles*sizeof(float);
	cudaSafeCall( cudaGraphicsMapResources(1, &resource_x) );
	cudaSafeCall( cudaGraphicsMapResources(1, &resource_y) );
	cudaSafeCall( cudaGraphicsMapResources(1, &resource_z) );
	cudaSafeCall( cudaGraphicsMapResources(1, &resource_l) );
	cudaSafeCall( cudaGraphicsResourceGetMappedPointer((void **)&d_x, &bytes, resource_x) );
	cudaSafeCall( cudaGraphicsResourceGetMappedPointer((void **)&d_y, &bytes, resource_y) );
	cudaSafeCall( cudaGraphicsResourceGetMappedPointer((void **)&d_z, &bytes, resource_z) );
	cudaSafeCall( cudaGraphicsResourceGetMappedPointer((void **)&d_lt, &bytes, resource_l) );

	//Execute Kernel
	//Random device States
	curandState* devStates;
	cudaMalloc ( &devStates, values::e_MaxParticles*sizeof( curandState ) );
	setup_kernel<<<gridSize, blockSize>>> ( devStates, rand()%10000, values::e_MaxParticles);

	kernel<<<gridSize, blockSize>>>(d_x, d_y, d_z, 
										d_vx, d_vy, d_vz,
										d_lt, d_lr, values::p_RLifeTime, values::p_LifeTime,
										devStates,
										dt,
										values::e_MaxParticles,
										values::e_EmissionFrec,
										values::p_VelocityDecay);
	
	cudaFree(devStates);



	//printData(PARTICLE_X);
	//printData(PARTICLE_VX);


	//Reset the buffers
	cudaGraphicsUnmapResources(1, &resource_x);
	cudaGraphicsUnmapResources(1, &resource_y);
	cudaGraphicsUnmapResources(1, &resource_z);
	cudaGraphicsUnmapResources(1, &resource_l);
}


void CudaControler::sendBuffer(unsigned int buffer)
{
	
	cudaSafeCall( cudaGraphicsGLRegisterBuffer(&resource_x, (GLuint)buffer, cudaGraphicsRegisterFlagsNone) );
}


void CudaControler::sendBuffers(unsigned int bufferX,unsigned int bufferY, unsigned int bufferZ, unsigned int bufferL)
{
	
	cudaSafeCall( cudaGraphicsGLRegisterBuffer(&resource_x, (GLuint)bufferX, cudaGraphicsRegisterFlagsNone) );
	cudaSafeCall( cudaGraphicsGLRegisterBuffer(&resource_y, (GLuint)bufferY, cudaGraphicsRegisterFlagsNone) );
	cudaSafeCall( cudaGraphicsGLRegisterBuffer(&resource_z, (GLuint)bufferZ, cudaGraphicsRegisterFlagsNone) );
	cudaSafeCall( cudaGraphicsGLRegisterBuffer(&resource_l, (GLuint)bufferL, cudaGraphicsRegisterFlagsNone) );

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