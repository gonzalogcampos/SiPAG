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

#include <CudaControler.h>
#include <Console.h>

//Funciones para el timestamp
typedef LARGE_INTEGER timeStamp;
void getCurrentTimeStamp(timeStamp& _time);
timeStamp getCurrentTimeStamp();
double getTimeMili(const timeStamp& start, const timeStamp& end);
double getTimeSecs(const timeStamp& start, const timeStamp& end);


__global__ void setup_kernel ( curandState * state, unsigned long seed )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init ( seed, idx, 0, &state[idx] );
} 

// CUDA kernel. Each thread takes care of one element of c
__global__ void vecAdd(double *a, 
						double *b, 
						double *c, 
						int n, 
						int nrandom, 
						bool print,
						curandState* state)
{

	// Get our global thread ID
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if (id < n)
	{
		for(int i = 0; i<nrandom; i++)
		{
			/* curand works like rand - except that it takes a state as a parameter */
			int r = curand(&state[id]) % 9999 ;

			if(print)
			{
				printf("Particula: %i Random: %i --- %i\n", id, i, r);
			}
		}
        c[id] = a[id] + b[id];
	}

}

double getTimeMili()
{
		timeStamp start;
		timeStamp dwFreq;
		QueryPerformanceFrequency(&dwFreq);
		QueryPerformanceCounter(&start);
		return double(start.QuadPart) / double(dwFreq.QuadPart);
}

void CudaControler::showDevices()
{
	    // Number of CUDA devices
		int devCount;
		cudaGetDeviceCount(&devCount);
		printf("CUDA Device Query...\n");
		printf("There are %d CUDA devices.\n", devCount);

		// Iterate through devices
		for (int i = 0; i < devCount; ++i)
		{
			// Get device properties
			printf("\nCUDA Device #%d\n", i);
			cudaDeviceProp devProp;
			cudaGetDeviceProperties(&devProp, i);
			printf("Major revision number:         %d\n",  devProp.major);
			printf("Minor revision number:         %d\n",  devProp.minor);
			printf("Name:                          %s\n",  devProp.name);
			printf("Total global memory:           %u\n",  devProp.totalGlobalMem);
			printf("Total shared memory per block: %u\n",  devProp.sharedMemPerBlock);
			printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
			printf("Warp size:                     %d\n",  devProp.warpSize);
			printf("Maximum memory pitch:          %u\n",  devProp.memPitch);
			printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
			for (int i = 0; i < 3; ++i)
			printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
			for (int i = 0; i < 3; ++i)
			printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
			printf("Clock rate:                    %d\n",  devProp.clockRate);
			printf("Total constant memory:         %u\n",  devProp.totalConstMem);
			printf("Texture alignment:             %u\n",  devProp.textureAlignment);
			printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
			printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
			printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
		}
}

void CudaControler::doSomething()
{
	cPrint("Ejecutando kernel de cuda\n", 0);

	//Max number of particles
	int n = 2;
	
	// Host input vectors
	/*
	float *h_x;		//position x
	float *h_y;		//position y
	float *h_z;		//position z

	float *h_vx;	//velocity x
	float *h_vy;	//velocity y
	float *h_vz;	//velocity x

	float *h_rb;	//random born

	float *h_lt;	//life time
	float *h_lr;	//life remaining
	float *h_rl;	//random life

	float *h_ro;	//random opacity
	float *h_o;		//opacity
*/


	// Host input vectors
	double *h_a;
	double *h_b;

	//Host output vector
	double *h_c;

	// Device input vectors
	double *d_a;
	double *d_b;
	//Device output vector
	double *d_c;

	// Size, in bytes, of each vector
	size_t bytes = n*sizeof(double);

	// Allocate memory for each vector on host
	h_a = (double*)malloc(bytes);
	h_b = (double*)malloc(bytes);
	h_c = (double*)malloc(bytes);

	// Allocate memory for each vector on GPU
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	int i;
	// Initialize vectors on host
	for( i = 0; i < n; i++ ) {
		h_a[i] = sin(i)*sin(i);
		h_b[i] = cos(i)*cos(i);
	}

	// Copy host vectors to device
	cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy( d_b, h_b, bytes, cudaMemcpyHostToDevice);

	int blockSize, gridSize;

	// Number of threads in each block
	blockSize = 1024;

	// Number of blocks in grid
	gridSize = (int)ceil((float)n/blockSize);

	printf("Numero de particulas: %i\n", n);

	curandState* devStates;
	cudaMalloc ( &devStates, n*sizeof( curandState ) );

	// Execute the kernel
	for(int i=0; i<3; i++)
	{
		setup_kernel<<<gridSize, blockSize>>> ( devStates, rand()%100000 );
		vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n, 3, true, devStates);
	}

	// Copy array back to host
	cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost );

	// Sum up vector c and print result divided by n, this should equal 1 within error
	double sum = 0;
	for(i=0; i<n; i++)
		sum += h_c[i];
	printf("final result: %f\n", sum/n);

	// Release device memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	// Release host memory
	free(h_a);
	free(h_b);
	free(h_c);
}