#pragma once
#include <iostream>

enum Data;


class CudaControler
{
    public:

        static CudaControler* getInstance(){
            static CudaControler only_instance;
            return &only_instance;
        }

        void step(float dt);
        void setKernel();
        void closeKernel();
        int testDevices();
        std::string getDevice();
        //void reserveGrid();
        void conectBuffers(unsigned int bufferX,unsigned int bufferY, unsigned int bufferZ, unsigned int bufferLT, unsigned int bufferLR);
        
    private:

        CudaControler(){}
        ~CudaControler(){}

        void showDevices();
        void cudaSafeCall(cudaError err);
        void printData(Data d);
        void copyConstants();

		float *h_resource;    //host resource for  copy buffers

        float *d_x;	    //position x
	    float *d_y;	    //position y
	    float *d_z;     //position z

	    float *d_vx;	//velocity x
	    float *d_vy;	//velocity y
	    float *d_vz;	//velocity x

	    float *d_lt;	//life time
	    float *d_lr;	//life remaining

        int blockSize, gridSize;

        cudaGraphicsResource_t resource_x = 0;
        cudaGraphicsResource_t resource_y = 0;
        cudaGraphicsResource_t resource_z = 0;
        cudaGraphicsResource_t resource_lt = 0;
        cudaGraphicsResource_t resource_lr = 0;

};