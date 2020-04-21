//MIT License
//Copyright (c) 2019 Gonzalo G Campos

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

        void step(double dt);
        void start();
        void close();
        void resize();
        int testDevices();
        void setImportData(bool i){importData = i;}
        void expData(float* x, float*  y, float* z, float* vx, float* vy, float* vz, float* lt, float* lr);
        void impData(float* x, float*  y, float* z, float* vx, float* vy, float* vz, float* lt, float* lr);
        std::string getDevice();
        void conectBuffers(unsigned int bufferX,unsigned int bufferY, unsigned int bufferZ, 
                            unsigned int bufferVX, unsigned int bufferVY, unsigned int bufferVZ, 
                            unsigned int bufferLT, unsigned int bufferLR);
        
    private:

        CudaControler(){}
        ~CudaControler(){}

        void showDevices();
        void cudaSafeCall(cudaError err);
        void printData(Data d);
        void copyConstants();
        void mapResources();
        void unmapResources();
        
		float *h_resource;    //host resource for copy buffers usually unused
        bool importData = false;

        /*
        Device Buffers
        */
        void* devStates;//random state

        float *d_x;	    //position x
	    float *d_y;	    //position y
	    float *d_z;     //position z

	    float *d_vx;	//velocity x
	    float *d_vy;	//velocity y
	    float *d_vz;	//velocity x

	    float *d_lt;	//life time
	    float *d_lr;	//life remaining

        /*
        OpenGL Buffers
        */
        cudaGraphicsResource_t resource_x = 0;
        cudaGraphicsResource_t resource_y = 0;
        cudaGraphicsResource_t resource_z = 0;        
        cudaGraphicsResource_t resource_vx = 0;
        cudaGraphicsResource_t resource_vy = 0;
        cudaGraphicsResource_t resource_vz = 0;
        cudaGraphicsResource_t resource_lt = 0;
        cudaGraphicsResource_t resource_lr = 0;

        bool mapped = false;

};