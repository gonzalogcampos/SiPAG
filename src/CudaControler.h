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
        
		float *h_resource;    //host resource for  copy buffers

        //If simple precision values we will use theese
        float *d_x_s;	    //position x
	    float *d_y_s;	    //position y
	    float *d_z_s;     //position z

	    float *d_vx_s;	//velocity x
	    float *d_vy_s;	//velocity y
	    float *d_vz_s;	//velocity x

	    float *d_lt_s;	//life time
	    float *d_lr_s;	//life remaining

        //If double precision values then
        float *d_x_d;	    //position x
	    float *d_y_d;	    //position y
	    float *d_z_d;     //position z

	    float *d_vx_d;	//velocity x
	    float *d_vy_d;	//velocity y
	    float *d_vz_d;	//velocity x

	    float *d_lt_d;	//life time
	    float *d_lr_d;	//life remaining

        cudaGraphicsResource_t resource_x = 0;
        cudaGraphicsResource_t resource_y = 0;
        cudaGraphicsResource_t resource_z = 0;        
        cudaGraphicsResource_t resource_vx = 0;
        cudaGraphicsResource_t resource_vy = 0;
        cudaGraphicsResource_t resource_vz = 0;
        cudaGraphicsResource_t resource_lt = 0;
        cudaGraphicsResource_t resource_lr = 0;

};