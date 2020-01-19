#pragma once

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
        
    private:

        CudaControler(){}
        ~CudaControler(){}

        void showDevices();

		float *h_lr; //life remaingin

        float *d_x;	//position x
	    float *d_y;	//position y
	    float *d_z;  //position z

	    float *d_vx;	//velocity x
	    float *d_vy;	//velocity y
	    float *d_vz;	//velocity x

	    float *d_lt;	//life time
	    float *d_lr;	//life remaining

        int blockSize, gridSize;
};