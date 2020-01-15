#pragma once
class CudaControler
{
    public:
        void step(float dt);
        void setKernel();
        void closeKernel();
    private:
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