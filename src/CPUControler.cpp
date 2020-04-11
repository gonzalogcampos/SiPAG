#include <CPUControler.h>
#include <Values.h>
#include <Windows.h>

void CPUControler::importBuffers()
{

}

void CPUControler::exportBuffers()
{

}

void CPUControler::start()
{
    // Size, in bytes, of Particles vector host
	size_t bytes = e_MaxParticles*sizeof(float);

	//Allocate memory for resource vector in host
	x = (float*)malloc(bytes);
	y = (float*)malloc(bytes);
	z = (float*)malloc(bytes);
	vx = (float*)malloc(bytes);
	vy = (float*)malloc(bytes);
	vz = (float*)malloc(bytes);
	lt = (float*)malloc(bytes);
	lr = (float*)malloc(bytes);
}

void CPUControler::step(double dt)
{

}

void CPUControler::close()
{
    free(x);
    free(y);
    free(z);
    free(vx);
    free(vy);
    free(vz);
    free(lt);
    free(lr);
}

void CPUControler::resize()
{
    close();
    start();
}