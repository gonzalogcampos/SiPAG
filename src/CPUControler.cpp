//MIT License
//Copyright (c) 2019 Gonzalo G Campos

#include <CPUControler.h>
#include <Values.h>
#include <Windows.h>
#include <CudaControler.h>
#include <NoiseFunc.h>
#include <Render.h>

void CPUControler::impData()
{
    CudaControler::getInstance()->expData(x, y, z, vx, vy, vz, lt, lr);
}

void CPUControler::expData()
{
    CudaControler::getInstance()->impData(x, y, z, vx, vy, vz, lt, lr);
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
    for(int  i = 0; i < e_MaxParticles; i++)
    {
        /* curand works like rand - except that it takes a state as a parameter */
        if(lr[i]<0.f)
        {
            //Space to create a particle
            int r = rand()%1000;
            if(r<e_EmissionFrec)
            {
                //Fist of all choose the position
                if(e_Type==2)
                {
                    r=rand()%1000;
                    float theta =  (r/1000.f) * 2.0 * 3.14159265359;
                    float phi = acos(2.0 *  (r/1000.f) - 1.0) - (3.14159265359/2);
                    float sinTheta = sin(theta);
                    float cosTheta = cos(theta);
                    float sinPhi = sin(phi);
                    float cosPhi = cos(phi);
                    x[i] = e_Length * cosPhi * cosTheta;
                    y[i] = e_Length * cosPhi * sinTheta;
                    z[i] = e_Length * sinPhi;
                }
                else if(e_Type==1)
                {
                    r=rand()%1000;
                    x[i] = e_Length*((r/1000.f)-.5f);
                    y[i] = 0.f;
                    z[i] = 0.f;
                }else
                {
                    r=rand()%1000;
                    float theta = (r/1000.f) * 2.0 * 3.14159265359;
                    r=rand()%1000;
                    float phi = acos(2.0 * (r/1000.f) - 1.0) - (3.14159265359/2);
                    float sinTheta = sin(theta);
                    float cosTheta = cos(theta);
                    float sinPhi = sin(phi);
                    float cosPhi = cos(phi);
                    x[i] = e_Length * cosPhi * cosTheta;
                    y[i] = e_Length * cosPhi * sinTheta;
                    z[i] = e_Length * sinPhi;
                }
            

                //Then calculate de init velocity
                r=rand()%1000;
                vx[i] = p_InitVelocity[0] + p_RInitVelocity[0]*2.f*((r/1000.f)-0.5f);

                r=rand()%1000;
                vy[i] = p_InitVelocity[1] + p_RInitVelocity[1]*2.f*((r/1000.f)-0.5f);

                r=rand()%1000;
                vz[i] = p_InitVelocity[2] + p_RInitVelocity[2]*2.f*((r/1000.f)-0.5f);


                //And last, calculate the life
                r=rand()%1000;
                lt[i] = 0.f;
                lr[i] = p_LifeTime + p_RLifeTime*(0.5*(r/1000.f));
            }
        }else
        {
            //Velocity Decay
            vx[i] = vx[i] - vx[i]*p_VelocityDecay*dt;
            vy[i] = vy[i] - vy[i]*p_VelocityDecay*dt;
            vz[i] = vz[i] - vz[i]*p_VelocityDecay*dt;

            //Wind constant velocity
            vx[i] = vx[i] + w_Constant[0];
            vy[i] = vy[i] + w_Constant[1];
            vz[i] = vz[i] + w_Constant[2];

            
			//Wind perlin Big
			if(w_1)
			{
				floatv pos = floatv(x[i], y[i], z[i]);
				float time = dt*timeEv;
				float pbx = w_1Amp[0]*_repeaterPerlin(pos, time, w_1Size, 27989,   w_1n, w_1lacunarity, w_1decay);
				float pby = w_1Amp[1]*_repeaterPerlin(pos, time, w_1Size, 8461126, w_1n, w_1lacunarity, w_1decay);
				float pbz = w_1Amp[2]*_repeaterPerlin(pos, time, w_1Size, 1892777, w_1n, w_1lacunarity, w_1decay);

				vx[i] = vx[i] + pbx;
				vy[i] = vy[i] + pby;
				vz[i] = vz[i] + pbz;
			}
			
			if(w_2)
			{
				floatv pos = floatv(x[i], y[i], z[i]);
				float time = dt*timeEv;
				float pbx = w_2Amp[0]*_repeaterPerlin(pos, time, w_2Size, 2989,   w_2n, w_2lacunarity, w_2decay);
				float pby = w_2Amp[1]*_repeaterPerlin(pos, time, w_2Size, 841126, w_2n, w_2lacunarity, w_2decay);
				float pbz = w_2Amp[2]*_repeaterPerlin(pos, time, w_2Size, 189277, w_2n, w_2lacunarity, w_2decay);

				vx[i] = vx[i] + pbx;
				vy[i] = vy[i] + pby;
				vz[i] = vz[i] + pbz;
			}


            //Position addition
            x[i] += vx[i]*dt;
            y[i] += vy[i]*dt;
            z[i] += vz[i]*dt;

            //Life set
            lr[i] = lr[i] - dt;
            lt[i] = lt[i] + dt;

        }
    }//ForLooop

    if(r_enable)
        Render::getInstance()->pasateBuffers(x, y, z, vx, vy, vz, lt, lr);
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