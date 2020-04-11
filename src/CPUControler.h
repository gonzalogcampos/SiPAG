#pragma once

class CPUControler
{
    private:
        CPUControler(){};
    public:
        static CPUControler* getInstance()
        {
            static CPUControler onlyInstance;
            return &onlyInstance;
        }

        void importBuffers();
        void exportBuffers();
        void resize();
        void step(double dt);
        void start();
        void close();

        float *x, *y, *z, *lt, *lr, *vx, *vy, *vz;
};