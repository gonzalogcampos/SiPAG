//MIT License
//Copyright (c) 2019 Gonzalo G Campos

/* PRAGMA */
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

        float *x, *y, *z, *lt, *lr, *vx, *vy, *vz;

        void expData();
        void impData();
        void resize();
        void step(double dt);
        void start();
        void close();
};