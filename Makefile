CPP := CudaControler.cu 
CU := src/SIPAG.cpp src/Console.cpp src/Render.cpp src/Simulation.cpp

CUDALIB := -L/usr/local/cuda/lib64 -lcuda -lcudart
GLFWLIB := -Llib/GL -lglew32 -lglfw3 -lfreeglut -lglu32 -lopengl32

all: CudaControler.o$
^Ig++ -o bin/$@ $(CPP) bin/CudaControler.o -Iinclude/ $(CUDALIB) $(GLFWLIB)$


CudaControler.o:
    nvcc -c -o bin/$@ -arch=sm_20 $(CU)


clean: rm -f *.o program