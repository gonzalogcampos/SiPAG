@ECHO Compilando...
nvcc -dc src/CudaControler.cu -o bin/CudaControler.obj -x c++ -ccbin g++ -Isrc/ -Iinclude/cu

g++ -c src/SIPAG.cpp -o bin/SIPAG.o -Isrc/. -Iinclude/cu
g++ -c src/Simulation.cpp -o bin/Simulation.o -Isrc/. -Iinclude/cu
g++ -c src/Console.cpp -o bin/Console.o -Isrc/. -Iinclude/cu

@ECHO Linkando...
g++ bin/SIPAG.o bin/Link.o bin/Console.o bin/Simulation.o -o Test.exe  -L/lib/cu -lcudart -lcuda -lz
PAUSE