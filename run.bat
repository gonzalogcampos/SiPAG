@ECHO Compilando...
ECHO nvcc -c src/CudaControler.cu -o bin/CudaControler.o -Isrc/ -Iinclude/cu

ECHO nvcc -c -x cu src/SIPAG.cpp -o bin/SIPAG.o -Isrc/. -Iinclude/cu
ECHO nvcc -c -x cu src/Simulation.cpp -o bin/Simulation.o -Isrc/. -Iinclude/cu
ECHO nvcc -c -x cu src/Console.cpp -o bin/Console.o -Isrc/. -Iinclude/cu

nvcc src/CudaControler.cu -x cu src/Console.cpp src/Simulation.cpp src/SIPAG.cpp -o Test -Isrc/ -Iinclude/cu

@ECHO Linkando...
cd bin
ECHO nvcc SIPAG.o CudaControler.o Console.o Simulation.o -o ../Test.
PAUSE