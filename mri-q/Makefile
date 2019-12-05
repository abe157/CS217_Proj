CC	= gcc
CPP	= g++
EXE	= mri-q
OBJ	= file.o parboil.o args.o main.o
APP_CXXFLAGS=-O3 -funroll-all-loops -ffast-math
APP_CFLAGS=-O3 -funroll-all-loops -ffast-math
APP_LDFLAGS=-lm -lstdc++
LANGUAGE=c

NVCC = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
GPUEXE = mri-q-gpu
GPU_OBJ = file.o parboil.o args.o main_gpu.o

# CPU BUILD

default: $(EXE)

main.o: main.c computeQ.cc parboil.h
	$(CC) $(APP_CFLAGS) -c main.c

parboil.o: parboil.c parboil.h
	$(CC) $(APP_CFLAGS) -c parboil.c

file.o: file.cc file.h
	$(CPP) $(APP_CXXFLAGS) -c file.cc

args.o: args.c 
	$(CC) $(APP_CFLAGS) -c args.c

mri-q: main.o file.o args.o parboil.o
	$(CPP) $(APP_CXXFLAGS) $(OBJ) -o $(EXE) $(APP_LDFLAGS)


# GPU BUILD

main_gpu.o:  main.cu computeQ_gpu.cu parboil.h support.h
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS) 

gpu: $(GPU_OBJ)
	$(NVCC) $(GPU_OBJ) -o $(GPUEXE) $(LD_FLAGS)
	
# RUNNING

cpu_small:
	./mri-q -i ./datasets/small/input/32_32_32_dataset.bin -o ./small.out
	/tools/compare-output ./small.out ./datasets/small/output/32_32_32_dataset.out

cpu_large:
	./mri-q -i ./datasets/large/input/64_64_64_dataset.bin -o ./large.out
	./tools/compare-output ./large.out ./datasets/large/output/64_64_64_dataset.out 

cpu_128:
	./mri-q -i ./datasets/128x128x128/input/128x128x128.bin -o ./128.out
	./tools/compare-output ./128.out ./datasets/large/output/64_64_64_dataset.out 

gpu_small:
	./mri-q-gpu -i ./datasets/small/input/32_32_32_dataset.bin -o ./small.out
	/tools/compare-output ./small.out ./datasets/small/output/32_32_32_dataset.out

gpu_large:
	./mri-q-gpu -i ./datasets/large/input/64_64_64_dataset.bin -o ./large.out
	./tools/compare-output ./large.out ./datasets/large/output/64_64_64_dataset.out 

gpu_128:
	./mri-q-gpu -i ./datasets/128x128x128/input/128x128x128.bin -o ./128.out
	./tools/compare-output ./128.out ./datasets/large/output/64_64_64_dataset.out 

clean:
	rm -f *.o $(EXE) $(GPUEXE)
