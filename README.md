# CS 217 Project MRI-q

Repo for the CS 217 GPGPU Architecture Class final project.

## Implementation Methods Attempted

Atempted Implementations
1. Naive Implementation of `ComputePhiMag()`
1. Naive Implementation of `ComputeQ()`
1. Naive Constant of `ComputeQ()` for Kvals
1. Naive Constant of `ComputeQ()` for XYZ vals
1. Memory Efficient Constant of `ComputeQ()` for Kvals
1. Memory Efficient Constant of `ComputeQ()` for XYZ Vals

Overarching Ideas Implemented
1. Constant Memory
1. Streams
1. Dynamic Parallelism
 <!-- Doesn't make since since we don't know how  -->


## Execution Notes

- `-i` inputfile
- `-o` outputfile


## Running Baseline

- `CPU`
	- `make cpu`: Use this command to make the executable `mri-q`
	- `make cpu_small`: Use this command to run the small file with the CPU executable and compare the outputs in the dataset directory
	- `make cpu_large`: Use this command to run the large file with the CPU executable and compare the outputs in the dataset directory
	- `make cpu_128`: Use this command to run the 128 file with the CPU executable and compare the outputs in the dataset directory
- `GPU`
	- `make gpu`: Use this command to make the executable `mri-q-gpu`
	- `make gpu_small`: Use this command to run the small file with the GPU executable and compare the outputs in the dataset directory
	- `make gpu_large`: Use this command to run the large file with the GPU executable and compare the outputs in the dataset directory
	- `make gpu_128`: Use this command to run the 128 file with the GPU executable and compare the outputs in the dataset directory


## Running CPU

```BASH
################# SMALL #####################
./mri-q -i ./datasets/small/input/32_32_32_dataset.bin -o ./small.out
/tools/compare-output ./small.out ./datasets/small/output/32_32_32_dataset.out

################# LARGE #####################
./mri-q -i ./datasets/large/input/64_64_64_dataset.bin -o ./large.out
./tools/compare-output ./large.out ./datasets/large/output/64_64_64_dataset.out 

################# 128 #####################
./mri-q -i ./datasets/128x128x128/input/128x128x128.bin -o ./128.out
./tools/compare-output ./128.out ./datasets/large/output/64_64_64_dataset.out 
```

## Running GPU

```BASH
################# SMALL #####################
./mri-q-gpu -i ./datasets/small/input/32_32_32_dataset.bin -o ./small.out
./tools/compare-output ./small.out ./datasets/small/output/32_32_32_dataset.out

################# LARGE #####################
./mri-q-gpu -i ./datasets/large/input/64_64_64_dataset.bin -o ./large.out
./tools/compare-output ./large.out ./datasets/large/output/64_64_64_dataset.out 

################# 128 #####################
./mri-q-gpu -i ./datasets/128x128x128/input/128x128x128.bin -o ./128.out
./tools/compare-output ./128.out ./datasets/large/output/64_64_64_dataset.out 
```