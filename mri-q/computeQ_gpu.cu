/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
#include "support.h"

#define PI   3.1415926535897932384626433832795029f
#define PIx2 6.2831853071795864769252867665590058f

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define K_ELEMS_PER_GRID 2048


#define KERNEL_PHI_MAG_THREADS_PER_BLOCK 512
#define KERNEL_Q_THREADS_PER_BLOCK 256
#define KERNEL_Q_K_ELEMS_PER_GRID 1024

#define PHIMAGBLOCK_SIZE 512 // 512 or 192

// 16 bytes structure
// constant memory is 64KB == 2^16
// Max kValues elements is 4094 ~= 2^16B / 16B
#define kValuesMax 4094

// Variables x,y,z are each a floats = 4B, for all 3 is 12B
// constant memory is 64KB == 2^16
// Max x,y,z elements is  5459 ~= 2^16B / 16B
#define floatMax 5459

struct kValues {
  float Kx;
  float Ky;
  float Kz;
  float PhiMag;
};

// kValues Constant
// __device__ __constant__ struct kValues Kvalues_c[kValuesMax];

X,Y,Z Constant
__device__ __constant__ float x_c[floatMax];
__device__ __constant__ float y_c[floatMax];
__device__ __constant__ float z_c[floatMax];


//--------------------------------------------------------------------------------------------------------------
// ALL CPU IMPLEMENTATIONS
//--------------------------------------------------------------------------------------------------------------

void createDataStructsCPU(int numK, int numX, float** phiMag, float** Qr, float** Qi){
  *phiMag = (float* ) memalign(16, numK * sizeof(float));
  *Qr = (float*) memalign(16, numX * sizeof (float));
  memset((void *)*Qr, 0, numX * sizeof(float));
  *Qi = (float*) memalign(16, numX * sizeof (float));
  memset((void *)*Qi, 0, numX * sizeof(float));
}

inline void ComputeQCPU(int numK, int numX, struct kValues *kVals, float* x, float* y, float* z, float *__restrict__ Qr, float *__restrict__ Qi) {
  float expArg;
  float cosArg;
  float sinArg;

  int indexK, indexX;

  // Loop over the space and frequency domains.
  // Generally, numX > numK.
  // Since loops are not tiled, it's better that the loop with the smaller
  // cache footprint be innermost.
  for (indexX = 0; indexX < numX; indexX++) {

    // Sum the contributions to this point over all frequencies
    float Qracc = 0.0f;
    float Qiacc = 0.0f;
    for (indexK = 0; indexK < numK; indexK++) {
      expArg = PIx2 * (kVals[indexK].Kx * x[indexX] +
                       kVals[indexK].Ky * y[indexX] +
                       kVals[indexK].Kz * z[indexX]);

      cosArg = cosf(expArg);
      sinArg = sinf(expArg);

      float phi = kVals[indexK].PhiMag;
      Qracc += phi * cosArg;
      Qiacc += phi * sinArg;
    }
    Qr[indexX] = Qracc;
    Qi[indexX] = Qiacc;
  }
}

inline void ComputePhiMagCPU(int numK, float* phiR, float* phiI, float* phiMag) {
  // Baseline Implementation
  int indexK = 0;
  for (indexK = 0; indexK < numK; indexK++) {
    float real = phiR[indexK];
    float imag = phiI[indexK];
    phiMag[indexK] = real*real + imag*imag;
  }
}


//--------------------------------------------------------------------------------------------------------------
// ALL GPU IMPLEMENTATIONS
//--------------------------------------------------------------------------------------------------------------


// NAIVE ComputePhiMag ON GPU

__global__ void ComputePhiMagGPUKernel(int numk, float* phiR, float* phiI, float* phiMag){

  unsigned int t = threadIdx.x;
  unsigned int offset = (blockIdx.x*blockDim.x) + t;

  // __shared__ float real[blockDim.x];
  // __shared__ float imag[blockDim.x];
  // __shared__ float phiOut[blockDim.x];

  if(offset < numk){
    float real = phiR[offset];
    float imag = phiI[offset];
    phiMag[offset] = real*real + imag*imag;
  }
}

inline void ComputePhiMagGPU(int numK, float* phiR, float* phiI, float* phiMag) {
  cudaError_t cuda_ret;

  float *phiR_d, *phiI_d, *phiMag_d;

  // Allocate device variables ---------------------------------
  cuda_ret = cudaMalloc((void**)&phiR_d, numK * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
  cuda_ret = cudaMalloc((void**)&phiI_d, numK * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
  cuda_ret = cudaMalloc((void**)&phiMag_d, numK * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");

  cuda_ret = cudaMemcpy(phiR_d, phiR, numK * sizeof(float), cudaMemcpyHostToDevice);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device ");
  cuda_ret = cudaMemcpy(phiI_d, phiI, numK * sizeof(float), cudaMemcpyHostToDevice);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device ");
  cuda_ret = cudaMemset(phiMag_d, 0, numK * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory ");


  // Launch kernel ----------------------------------------------------------

  dim3 dim_grid, dim_block;
  unsigned block, grid;
  block = PHIMAGBLOCK_SIZE;
  grid = numK / (PHIMAGBLOCK_SIZE);
  if( numK % (PHIMAGBLOCK_SIZE * grid)) grid++;

  printf("\tBLOCK: %d\n\tGRID: %d\n", block, grid);

  dim_block.x = block;
  dim_block.y = 1;
  dim_block.z = 1;

  dim_grid.x = grid;
  dim_grid.y = 1;
  dim_grid.z = 1;

  ComputePhiMagGPUKernel<<<dim_grid, dim_block>>>(numK, phiR_d, phiI_d, phiMag_d);

  cuda_ret = cudaDeviceSynchronize();
  if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");


  cuda_ret = cudaMemcpy(phiMag, phiMag_d, numK * sizeof(float), cudaMemcpyDeviceToHost);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host in naive reduction.");

  cudaFree(phiR_d);
  cudaFree(phiI_d);
  cudaFree(phiMag_d);
}


// USING X,Y,Z AS A CONSTANT------------------------------------------------------------------------------------

__global__ void ComputeQGPUKernel_3(int numK, int numX, struct kValues *kVals, float* x, float* y, float* z, float *Qr, float *Qi){
  __shared__ float x_s[PHIMAGBLOCK_SIZE];
  __shared__ float y_s[PHIMAGBLOCK_SIZE];
  __shared__ float z_s[PHIMAGBLOCK_SIZE];
  // Store this in cache memory, has a lot of resue ability
  // __shared__ struct kValues kVals_s[numK];

  unsigned int t = threadIdx.x;
  unsigned int offset = (blockIdx.x*blockDim.x) + t;


  if(offset < numX){
    x_s[t] = x[offset];
    y_s[t] = y[offset];
    z_s[t] = z[offset];
    // kVals_s[t] = kVals[offset];

    int indexK;
    float Qracc = 0.0f;
    float Qiacc = 0.0f;
    float expArg = 0.0f;
    float cosArg = 0.0f;
    float sinArg = 0.0f;
    float phi = 0.0f;

    for (indexK = 0; indexK < numK; indexK++) {
      // Generally, numX > numK

      if(offset < floatMax){ // Use constant memory
        expArg = PIx2 * (kVals[indexK].Kx * x_c[offset] + kVals[indexK].Ky * y_c[offset] + kVals[indexK].Kz * z_c[offset]);
        phi = kVals[indexK].PhiMag;
      } 
      else { // Use global memory
        expArg = PIx2 * (kVals[indexK].Kx * x_s[t] + kVals[indexK].Ky * y_s[t] + kVals[indexK].Kz * z_s[t]);
        phi = kVals[indexK].PhiMag;
      }

      cosArg = cosf(expArg);
      sinArg = sinf(expArg);

      Qracc += phi * cosArg;
      Qiacc += phi * sinArg;
    }
    Qr[offset] = Qracc;
    Qi[offset] = Qiacc;
  }
  __syncthreads();
}

void ComputeQGPU_3(int numK, int numX, struct kValues *kVals, float* x, float* y, float* z, float *__restrict__ Qr, float *__restrict__ Qi){
  cudaError_t cuda_ret;

  float *x_d; //numX
  float *y_d; //numX
  float *z_d; //numX
  float *Qr_d; //numX
  float *Qi_d; //numX
  
  struct kValues *kVals_d;

  // Allocate device variables ---------------------------------
  cuda_ret = cudaMalloc((void**)&x_d, numX * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
  cuda_ret = cudaMalloc((void**)&y_d, numX * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
  cuda_ret = cudaMalloc((void**)&z_d, numX * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
  cuda_ret = cudaMalloc((void**)&Qr_d, numX * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
  cuda_ret = cudaMalloc((void**)&Qi_d, numX * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
  cuda_ret = cudaMalloc((void**)&kVals_d, numK * sizeof(struct kValues));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");


  cuda_ret = cudaMemcpy(x_d, x, numX * sizeof(float), cudaMemcpyHostToDevice);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device ");
  cuda_ret = cudaMemcpy(y_d, y, numX * sizeof(float), cudaMemcpyHostToDevice);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device ");
  cuda_ret = cudaMemcpy(z_d, z, numX * sizeof(float), cudaMemcpyHostToDevice);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device ");
  cuda_ret = cudaMemset(Qr_d, 0, numX * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory ");
  cuda_ret = cudaMemset(Qi_d, 0, numX * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory ");

  // Use constant memory for the first floatMax elements
  int constCopies = 0;
  if(numX < floatMax )
    constCopies = numX;
  else
    constCopies = floatMax;

  cuda_ret = cudaMemcpyToSymbol(x_c, x, constCopies * sizeof(float), 0, cudaMemcpyHostToDevice);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device ");
  cuda_ret = cudaMemcpyToSymbol(y_c, y, constCopies * sizeof(float), 0, cudaMemcpyHostToDevice);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device ");
  cuda_ret = cudaMemcpyToSymbol(z_c, z, constCopies * sizeof(float), 0, cudaMemcpyHostToDevice);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device ");

  // Use global memory for the [kValuesMax:numK] elements not already in constant memory
  // cuda_ret = cudaMemcpy(kVals_d, &kVals[kValuesMax], (numK-kValuesMax) * sizeof(struct kValues), cudaMemcpyHostToDevice);
  cuda_ret = cudaMemcpy(kVals_d, kVals, numK * sizeof(struct kValues), cudaMemcpyHostToDevice);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device ");

  // Launch kernel ----------------------------------------------------------

  dim3 dim_grid, dim_block;
  unsigned block, grid;
  block = PHIMAGBLOCK_SIZE;
  grid = numX / (PHIMAGBLOCK_SIZE);
  if( numX % (PHIMAGBLOCK_SIZE * grid)) 
    grid++;

  printf("\tBLOCK: %d\n\tGRID: %d\n\tnumX: %d\n\tnumK: %d\n", block, grid, numX, numK);

  dim_block.x = block;
  dim_block.y = 1;
  dim_block.z = 1;

  dim_grid.x = grid;
  dim_grid.y = 1;
  dim_grid.z = 1;

  ComputeQGPUKernel_3<<<dim_grid, dim_block>>>(numK, numX, kVals_d, x_d, y_d, z_d, Qr_d, Qi_d);
  // ComputeQGPUKernel_2<<<dim_grid, dim_block>>>(numK, numX, x_d, y_d, z_d, Qr_d, Qi_d);


  cuda_ret = cudaDeviceSynchronize();
  if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");


  cuda_ret = cudaMemcpy(Qr, Qr_d, numX * sizeof(float), cudaMemcpyDeviceToHost);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");
  cuda_ret = cudaMemcpy(Qi, Qi_d, numX * sizeof(float), cudaMemcpyDeviceToHost);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");

  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(z_d);
  cudaFree(Qr_d);
  cudaFree(Qi_d);
  cudaFree(kVals_d);
  // cudaFree(Kvalues_c);
}


// USING Kvalues_c AS A CONSTANT--------------------------------------------------------------------------------

// __global__ void ComputeQGPUKernel_2(int numK, int numX, struct kValues *kVals, float* x, float* y, float* z, float *Qr, float *Qi, int kValsCopies){
//   __shared__ float x_s[PHIMAGBLOCK_SIZE];
//   __shared__ float y_s[PHIMAGBLOCK_SIZE];
//   __shared__ float z_s[PHIMAGBLOCK_SIZE];
//   // Store this in cache memory, has a lot of resue ability
//   // __shared__ struct kValues kVals_s[numK];

//   unsigned int t = threadIdx.x;
//   unsigned int offset = (blockIdx.x*blockDim.x) + t;

  
//   // unsigned int i;
//   // unsigned int tmp_offset;
//   // unsigned int iterations;
//   // iterations = numK / PHIMAGBLOCK_SIZE;
//   // if(numK % (iterations*PHIMAGBLOCK_SIZE))
//   //   iterations++;
//   // for(i = 0; i < iterations; i++)
//   //   tmp_offset = (i*PHIMAGBLOCK_SIZE) + t;
//   //   if(tmp_offset < numK)
//   //     kVals_s[tmp_offset] = kVals[offset];
  

//   if(offset < numX){
//     x_s[t] = x[offset];
//     y_s[t] = y[offset];
//     z_s[t] = z[offset];
//     // kVals_s[t] = kVals[offset];

//     int indexK;
//     float Qracc = 0.0f;
//     float Qiacc = 0.0f;
//     float expArg = 0.0f;
//     float cosArg = 0.0f;
//     float sinArg = 0.0f;
//     float phi = 0.0f;

//     for (indexK = 0; indexK < numK; indexK++) {
//       // Generally, numX > numK

//       if(indexK < kValsCopies){ // Use constant memory
//         expArg = PIx2 * (Kvalues_c[indexK].Kx * x_s[t] + Kvalues_c[indexK].Ky * y_s[t] + Kvalues_c[indexK].Kz * z_s[t]);
//         phi = Kvalues_c[indexK].PhiMag;
//       } else { // Use global memory
//         expArg = PIx2 * (kVals[indexK-kValsCopies].Kx * x_s[t] + kVals[indexK-kValsCopies].Ky * y_s[t] + kVals[indexK-kValsCopies].Kz * z_s[t]);
//         phi = kVals[indexK-kValsCopies].PhiMag;
//       }

//       cosArg = cosf(expArg);
//       sinArg = sinf(expArg);

//       Qracc += phi * cosArg;
//       Qiacc += phi * sinArg;
//     }
//     Qr[offset] = Qracc;
//     Qi[offset] = Qiacc;
//   }
//   __syncthreads();
// }

// void ComputeQGPU_2(int numK, int numX, struct kValues *kVals, float* x, float* y, float* z, float *__restrict__ Qr, float *__restrict__ Qi){
//   cudaError_t cuda_ret;

//   float *x_d; //numX
//   float *y_d; //numX
//   float *z_d; //numX
//   float *Qr_d; //numX
//   float *Qi_d; //numX
  
//   struct kValues *kVals_d;

//   int kValsCopies = 0;
//   int kValsGlobal = 0;
//   if(numK < kValuesMax)
//     kValsCopies = numK;
//   else
//     kValsCopies = kValuesMax;
//   kValsGlobal = numK - kValsCopies;


//   // Allocate device variables ---------------------------------
//   cuda_ret = cudaMalloc((void**)&x_d, numX * sizeof(float));
//   if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
//   cuda_ret = cudaMalloc((void**)&y_d, numX * sizeof(float));
//   if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
//   cuda_ret = cudaMalloc((void**)&z_d, numX * sizeof(float));
//   if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
//   cuda_ret = cudaMalloc((void**)&Qr_d, numX * sizeof(float));
//   if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
//   cuda_ret = cudaMalloc((void**)&Qi_d, numX * sizeof(float));
//   if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");

//   if(kValsGlobal){
//     cuda_ret = cudaMalloc((void**)&kVals_d, kValsGlobal * sizeof(struct kValues));
//     if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
//   }


//   cuda_ret = cudaMemcpy(x_d, x, numX * sizeof(float), cudaMemcpyHostToDevice);
//   if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device ");
//   cuda_ret = cudaMemcpy(y_d, y, numX * sizeof(float), cudaMemcpyHostToDevice);
//   if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device ");
//   cuda_ret = cudaMemcpy(z_d, z, numX * sizeof(float), cudaMemcpyHostToDevice);
//   if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device ");
//   cuda_ret = cudaMemset(Qr_d, 0, numX * sizeof(float));
//   if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory ");
//   cuda_ret = cudaMemset(Qi_d, 0, numX * sizeof(float));
//   if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory ");



//   // Use constant memory for the first kValuesMax elements
//   cuda_ret = cudaMemcpyToSymbol(Kvalues_c, kVals, kValsCopies * sizeof(struct kValues), 0, cudaMemcpyHostToDevice);
//   if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device ");

//   // Use global memory for the [kValuesMax:numK] elements not already in constant memory
//   if(kValsGlobal){
//     cuda_ret = cudaMemcpy(kVals_d, &kVals[kValsGlobal], kValsGlobal * sizeof(struct kValues), cudaMemcpyHostToDevice);
//     if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device ");
//   }

//   // Launch kernel ----------------------------------------------------------

//   dim3 dim_grid, dim_block;
//   unsigned block, grid;
//   block = PHIMAGBLOCK_SIZE;
//   grid = numX / (PHIMAGBLOCK_SIZE);
//   if( numX % (PHIMAGBLOCK_SIZE * grid)) 
//     grid++;

//   printf("\tBLOCK: %d\n\tGRID: %d\n\tnumX: %d\n\tnumK: %d\n", block, grid, numX, numK);

//   dim_block.x = block;
//   dim_block.y = 1;
//   dim_block.z = 1;

//   dim_grid.x = grid;
//   dim_grid.y = 1;
//   dim_grid.z = 1;

//   ComputeQGPUKernel_2<<<dim_grid, dim_block>>>(numK, numX, kVals_d, x_d, y_d, z_d, Qr_d, Qi_d, kValsCopies);
//   // ComputeQGPUKernel_2<<<dim_grid, dim_block>>>(numK, numX, kVals_d, x_d, y_d, z_d, Qr_d, Qi_d);
//   // ComputeQGPUKernel_2<<<dim_grid, dim_block>>>(numK, numX, x_d, y_d, z_d, Qr_d, Qi_d);


//   cuda_ret = cudaDeviceSynchronize();
//   if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");


//   cuda_ret = cudaMemcpy(Qr, Qr_d, numX * sizeof(float), cudaMemcpyDeviceToHost);
//   if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");
//   cuda_ret = cudaMemcpy(Qi, Qi_d, numX * sizeof(float), cudaMemcpyDeviceToHost);
//   if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");

//   cudaFree(x_d);
//   cudaFree(y_d);
//   cudaFree(z_d);
//   cudaFree(Qr_d);
//   cudaFree(Qi_d);
//   if(kValsGlobal)
//     cudaFree(kVals_d);
// }


// NAIVE ComputeQGPU IMPLEMENTATION-----------------------------------------------------------------------------

__global__ void ComputeQGPUKernel(int numK, int numX, struct kValues *kVals, float* x, float* y, float* z, float *Qr, float *Qi){
  __shared__ float x_s[PHIMAGBLOCK_SIZE];
  __shared__ float y_s[PHIMAGBLOCK_SIZE];
  __shared__ float z_s[PHIMAGBLOCK_SIZE];
  // Store this in cache memory, has a lot of resue ability
  // __shared__ struct kValues kVals_s[numK];

  unsigned int t = threadIdx.x;
  unsigned int offset = (blockIdx.x*blockDim.x) + t;

  
  // unsigned int i;
  // unsigned int tmp_offset;
  // unsigned int iterations;
  // iterations = numK / PHIMAGBLOCK_SIZE;
  // if(numK % (iterations*PHIMAGBLOCK_SIZE))
  //   iterations++;
  // for(i = 0; i < iterations; i++)
  //   tmp_offset = (i*PHIMAGBLOCK_SIZE) + t;
  //   if(tmp_offset < numK)
  //     kVals_s[tmp_offset] = kVals[offset];
  

  if(offset < numX){
    x_s[t] = x[offset];
    y_s[t] = y[offset];
    z_s[t] = z[offset];
    // kVals_s[t] = kVals[offset];

    int indexK;
    float Qracc = 0.0f;
    float Qiacc = 0.0f;
    float expArg = 0.0f;
    float cosArg = 0.0f;
    float sinArg = 0.0f;

    for (indexK = 0; indexK < numK; indexK++) {
      // Generally, numX > numK
      expArg = PIx2 * (kVals[indexK].Kx * x_s[t] + kVals[indexK].Ky * y_s[t] + kVals[indexK].Kz * z_s[t]);

      cosArg = cosf(expArg);
      sinArg = sinf(expArg);

      float phi = kVals[indexK].PhiMag;
      Qracc += phi * cosArg;
      Qiacc += phi * sinArg;
    }
    Qr[offset] = Qracc;
    Qi[offset] = Qiacc;
  }
  __syncthreads();
}

void ComputeQGPU(int numK, int numX, struct kValues *kVals, float* x, float* y, float* z, float *__restrict__ Qr, float *__restrict__ Qi){
  cudaError_t cuda_ret;

  float *x_d; //numX
  float *y_d; //numX
  float *z_d; //numX
  float *Qr_d; //numX
  float *Qi_d; //numX
  
  struct kValues *kVals_d;

  // Allocate device variables ---------------------------------
  cuda_ret = cudaMalloc((void**)&x_d, numX * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
  cuda_ret = cudaMalloc((void**)&y_d, numX * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
  cuda_ret = cudaMalloc((void**)&z_d, numX * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
  cuda_ret = cudaMalloc((void**)&Qr_d, numX * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
  cuda_ret = cudaMalloc((void**)&Qi_d, numX * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
  cuda_ret = cudaMalloc((void**)&kVals_d, numK * sizeof(struct kValues));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");


  cuda_ret = cudaMemcpy(x_d, x, numX * sizeof(float), cudaMemcpyHostToDevice);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device ");
  cuda_ret = cudaMemcpy(y_d, y, numX * sizeof(float), cudaMemcpyHostToDevice);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device ");
  cuda_ret = cudaMemcpy(z_d, z, numX * sizeof(float), cudaMemcpyHostToDevice);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device ");
  cuda_ret = cudaMemset(Qr_d, 0, numX * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory ");
  cuda_ret = cudaMemset(Qi_d, 0, numX * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory ");
  cuda_ret = cudaMemcpy(kVals_d, kVals, numK * sizeof(struct kValues), cudaMemcpyHostToDevice);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device ");


  // Launch kernel ----------------------------------------------------------

  dim3 dim_grid, dim_block;
  unsigned block, grid;
  block = PHIMAGBLOCK_SIZE;
  grid = numX / (PHIMAGBLOCK_SIZE);
  if( numX % (PHIMAGBLOCK_SIZE * grid)) 
    grid++;

  printf("\tBLOCK: %d\n\tGRID: %d\n\tnumX: %d\n\tnumK: %d\n", block, grid, numX, numK);

  dim_block.x = block;
  dim_block.y = 1;
  dim_block.z = 1;

  dim_grid.x = grid;
  dim_grid.y = 1;
  dim_grid.z = 1;

  // ComputeQGPUKernel(numK, numX, kVals, x, y, z, Qr, Qi)
  ComputeQGPUKernel<<<dim_grid, dim_block>>>(numK, numX, kVals_d, x_d, y_d, z_d, Qr_d, Qi_d);


  cuda_ret = cudaDeviceSynchronize();
  if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");


  cuda_ret = cudaMemcpy(Qr, Qr_d, numX * sizeof(float), cudaMemcpyDeviceToHost);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");
  cuda_ret = cudaMemcpy(Qi, Qi_d, numX * sizeof(float), cudaMemcpyDeviceToHost);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");

  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(z_d);
  cudaFree(Qr_d);
  cudaFree(Qi_d);
  cudaFree(kVals_d);
}


//--------------------------------------------------------------------------------------------------------------


__global__ void ComputeCombined(){

}

__global__ void CreatekValsGPUKernel(int numk, struct kValues *kVals, float* kx, float* ky, float* kz, float* phiMag){
  unsigned int t = threadIdx.x;
  unsigned int offset = (blockIdx.x*blockDim.x) + t;

  if(offset < numk){
    kVals[offset].Kx = kx[offset];
    kVals[offset].Ky = ky[offset];
    kVals[offset].Kz = kz[offset];
    kVals[offset].PhiMag = phiMag[offset];
  }
}


void ComputeOnGPU(int numK, int numX, float* phiR, float* phiI, float* phiMag, float *kx, float *ky, float *kz, float* x, float* y, float* z, float *__restrict__ Qr, float *__restrict__ Qi){
  cudaError_t cuda_ret;

  float *phiR_d, *phiI_d, *phiMag_d; //numK
  float *kx_d, *ky_d, *kz_d; //numK
  struct kValues *kVals_d; //numK
  float *x_d, *y_d, *z_d, *Qr_d, *Qi_d; //numX


  // Allocate device variables ---------------------------------
  //// Used for ComputePhiMag
  cuda_ret = cudaMalloc((void**)&phiR_d, numK * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
  cuda_ret = cudaMalloc((void**)&phiI_d, numK * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
  cuda_ret = cudaMalloc((void**)&phiMag_d, numK * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");

  cuda_ret = cudaMemcpy(phiR_d, phiR, numK * sizeof(float), cudaMemcpyHostToDevice);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device ");
  cuda_ret = cudaMemcpy(phiI_d, phiI, numK * sizeof(float), cudaMemcpyHostToDevice);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device ");
  cuda_ret = cudaMemset(phiMag_d, 0, numK * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory ");


  //// Creating Kvalues
  cuda_ret = cudaMalloc((void**)&kVals_d, numK * sizeof(struct kValues));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");

  cuda_ret = cudaMalloc((void**)&kx_d, numK * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
  cuda_ret = cudaMalloc((void**)&ky_d, numK * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
  cuda_ret = cudaMalloc((void**)&kz_d, numK * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");

  cuda_ret = cudaMemcpy(kx_d, kx, numK * sizeof(float), cudaMemcpyHostToDevice);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device ");
  cuda_ret = cudaMemcpy(ky_d, ky, numK * sizeof(float), cudaMemcpyHostToDevice);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device ");
  cuda_ret = cudaMemcpy(kz_d, kz, numK * sizeof(float), cudaMemcpyHostToDevice);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device ");

  //// ComputingQ
  cuda_ret = cudaMalloc((void**)&x_d, numX * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
  cuda_ret = cudaMalloc((void**)&y_d, numX * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
  cuda_ret = cudaMalloc((void**)&z_d, numX * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
  cuda_ret = cudaMalloc((void**)&Qr_d, numX * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
  cuda_ret = cudaMalloc((void**)&Qi_d, numX * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");

  cuda_ret = cudaMemcpy(x_d, x, numX * sizeof(float), cudaMemcpyHostToDevice);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device ");
  cuda_ret = cudaMemcpy(y_d, y, numX * sizeof(float), cudaMemcpyHostToDevice);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device ");
  cuda_ret = cudaMemcpy(z_d, z, numX * sizeof(float), cudaMemcpyHostToDevice);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device ");
  cuda_ret = cudaMemset(Qr_d, 0, numX * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory ");
  cuda_ret = cudaMemset(Qi_d, 0, numX * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory ");


  // Launch kernel ----------------------------------------------------------

  //// Launching ComputePhiMag
  dim3 dim_grid, dim_block;
  unsigned block, grid;
  block = PHIMAGBLOCK_SIZE;
  grid = numK / (PHIMAGBLOCK_SIZE);
  if( numK % (PHIMAGBLOCK_SIZE * grid)) grid++;

  printf("\tBLOCK: %d\n\tGRID: %d\n", block, grid);

  dim_block.x = block;
  dim_block.y = 1;
  dim_block.z = 1;

  dim_grid.x = grid;
  dim_grid.y = 1;
  dim_grid.z = 1;

  ///// Launching ComputePhiMag
  ComputePhiMagGPUKernel<<<dim_grid, dim_block>>>(numK, phiR_d, phiI_d, phiMag_d);
  cuda_ret = cudaDeviceSynchronize();
  if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");


  //// Launching CreatingkValusGPU
  CreatekValsGPUKernel<<<dim_grid, dim_block>>>(numK, kVals_d, kx_d, ky_d, kz_d, phiMag_d);
  cuda_ret = cudaDeviceSynchronize();
  if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");

  //// Launching ComputeQGPU
  block = PHIMAGBLOCK_SIZE;
  grid = numX / (PHIMAGBLOCK_SIZE);
  if( numX % (PHIMAGBLOCK_SIZE * grid)) 
    grid++;

  printf("\tBLOCK: %d\n\tGRID: %d\n\tnumX: %d\n\tnumK: %d\n", block, grid, numX, numK);

  dim_block.x = block;
  dim_block.y = 1;
  dim_block.z = 1;

  dim_grid.x = grid;
  dim_grid.y = 1;
  dim_grid.z = 1;

  // ComputeQGPUKernel(numK, numX, kVals, x, y, z, Qr, Qi)
  ComputeQGPUKernel<<<dim_grid, dim_block>>>(numK, numX, kVals_d, x_d, y_d, z_d, Qr_d, Qi_d);
  cuda_ret = cudaDeviceSynchronize();
  if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");

  //// Returning Output 
  // Qr, Qi
  cuda_ret = cudaMemcpy(Qr, Qr_d, numX * sizeof(float), cudaMemcpyDeviceToHost);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");
  cuda_ret = cudaMemcpy(Qi, Qi_d, numX * sizeof(float), cudaMemcpyDeviceToHost);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");

  cudaFree(phiR_d);
  cudaFree(phiI_d);
  cudaFree(phiMag_d);
  cudaFree(kx_d);
  cudaFree(ky_d);
  cudaFree(kz_d);
  cudaFree(kVals_d);
  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(z_d);
  cudaFree(Qr_d);
  cudaFree(Qi_d);
}

void StreamComputeOnGPU(int numK, int numX, float* phiR, float* phiI, float* phiMag, float *kx, float *ky, float *kz, float* x, float* y, float* z, float *__restrict__ Qr, float *__restrict__ Qi){
  cudaError_t cuda_ret;
  cudaStream_t stream0, stream1, stream2;
  cudaStreamCreate(&stream0);
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  float *phiR_d, *phiI_d, *phiMag_d; //numK
  float *kx_d, *ky_d, *kz_d; //numK
  struct kValues *kVals_d; //numK
  float *x_d, *y_d, *z_d, *Qr_d, *Qi_d; //numX


  // Allocate device variables ---------------------------------
  //// Used for ComputePhiMag
  cuda_ret = cudaMalloc((void**)&phiR_d, numK * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
  cuda_ret = cudaMalloc((void**)&phiI_d, numK * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
  cuda_ret = cudaMalloc((void**)&phiMag_d, numK * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");

  cuda_ret = cudaMemcpyAsync(phiR_d, phiR, numK * sizeof(float), cudaMemcpyHostToDevice, stream0);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device ");
  cuda_ret = cudaMemcpyAsync(phiI_d, phiI, numK * sizeof(float), cudaMemcpyHostToDevice, stream0);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device ");
  cuda_ret = cudaMemset(phiMag_d, 0, numK * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory ");


  //// Creating Kvalues
  cuda_ret = cudaMalloc((void**)&kVals_d, numK * sizeof(struct kValues));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");

  cuda_ret = cudaMalloc((void**)&kx_d, numK * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
  cuda_ret = cudaMalloc((void**)&ky_d, numK * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
  cuda_ret = cudaMalloc((void**)&kz_d, numK * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");

  cuda_ret = cudaMemcpyAsync(kx_d, kx, numK * sizeof(float), cudaMemcpyHostToDevice, stream1);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device ");
  cuda_ret = cudaMemcpyAsync(ky_d, ky, numK * sizeof(float), cudaMemcpyHostToDevice, stream1);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device ");
  cuda_ret = cudaMemcpyAsync(kz_d, kz, numK * sizeof(float), cudaMemcpyHostToDevice, stream1);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device ");

  //// ComputingQ
  cuda_ret = cudaMalloc((void**)&x_d, numX * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
  cuda_ret = cudaMalloc((void**)&y_d, numX * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
  cuda_ret = cudaMalloc((void**)&z_d, numX * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
  cuda_ret = cudaMalloc((void**)&Qr_d, numX * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
  cuda_ret = cudaMalloc((void**)&Qi_d, numX * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");

  cuda_ret = cudaMemcpyAsync(x_d, x, numX * sizeof(float), cudaMemcpyHostToDevice, stream2);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device ");
  cuda_ret = cudaMemcpyAsync(y_d, y, numX * sizeof(float), cudaMemcpyHostToDevice, stream2);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device ");
  cuda_ret = cudaMemcpyAsync(z_d, z, numX * sizeof(float), cudaMemcpyHostToDevice, stream2);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device ");
  cuda_ret = cudaMemset(Qr_d, 0, numX * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory ");
  cuda_ret = cudaMemset(Qi_d, 0, numX * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory ");


  // Launch kernel ----------------------------------------------------------

  //// Launching ComputePhiMag
  dim3 dim_grid, dim_block;
  unsigned block, grid;
  block = PHIMAGBLOCK_SIZE;
  grid = numK / (PHIMAGBLOCK_SIZE);
  if( numK % (PHIMAGBLOCK_SIZE * grid)) grid++;

  printf("\tBLOCK: %d\n\tGRID: %d\n", block, grid);

  dim_block.x = block;
  dim_block.y = 1;
  dim_block.z = 1;

  dim_grid.x = grid;
  dim_grid.y = 1;
  dim_grid.z = 1;

  ///// Launching ComputePhiMag
  ComputePhiMagGPUKernel<<<dim_grid, dim_block, 0, stream0>>>(numK, phiR_d, phiI_d, phiMag_d);
  cuda_ret = cudaDeviceSynchronize();
  if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");




  //// Launching CreatingkValusGPU
  CreatekValsGPUKernel<<<dim_grid, dim_block, 0, stream1>>>(numK, kVals_d, kx_d, ky_d, kz_d, phiMag_d);
  cuda_ret = cudaDeviceSynchronize();
  if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");




  //// Launching ComputeQGPU
  block = PHIMAGBLOCK_SIZE;
  grid = numX / (PHIMAGBLOCK_SIZE);
  if( numX % (PHIMAGBLOCK_SIZE * grid)) 
    grid++;

  printf("\tBLOCK: %d\n\tGRID: %d\n\tnumX: %d\n\tnumK: %d\n", block, grid, numX, numK);

  dim_block.x = block;
  dim_block.y = 1;
  dim_block.z = 1;

  dim_grid.x = grid;
  dim_grid.y = 1;
  dim_grid.z = 1;

  // ComputeQGPUKernel(numK, numX, kVals, x, y, z, Qr, Qi)
  ComputeQGPUKernel<<<dim_grid, dim_block, 0, stream2>>>(numK, numX, kVals_d, x_d, y_d, z_d, Qr_d, Qi_d);
  cuda_ret = cudaDeviceSynchronize();
  if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");

  //// Returning Output 
  // Qr, Qi
  cuda_ret = cudaMemcpy(Qr, Qr_d, numX * sizeof(float), cudaMemcpyDeviceToHost);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");
  cuda_ret = cudaMemcpy(Qi, Qi_d, numX * sizeof(float), cudaMemcpyDeviceToHost);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");

  cudaStreamDestroy(stream0);
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);

  cudaFree(phiR_d);
  cudaFree(phiI_d);
  cudaFree(phiMag_d);
  cudaFree(kx_d);
  cudaFree(ky_d);
  cudaFree(kz_d);
  cudaFree(kVals_d);
  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(z_d);
  cudaFree(Qr_d);
  cudaFree(Qi_d);

}

//--------------------------------------------------------------------------------------------------------------























