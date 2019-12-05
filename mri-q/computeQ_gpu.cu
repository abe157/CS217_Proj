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

#define PHIMAGBLOCK_SIZE 512 // 512 or 192

struct kValues {
  float Kx;
  float Ky;
  float Kz;
  float PhiMag;
};



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


// (TODO) This is actually slower????
inline
void 
ComputePhiMagGPU(int numK, float* phiR, float* phiI, float* phiMag) {
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

inline
void 
ComputePhiMagCPU(int numK, float* phiR, float* phiI, float* phiMag) {
  // Baseline Implementation
  int indexK = 0;
  for (indexK = 0; indexK < numK; indexK++) {
    float real = phiR[indexK];
    float imag = phiI[indexK];
    phiMag[indexK] = real*real + imag*imag;
  }
}



// __global__ void ComputeQGPUKernel_Risky(int numK, int numX, struct kValues *kVals, float* x, float* y, float* z, float *__restrict__ Qr, float *__restrict__ Qi){
// }
// void ComputeQGPU_Risky(int numK, int numX, struct kValues *kVals, float* x, float* y, float* z, float *__restrict__ Qr, float *__restrict__ Qi){
// }

__global__ void ComputeQGPUKernel(int numK, int numX, struct kValues *kVals, float* x, float* y, float* z, float *__restrict__ Qr, float *__restrict__ Qi){
  __shared__ float x_s[PHIMAGBLOCK_SIZE];
  __shared__ float y_s[PHIMAGBLOCK_SIZE];
  __shared__ float z_s[PHIMAGBLOCK_SIZE];
  __shared__ struct kValues kVals_s[PHIMAGBLOCK_SIZE];

  unsigned int t = threadIdx.x;
  unsigned int offset = (blockIdx.x*PHIMAGBLOCK_SIZE) + t;


  if(offset < numX){
    x_s[t] = x[offset];
    y_s[t] = y[offset];
    z_s[t] = z[offset];
    kVals_s[t] = kVals[offset];

    int indexK;
    float Qracc = 0.0f;
    float Qiacc = 0.0f;
    float expArg;
    float cosArg;
    float sinArg;

    for (indexK = 0; indexK < numK; indexK++) {
      expArg = PIx2 * (kVals_s[indexK].Kx * x_s[offset] + kVals_s[indexK].Ky * y_s[offset] + kVals_s[indexK].Kz * z_s[offset]);

      cosArg = cosf(expArg);
      sinArg = sinf(expArg);

      float phi = kVals[indexK].PhiMag;
      Qracc += phi * cosArg;
      Qiacc += phi * sinArg;
    }
    Qr[offset] = Qracc;
    Qi[offset] = Qiacc;

  }

}

void ComputeQGPU(int numK, int numX, struct kValues *kVals, float* x, float* y, float* z, float *__restrict__ Qr, float *__restrict__ Qi){
  cudaError_t cuda_ret;

  float *x_d, *y_d, *z_d, *Qr_d, *Qi_d;
  struct kValues *kVals_d;

  // Allocate device variables ---------------------------------
  cuda_ret = cudaMalloc((void**)&x_d, numX * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
  cuda_ret = cudaMalloc((void**)&y_d, numX * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
  cuda_ret = cudaMalloc((void**)&z_d, numX * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
  cuda_ret = cudaMalloc((void**)&Qr_d, numK * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
  cuda_ret = cudaMalloc((void**)&Qi_d, numK * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");
  cuda_ret = cudaMalloc((void**)&kVals_d, numK * sizeof(struct kValues));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory ");


  cuda_ret = cudaMemcpy(x_d, x, numX * sizeof(float), cudaMemcpyHostToDevice);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device ");
  cuda_ret = cudaMemcpy(y_d, y, numX * sizeof(float), cudaMemcpyHostToDevice);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device ");
  cuda_ret = cudaMemcpy(z_d, z, numX * sizeof(float), cudaMemcpyHostToDevice);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device ");
  cuda_ret = cudaMemset(Qr_d, 0, numK * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory ");
  cuda_ret = cudaMemset(Qi_d, 0, numK * sizeof(float));
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

  printf("\tBLOCK: %d\n\tGRID: %d\n", block, grid);

  dim_block.x = block;
  dim_block.y = 1;
  dim_block.z = 1;

  dim_grid.x = grid;
  dim_grid.y = 1;
  dim_grid.z = 1;

  ComputeQGPUKernel<<<dim_grid, dim_block>>>(numK, numX, kVals_d, x_d, y_d, z_d, Qr_d, Qi_d);




  cuda_ret = cudaMemcpy(Qr, Qr_d, numK * sizeof(float), cudaMemcpyDeviceToHost);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host in naive reduction.");
  cuda_ret = cudaMemcpy(Qi, Qi_d, numK * sizeof(float), cudaMemcpyDeviceToHost);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host in naive reduction.");

  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(z_d);
  cudaFree(Qr_d);
  cudaFree(Qi_d);
  cudaFree(kVals_d);

  cuda_ret = cudaDeviceSynchronize();
  if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");

}


inline
void
ComputeQCPU(int numK, int numX, struct kValues *kVals, float* x, float* y, float* z, float *__restrict__ Qr, float *__restrict__ Qi) {
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

void createDataStructsCPU(int numK, int numX, float** phiMag, float** Qr, float** Qi){
  *phiMag = (float* ) memalign(16, numK * sizeof(float));
  *Qr = (float*) memalign(16, numX * sizeof (float));
  memset((void *)*Qr, 0, numX * sizeof(float));
  *Qi = (float*) memalign(16, numX * sizeof (float));
  memset((void *)*Qi, 0, numX * sizeof(float));
}





















