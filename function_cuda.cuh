#pragma once
#include "function.h"

#ifdef CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define TILE_WIDTH 16

namespace LSQ_CUDA
{
	__global__ void transpose_cuda(double* T, double* A_T, int raw, int col);
	__global__ void inverse_cuda(double* T, double* A_inv, int raw, int col);
	__global__ void multiply_cuda(double* A, double* B, double* C, int rowA, int colA, int rowB, int colB);

}
#endif // 