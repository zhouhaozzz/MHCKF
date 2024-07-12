#pragma once

#ifdef CUDA
#include "function_cuda.cuh"

__global__ void LSQ_CUDA::transpose_cuda(double* T, double* A_T, int raw, int col) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < raw && j < col) {
		T[j * raw + i] = A_T[i * col + j];
	}
}

__global__ void LSQ_CUDA::multiply_cuda(double* A, double* B, double* C, int rowA, int colA, int rowB, int colB)
{
    __shared__ double sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ double sB[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    double Cvalue = 0.0;

    for (int t = 0; t < (colA - 1) / TILE_WIDTH + 1; ++t) {
        if (Row < rowA && t * TILE_WIDTH + tx < colA)
            sA[ty][tx] = A[Row * colA + t * TILE_WIDTH + tx];
        else
            sA[ty][tx] = 0.0;
        if (Col < colB && t * TILE_WIDTH + ty < rowB)
            sB[ty][tx] = B[(t * TILE_WIDTH + ty) * colB + Col];
        else
            sB[ty][tx] = 0.0;
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i)
            Cvalue += sA[ty][i] * sB[i][tx];
        __syncthreads();
    }

    if (Row < rowA && Col < colB)
        C[Row * colB + Col] = Cvalue;
}

#endif // CUDA