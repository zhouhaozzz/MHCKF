#include "LSQ_cuda.cuh"

#ifdef CUDA
std::vector<double> LSQ_CUDA::normal_equation(const std::vector<std::vector<double>>& M, const std::vector<double>& H)
{
    // 获取当前时间作为起始时间点
    auto start = std::chrono::high_resolution_clock::now(); 
    
    int m = M.size();
    int n = M[0].size();    
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + 15) / 16, (m + 15) / 16);
    double* d_M;
    double* d_M_T;
    double* d_H;
    double* A;
    cudaMalloc(&d_M, n * m * sizeof(double));
    cudaMalloc(&d_H, m * sizeof(double));
    cudaMalloc(&d_M_T, m * n * sizeof(double));
    cudaMalloc(&A, m * n * sizeof(double));

    cudaMemcpy(d_M, M.data(), n * m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_H, H.data(), m * sizeof(double), cudaMemcpyHostToDevice);

    LSQ_CUDA::transpose_cuda << <numBlocks, threadsPerBlock >> > (d_M, d_M_T, m, n);
    LSQ_CUDA::multiply_cuda << <numBlocks, threadsPerBlock >> > (d_M_T, d_M, A, m, n, n, m);

    //LSQ_CUDA::inverse_cuda << <numBlocks, threadsPerBlock >> > (d_M, d_M_T, m, n);


    cudaFree(d_M);
    cudaFree(d_H);

    // 获取当前时间作为结束时间点
    auto end = std::chrono::high_resolution_clock::now();
    // 计算函数执行时间
    std::chrono::duration<double> duration = end - start;
    std::cout << "cuda execution time: " << duration.count() << " seconds" << std::endl;

    return H;
}
#endif // CUDA
