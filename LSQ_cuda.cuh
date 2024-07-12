#pragma once
#include "function.h"

#ifdef CUDA
#include "function_cuda.cuh"

namespace LSQ_CUDA
{
	std::vector<double> normal_equation(const std::vector<std::vector<double>>& M, const std::vector<double>& H);
}
#endif // CUDA