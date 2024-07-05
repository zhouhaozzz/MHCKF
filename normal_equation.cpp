#include "normal_equation.h"

normal_equation::normal_equation()
{
}

normal_equation::~normal_equation()
{
}

std::vector<double> normal_equation::linear_least_squares(const std::vector<std::vector<double>>& M, const std::vector<double>& H) 
{
	// 获取当前时间作为起始时间点
	auto start = std::chrono::high_resolution_clock::now();

	std::vector<std::vector<double>> MT = transpose(M);
	std::vector<std::vector<double>> A = multiply(MT, M);
	std::vector<double> b = multiply(MT, H);

	std::vector<std::vector<double>> invA = inverse(A);
	std::vector<double> x = multiply(invA, b);

	// 获取当前时间作为结束时间点
	auto end = std::chrono::high_resolution_clock::now();

	// 计算函数执行时间
	std::chrono::duration<double> duration = end - start;
	std::cout << "normal_equation execution time: " << duration.count() << " seconds" << std::endl;

	return x;
}