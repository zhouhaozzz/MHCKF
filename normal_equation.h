#pragma once
#include "function.h"
// ��С���˷�������Իع�����

class normal_equation
{
public:
	normal_equation();
	~normal_equation();

	std::vector<double> linear_least_squares(const std::vector<std::vector<double>>& M, const std::vector<double>& H);
};

