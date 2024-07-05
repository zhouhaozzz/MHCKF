#pragma once
#include "function.h"

class colPivHouseholderQr
{
public:
	colPivHouseholderQr();
	~colPivHouseholderQr();
	std::vector<double> linear_least_squares(const std::vector<std::vector<double>>& M, const std::vector<double>& K);
};

