#pragma once
#include "function.h"

// 改进的Gram-Schmidt QR分解

class QR_MGS
{
public:
	QR_MGS();
	~QR_MGS();

	void modifiedGramSchmidt(const std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& Q, std::vector<std::vector<double>>& R);
	std::vector<double> backSubstitution(const std::vector<std::vector<double>>& R, const std::vector<double>& y);
	std::vector<double> linear_least_squares(const std::vector<std::vector<double>>& A, const std::vector<double>& b);


};

