#pragma once

#include "function.h"
#include "QR_MGS.h"
#include "normal_equation.h"

class NLSS
{
public:
	NLSS();
	~NLSS();

	int maxIter = 10000;
	double tol = 1e-6;

	QR_MGS* qr_mgs = new QR_MGS();
	normal_equation* n_e = new normal_equation();

	std::vector<double> conjugateGradient(const std::vector<std::vector<double>>& A, const std::vector<double>& b, const std::vector<double>& x0);
	std::vector<double> linear_least_squares(const std::vector<std::vector<double>>& A, const std::vector<double>& b);

	std::vector<double> spark_nlss(const std::vector<std::vector<double>>& A, const std::vector<double>& b);
	std::vector<double> lsqnonneg(const std::vector<std::vector<double>>& A, const std::vector<double>& b);
};


