#include <iostream>
#include <math.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string.h>
#include <stdio.h>
#include <signal.h>
#include <stdlib.h>
#include <time.h>
#include <iomanip>
#include <cstring>
#include <chrono>
#include <cmath>
#include <limits>
#include <algorithm>
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <mpi.h>

using namespace std;
using namespace Eigen;

#define BLOCK_SIZE 256

std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& A);
std::vector<double> multiplyTranspose(const std::vector<std::vector<double>>& A, const std::vector<double>& x);
std::vector<std::vector<double>> inverse(const std::vector<std::vector<double>>& A);
std::vector<std::vector<double>> multiply(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B);
std::vector<double> multiply(const std::vector<std::vector<double>>& A, const std::vector<double>& x);
std::vector<double> subtract(const std::vector<double>& a, const std::vector<double>& b);
std::vector<double> add(const std::vector<double>& a, const std::vector<double>& b);
double dotProduct(const std::vector<double>& a, const std::vector<double>& b);
std::vector<double> multiply(const std::vector<double>& a, double scalar);
std::vector<std::vector<double>> multiply(const std::vector<std::vector<double>>& A, double scalar);
void printVector(const std::vector<double>& v);
void printMatrix(const std::vector<std::vector<double>>& A);
std::vector<double> clipToNonNegative(const std::vector<double>& a);
void luDecomposition(const std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& L, std::vector<std::vector<double>>& U);
std::vector<double> forwardSubstitution(const std::vector<std::vector<double>>& L, const std::vector<double>& b);
std::vector<double> backSubstitution(const std::vector<std::vector<double>>& U, const std::vector<double>& y);

