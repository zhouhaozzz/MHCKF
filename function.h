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

using namespace std;

std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& A);
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

//template <typename TYPE> void destory_1d(TYPE*& data)
//{
//    if (data == nullptr) return;
//    delete[] data;
//    data = nullptr;
//}
//
//template <typename TYPE> void destory_2d(TYPE**& data, int N)
//{
//    if (data == nullptr) return;
//    for (int i = 0; i < N; i++)
//    {
//        delete[] data[i];
//    }
//    delete[] data;
//    data = nullptr;
//}
//
//template <typename TYPE> void destory_3d(TYPE**& data, int N1, int N2)
//{
//    if (data == nullptr) return;
//    for (int i = 0; i < N1; ++i) {
//        for (int j = 0; j < N2; ++j) {
//            delete[] data[i][j];
//        }
//        delete[] data[i];
//    }
//    delete[] data;
//    data = nullptr;
//}
//
//template <typename TYPE> void create_1d(TYPE*& data, int N)
//{
//    data = new TYPE[N];
//}
//
//template <typename TYPE> void create_2d(TYPE**& data, int N1, int N2) {
//    data = new TYPE * [N1];
//    for (int i = 0; i < N1; i++) {
//        data[i] = new TYPE[N2];
//    }
//}
//
//template <typename TYPE> void create_3d(TYPE***& data, int N1, int N2, int N3) {
//    data = new TYPE * *[N1];
//    for (int i = 0; i < N1; i++) {
//        data[i] = new TYPE * [N2];
//        for (int j = 0; j < N2; j++)
//        {
//            data[i][j] = new TYPE[N3];
//        }
//    }
//}