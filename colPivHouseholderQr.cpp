#include "colPivHouseholderQr.h"

colPivHouseholderQr::colPivHouseholderQr()
{
}

colPivHouseholderQr::~colPivHouseholderQr()
{
}

std::vector<double> colPivHouseholderQr::linear_least_squares(const std::vector<std::vector<double>>& M, const std::vector<double>& K)
{
    // 获取当前时间作为起始时间点
    auto start = std::chrono::high_resolution_clock::now(); 
    
    int datax_num = M.size();
    int heat_num = M[0].size();

    Eigen::MatrixXd m = Eigen::MatrixXd::Zero(datax_num, heat_num);
    for (int i = 0; i < datax_num; i++)
    {
        for (int j = 0; j < heat_num; j++)
        {
            m(i, j) = M[i][j];
        }
    }

    Eigen::VectorXd init_face = Eigen::VectorXd::Zero(datax_num);
    Eigen::VectorXd k = Eigen::VectorXd::Zero(datax_num);
    for (int i = 0; i < datax_num; i++)
    {
        k(i) = K[i];
    }
    
    
    Eigen::VectorXd H;
    H = m.colPivHouseholderQr().solve(k);

    //H = lsqnonneg(m, k);

    std::vector<double> x = std::vector<double>(heat_num);
    for (int i = 0; i < heat_num; i++)
    {
        x[i] = H(i);
    }

    // 获取当前时间作为结束时间点
    auto end = std::chrono::high_resolution_clock::now();

    // 计算函数执行时间
    std::chrono::duration<double> duration = end - start;
    std::cout << "colPivHouseholderQr execution time: " << duration.count() << " seconds" << std::endl;
    return x;
}



VectorXd colPivHouseholderQr::lsqnonneg(const MatrixXd& A, const VectorXd& b) {
    int n = A.cols();
    VectorXd x = VectorXd::Zero(n);
    VectorXi passiveSet = VectorXi::Zero(n);
    VectorXd gradient = A.transpose() * (b - A * x);
    VectorXd z = VectorXd::Zero(n);

    while (gradient.maxCoeff() > 1e-10) {
        int t = -1;
        double maxGradient = -1;
        for (int i = 0; i < n; ++i) {
            if (passiveSet[i] == 0 && gradient[i] > maxGradient) {
                maxGradient = gradient[i];
                t = i;
            }
        }

        if (t == -1) break;

        passiveSet[t] = 1;

        bool converged = false;
        while (!converged) {
            VectorXi activeSetIndices;
            int activeSetSize = 0;
            for (int i = 0; i < n; ++i) {
                if (passiveSet[i] == 1) {
                    activeSetIndices.conservativeResize(activeSetSize + 1);
                    activeSetIndices[activeSetSize++] = i;
                }
            }

            MatrixXd A_active(activeSetSize, activeSetSize);
            VectorXd b_active(activeSetSize);
            for (int i = 0; i < activeSetSize; ++i) {
                for (int j = 0; j < activeSetSize; ++j) {
                    A_active(i, j) = A.col(activeSetIndices[i]).dot(A.col(activeSetIndices[j]));
                }
                b_active[i] = A.col(activeSetIndices[i]).dot(b);
            }

            VectorXd z_active = A_active.ldlt().solve(b_active);
            z.setZero();
            for (int i = 0; i < activeSetSize; ++i) {
                z[activeSetIndices[i]] = z_active[i];
            }

            if ((z.array() >= 0).all()) {
                x = z;
                converged = true;
            }
            else {
                double alpha = numeric_limits<double>::infinity();
                for (int i = 0; i < n; ++i) {
                    if (passiveSet[i] == 1 && z[i] <= 0) {
                        alpha = min(alpha, x[i] / (x[i] - z[i]));
                    }
                }
                x = x + alpha * (z - x);
                for (int i = 0; i < n; ++i) {
                    if (passiveSet[i] == 1 && abs(x[i]) < 1e-10) {
                        passiveSet[i] = 0;
                    }
                }
            }
        }

        gradient = A.transpose() * (b - A * x);
    }

    return x;
}