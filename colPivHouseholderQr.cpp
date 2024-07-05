#include "colPivHouseholderQr.h"

colPivHouseholderQr::colPivHouseholderQr()
{
}

colPivHouseholderQr::~colPivHouseholderQr()
{
}

std::vector<double> colPivHouseholderQr::linear_least_squares(const std::vector<std::vector<double>>& M, const std::vector<double>& K)
{
    // ��ȡ��ǰʱ����Ϊ��ʼʱ���
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

    std::vector<double> x = std::vector<double>(heat_num);
    for (int i = 0; i < heat_num; i++)
    {
        x[i] = H(i);
    }

    // ��ȡ��ǰʱ����Ϊ����ʱ���
    auto end = std::chrono::high_resolution_clock::now();

    // ���㺯��ִ��ʱ��
    std::chrono::duration<double> duration = end - start;
    std::cout << "colPivHouseholderQr execution time: " << duration.count() << " seconds" << std::endl;
    return x;
}
