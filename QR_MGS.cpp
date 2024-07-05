#include "QR_MGS.h"

QR_MGS::QR_MGS()
{
}

QR_MGS::~QR_MGS()
{
}

// �Ľ���Gram-Schmidt QR�ֽ�
void QR_MGS::modifiedGramSchmidt(const std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& Q, std::vector<std::vector<double>>& R) 
{
    int rows = A.size();
    int cols = A[0].size();

    Q = std::vector<std::vector<double>>(rows, std::vector<double>(cols, 0));
    R = std::vector<std::vector<double>>(cols, std::vector<double>(cols, 0));

    for (int k = 0; k < cols; ++k) {
        std::vector<double> qk(rows);
        for (int i = 0; i < rows; ++i) {
            qk[i] = A[i][k];
        }

        for (int j = 0; j < k; ++j) {
            std::vector<double> qj(rows);
            for (int i = 0; i < rows; ++i) {
                qj[i] = Q[i][j];
            }
            double rjk = dotProduct(qj, qk);
            R[j][k] = rjk;
            std::vector<double> subtracted = multiply(qj, rjk);
            qk = subtract(qk, subtracted);
        }

        double norm = std::sqrt(dotProduct(qk, qk));
        R[k][k] = norm;

        for (int i = 0; i < rows; ++i) {
            Q[i][k] = qk[i] / norm;
        }
    }
}

// �ش�����������Ǿ��󷽳� R x = y
std::vector<double> QR_MGS::backSubstitution(const std::vector<std::vector<double>>& R, const std::vector<double>& y) 
{
    int n = R.size();
    std::vector<double> x(n);

    for (int i = n - 1; i >= 0; --i) {
        x[i] = y[i];
        for (int j = i + 1; j < n; ++j) {
            x[i] -= R[i][j] * x[j];
        }
        x[i] /= R[i][i];
    }

    return x;
}

// ��С���˷���⺯��
std::vector<double> QR_MGS::linear_least_squares(const std::vector<std::vector<double>>& A, const std::vector<double>& b) 
{
    // ��ȡ��ǰʱ����Ϊ��ʼʱ���
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<double>> Q, R;
    modifiedGramSchmidt(A, Q, R);

    std::vector<std::vector<double>> Qt = transpose(Q);
    std::vector<double> Qtb = multiply(Qt, b);
    std::vector<double> x = backSubstitution(R, Qtb);

    // ��ȡ��ǰʱ����Ϊ����ʱ���
    auto end = std::chrono::high_resolution_clock::now();
    // ���㺯��ִ��ʱ��
    std::chrono::duration<double> duration = end - start;
    std::cout << "QR_MGS execution time: " << duration.count() << " seconds" << std::endl;

    return x;
}