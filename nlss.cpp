#include "nlss.h"

NLSS::NLSS()
{
}

NLSS::~NLSS()
{

}
// �����ݶȷ���⺯��
std::vector<double> NLSS::conjugateGradient(const std::vector<std::vector<double>>& A, const std::vector<double>& b, const std::vector<double>& x0) {
    std::vector<double> x = x0;
    std::vector<double> r = subtract(b, multiply(A, x));
    std::vector<double> p = r;
    double rsold = dotProduct(r, r);

    for (int i = 0; i < maxIter; ++i) {
        std::vector<double> Ap = multiply(A, p);
        double alpha = rsold / dotProduct(p, Ap);
        x = add(x, multiply(p, alpha));
        r = subtract(r, multiply(Ap, alpha));
        double rsnew = dotProduct(r, r);
        if (std::sqrt(rsnew) < tol) {
            break;
        }
        p = add(r, multiply(p, rsnew / rsold));
        rsold = rsnew;
    }

    return x;
}

// �Ǹ���С���˷���⺯��
std::vector<double> NLSS::linear_least_squares(const std::vector<std::vector<double>>& A, const std::vector<double>& b) {
    // ��ȡ��ǰʱ����Ϊ��ʼʱ���
    auto start = std::chrono::high_resolution_clock::now();

    int m = A.size();
    int n = A[0].size();
    std::vector<double> x(n, 0.0); // ��ʼ��������
    std::vector<double> grad(n);   // �ݶ�����

    std::vector<std::vector<double>> At = transpose(A);         // ���� A ��ת��
    std::vector<std::vector<double>> AtA = multiply(At, A);     // ���� A^T * A
    std::vector<double> Atb = multiply(At, b);     // ���� A^T * b

    for (int iter = 0; iter < maxIter; ++iter) {
        // �����ݶ� grad = A^T * (A * x - b)
        std::vector<double> Ax = multiply(A, x);
        std::vector<double> r = subtract(Ax, b);
        grad = multiplyTranspose(A, r);

        // �������
        double gradNorm = dotProduct(grad, grad);
        if (gradNorm < tol) {
            break;
        }

        // ͶӰ�ݶȷ�
        std::vector<double> xOld = x;
        x = subtract(x, multiply(grad, 1.0 / (iter + 1))); // ����С����
        x = clipToNonNegative(x);

        // �����ݶȷ������ڲ��Ż�
        std::vector<double> dx = subtract(x, xOld);
        x = conjugateGradient(AtA, Atb, x);

        // ͶӰ���Ǹ���ռ�
        x = clipToNonNegative(x);
    }

    // ��ȡ��ǰʱ����Ϊ����ʱ���
        auto end = std::chrono::high_resolution_clock::now();
    // ���㺯��ִ��ʱ��
    std::chrono::duration<double> duration = end - start;
    std::cout << "NLSS execution time: " << duration.count() << " seconds" << std::endl;

    return x;
}

// Function to solve the non-negative least squares problem
std::vector<double> NLSS::lsqnonneg(const std::vector<std::vector<double>>& A, const std::vector<double>& b) {
    // ��ȡ��ǰʱ����Ϊ��ʼʱ���
    auto start = std::chrono::high_resolution_clock::now();

    size_t n = A[0].size();
    std::vector<double> x(n, 0.0); // Initialize solution vector x to zero
    std::vector<int> passiveSet(n, 0); // Initialize passive set to zero
    std::vector<std::vector<double>> A_t = transpose(A); // Transpose of A
    std::vector<double> gradient = multiply(A_t, subtract(b, multiply(A, x))); // Compute initial gradient
    std::vector<double> z(n, 0.0); // Temporary vector for the solution

    while (*std::max_element(gradient.begin(), gradient.end()) > 1e-10) { // While the maximum gradient is greater than tolerance
        int t = -1;
        double maxGradient = -1;

        // Find the variable with the maximum positive gradient
        for (size_t i = 0; i < n; ++i) {
            if (passiveSet[i] == 0 && gradient[i] > maxGradient) {
                maxGradient = gradient[i];
                t = i;
            }
        }

        if (t == -1) break; // If no valid variable is found, break the loop

        passiveSet[t] = 1; // Add the variable to the passive set

        bool converged = false;
        while (!converged) {
            std::vector<size_t> activeSetIndices;
            for (size_t i = 0; i < n; ++i) {
                if (passiveSet[i] == 1) {
                    activeSetIndices.push_back(i);
                }
            }

            // Construct sub-matrix and sub-vector for the active set
            std::vector<std::vector<double>> A_active(activeSetIndices.size(), std::vector<double>(activeSetIndices.size(), 0.0));
            std::vector<double> b_active(activeSetIndices.size(), 0.0);
            for (size_t i = 0; i < activeSetIndices.size(); ++i) {
                for (size_t j = 0; j < activeSetIndices.size(); ++j) {
                    A_active[i][j] = dotProduct(A_t[activeSetIndices[i]], A_t[activeSetIndices[j]]);
                }
                b_active[i] = dotProduct(A_t[activeSetIndices[i]], b);
            }

            // Solve the sub-problem
            // LU�ֽ���������Է�����

            std::vector<std::vector<double>> L, U;
            luDecomposition(A_active, L, U);
            std::vector<double> y = forwardSubstitution(L, b_active);
            std::vector<double> z_active = backSubstitution(U, y);

            z = std::vector<double>(n, 0.0);
            for (size_t i = 0; i < activeSetIndices.size(); ++i) {
                z[activeSetIndices[i]] = z_active[i];
            }

            // Check if all variables in the active set are non-negative
            if (all_of(z.begin(), z.end(), [](double zi) { return zi >= 0; })) {
                x = z;
                converged = true;
            }
            else {
                double alpha = numeric_limits<double>::infinity();
                for (size_t i = 0; i < n; ++i) {
                    if (passiveSet[i] == 1 && z[i] <= 0) {
                        alpha = min(alpha, x[i] / (x[i] - z[i]));
                    }
                }
                x = add(x, multiply(subtract(z, x), alpha));
                for (size_t i = 0; i < n; ++i) {
                    if (passiveSet[i] == 1 && abs(x[i]) < 1e-10) {
                        passiveSet[i] = 0;
                    }
                }
            }
        }

        // Recompute the gradient
        gradient = multiply(A_t, subtract(b, multiply(A, x)));
    }

    // ��ȡ��ǰʱ����Ϊ����ʱ���
    auto end = std::chrono::high_resolution_clock::now();
    // ���㺯��ִ��ʱ��
    std::chrono::duration<double> duration = end - start;
    std::cout << "NLSS execution time: " << duration.count() << " seconds" << std::endl;

    return x; // Return the solution vector x
}