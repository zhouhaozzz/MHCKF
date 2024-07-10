#include "function.h"

// ����ת�ú���
std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& A) {
    if (A.empty() || A[0].empty()) {
        return std::vector<std::vector<double>>();
    }

    int rows = A.size();
    int cols = A[0].size();
    std::vector<std::vector<double>> result(cols, std::vector<double>(rows));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[j][i] = A[i][j];
        }
    }

    return result;
}

// ����ת�ó˷�����
std::vector<double> multiplyTranspose(const std::vector<std::vector<double>>& A, const std::vector<double>& x) {
    int rows = A.size();
    int cols = A[0].size();
    std::vector<double> result(cols, 0.0);
    for (int i = 0; i < cols; ++i) {
        for (int j = 0; j < rows; ++j) {
            result[i] += A[j][i] * x[j];
        }
    }
    return result;
}

// �������溯��
std::vector<std::vector<double>> inverse(const std::vector<std::vector<double>>& A) {
    int n = A.size();
    if (n == 0 || A[0].size() != n) {
        throw std::invalid_argument("Matrix must be square.");
    }

    std::vector<std::vector<double>> aug(n, std::vector<double>(n * 2));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            aug[i][j] = A[i][j];
        }
        aug[i][i + n] = 1.0;
    }

    for (int i = 0; i < n; ++i) {
        double diag = aug[i][i];
        if (fabs(diag) < 1e-9) {
            throw std::runtime_error("Matrix is singular and cannot be inverted.");
        }
        for (int j = 0; j < 2 * n; ++j) {
            aug[i][j] /= diag;
        }
        for (int k = 0; k < n; ++k) {
            if (i != k) {
                double factor = aug[k][i];
                for (int j = 0; j < 2 * n; ++j) {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }
    }

    std::vector<std::vector<double>> inv(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            inv[i][j] = aug[i][j + n];
        }
    }

    return inv;
}

// ����˷�����
std::vector<std::vector<double>> multiply(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();

    if (colsA != rowsB) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    std::vector<std::vector<double>> result(rowsA, std::vector<double>(colsB, 0));

    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            for (int k = 0; k < colsA; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return result;
}

// �������Ծ�����
std::vector<double> multiply(const std::vector<std::vector<double>>& A, const std::vector<double>& x) {
    int rows = A.size();
    int cols = A[0].size();

    if (cols != x.size()) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    std::vector<double> result(rows, 0);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i] += A[i][j] * x[j];
        }
    }

    return result;
}

// ������������
std::vector<double> subtract(const std::vector<double>& a, const std::vector<double>& b) {
    int size = a.size();

    if (size != b.size()) {
        throw std::invalid_argument("Vector dimensions do not match for subtraction.");
    }

    std::vector<double> result(size);

    for (int i = 0; i < size; ++i) {
        result[i] = a[i] - b[i];
    }

    return result;
}

// �����ӷ�����
std::vector<double> add(const std::vector<double>& a, const std::vector<double>& b) {
    int size = a.size();

    if (size != b.size()) {
        throw std::invalid_argument("Vector dimensions do not match for addition.");
    }

    std::vector<double> result(size);

    for (int i = 0; i < size; ++i) {
        result[i] = a[i] + b[i];
    }

    return result;
}

// �������
double dotProduct(const std::vector<double>& a, const std::vector<double>& b) {
    int size = a.size();
    double result = 0.0;

    for (int i = 0; i < size; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

// �������Ա�������
std::vector<double> multiply(const std::vector<double>& a, double scalar) {
    int size = a.size();
    std::vector<double> result(size);

    for (int i = 0; i < size; ++i) {
        result[i] = a[i] * scalar;
    }

    return result;
}

// �������еĸ�ֵ����Ϊ��
std::vector<double> clipToNonNegative(const std::vector<double>& a) {
    int size = a.size();
    std::vector<double> result(size);

    for (int i = 0; i < size; ++i) {
        result[i] = std::max(a[i], 0.0);
    }

    return result;
}

// �Ǹ���С���˷���⺯��
std::vector<double> nonNegativeLeastSquares(const std::vector<std::vector<double>>& A, const std::vector<double>& b, int maxIter = 1000, double learningRate = 1e-3) {
    int n = A[0].size();
    std::vector<double> x(n, 0); // ��ʼ��������

    for (int iter = 0; iter < 1; ++iter) {
        std::vector<double> Ax = multiply(A, x);
        std::vector<double> gradient = multiply(transpose(A), subtract(Ax, b));
        std::vector<double> newX = subtract(x, multiply(gradient, learningRate));
        x = clipToNonNegative(newX);
    }

    return x;
}

// ��ӡ��������
void printVector(const std::vector<double>& vec) {
    for (double val : vec) {
        std::cout << std::setw(10) << val << " ";
    }
    std::cout << std::endl;
}

// ��ӡ������
void printMatrix(const std::vector<std::vector<double>>& mat) {
	for (const std::vector<double>& row : mat) {
		for (double val : row) {
			std::cout << std::setw(10) << val << " ";
		}
		std::cout << std::endl;
	}
}

// ����������LU�ֽ�
void luDecomposition(const std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& L, std::vector<std::vector<double>>& U) {
    size_t n = A.size();
    L = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
    U = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));

    for (size_t i = 0; i < n; ++i) {
        // �����Ǿ���
        for (size_t k = i; k < n; ++k) {
            double sum = 0.0;
            for (size_t j = 0; j < i; ++j) {
                sum += (L[i][j] * U[j][k]);
            }
            U[i][k] = A[i][k] - sum;
        }

        // �����Ǿ���
        for (size_t k = i; k < n; ++k) {
            if (i == k) {
                L[i][i] = 1.0; // �Խ���Ϊ1
            }
            else {
                double sum = 0.0;
                for (size_t j = 0; j < i; ++j) {
                    sum += (L[k][j] * U[j][i]);
                }
                L[k][i] = (A[k][i] - sum) / U[i][i];
            }
        }
    }
}

// ����������ǰ�����
std::vector<double> forwardSubstitution(const std::vector<std::vector<double>>& L, const std::vector<double>& b) {
    size_t n = L.size();
    std::vector<double> y(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < i; ++j) {
            sum += L[i][j] * y[j];
        }
        y[i] = b[i] - sum;
    }
    return y;
}

// �����������������
std::vector<double> backSubstitution(const std::vector<std::vector<double>>& U, const std::vector<double>& y) {
    size_t n = U.size();
    std::vector<double> x(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double sum = 0.0;
        for (size_t j = i + 1; j < n; ++j) {
            sum += U[i][j] * x[j];
        }
        x[i] = (y[i] - sum) / U[i][i];
    }
    return x;
}