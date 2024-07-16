#include "function.h"
#include "MHCKF.h"
#include "colPivHouseholderQR.h"
#include "normal_equation.h"
#include "QR_MGS.h"
#include "nlss.h";
#include "LSQ_cuda.cuh"


int main(int argc, char* argv[])
{
    int rank = 0;
    int size = 1;

    MPI_Init(0, 0);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    MHCKF* mhckf = new MHCKF(rank, size);

    // ¶ÁÈ¡Êý¾Ý
    mhckf->Initialization(10);
    MPI_Barrier(MPI_COMM_WORLD);

    // colPivHouseholderQR
    colPivHouseholderQr* col_qr = new colPivHouseholderQr();
    mhckf->H = col_qr->linear_least_squares(mhckf->M, mhckf->K);
    printVector(mhckf->H);
    mhckf->writeData("col_qr");
    std::cout << "\n";
    MPI_Barrier(MPI_COMM_WORLD);

    // normal_equation
    normal_equation* n_e = new normal_equation();
    mhckf->H = n_e->linear_least_squares(mhckf->M, mhckf->K);
    printVector(mhckf->H);
    mhckf->writeData("normal_equation");
    std::cout << "\n";
    MPI_Barrier(MPI_COMM_WORLD);

    // QR_MGS
    QR_MGS* qr_mgs = new QR_MGS();
    mhckf->H = qr_mgs->linear_least_squares(mhckf->M, mhckf->K);
    printVector(mhckf->H);
    mhckf->writeData("QR_MGS");
    std::cout << "\n";
    MPI_Barrier(MPI_COMM_WORLD);

    // NLSS
    NLSS* nlss = new NLSS();
    //mhckf->H = nlss->linear_least_squares(mhckf->M, mhckf->K);
    //mhckf->H = nlss->spark_nlss(mhckf->M, mhckf->K);
    mhckf->H = nlss->lsqnonneg(mhckf->M, mhckf->K);
    printVector(mhckf->H);
    mhckf->writeData("nlss");
    std::cout << "\n";
    MPI_Barrier(MPI_COMM_WORLD);

#ifdef CUDA
    // LSQ_cuda
    //LSQ_CUDA::cuda_text();
    LSQ_CUDA::normal_equation(mhckf->M, mhckf->K);
#endif

    MPI_Finalize();
	return 0;
}