#include "function.h"
#include "MHCKF.h"
#include "colPivHouseholderQR.h"
#include "normal_equation.h"
#include "QR_MGS.h"


int main()
{
	MHCKF* mhckf = new MHCKF();

    // ¶ÁÈ¡Êı¾İ
    mhckf->Initialization(10);

    // colPivHouseholderQR
    colPivHouseholderQr* col_qr = new colPivHouseholderQr();
    mhckf->H = col_qr->linear_least_squares(mhckf->M, mhckf->K);
    printVector(mhckf->H);
    std::cout << "\n";

    // normal_equation
    normal_equation* n_e = new normal_equation();
    mhckf->H = n_e->linear_least_squares(mhckf->M, mhckf->K);
    printVector(mhckf->H);
    std::cout << "\n";

    // QR_MGS
    QR_MGS* qr_mgs = new QR_MGS();
    mhckf->H = qr_mgs->linear_least_squares(mhckf->M, mhckf->K);
    printVector(mhckf->H);
    mhckf->writeData();
    std::cout << "\n";


	return 0;
}