#include "function.h"
#include "MHCKF.h"
#include "colPivHouseholderQR.h"
#include "normal_equation.h"
#include "QR_MGS.h"
#include "nlss.h";


int main()
{
	MHCKF* mhckf = new MHCKF();

    // ¶ÁÈ¡Êý¾Ý
    mhckf->Initialization(10);

    // colPivHouseholderQR
    colPivHouseholderQr* col_qr = new colPivHouseholderQr();
    mhckf->H = col_qr->linear_least_squares(mhckf->M, mhckf->K);
    printVector(mhckf->H);
    mhckf->writeData("col_qr");
    std::cout << "\n";

    // normal_equation
    normal_equation* n_e = new normal_equation();
    mhckf->H = n_e->linear_least_squares(mhckf->M, mhckf->K);
    printVector(mhckf->H);
    mhckf->writeData("normal_equation");
    std::cout << "\n";

    // QR_MGS
    QR_MGS* qr_mgs = new QR_MGS();
    mhckf->H = qr_mgs->linear_least_squares(mhckf->M, mhckf->K);
    printVector(mhckf->H);
    mhckf->writeData("QR_MGS");
    std::cout << "\n";

    // NLSS
    NLSS* nlss = new NLSS();
    //mhckf->H = nlss->linear_least_squares(mhckf->M, mhckf->K);
    //mhckf->H = nlss->spark_nlss(mhckf->M, mhckf->K);
    mhckf->H = nlss->lsqnonneg(mhckf->M, mhckf->K);
    printVector(mhckf->H);
    mhckf->writeData("nlss");
    std::cout << "\n";


	return 0;
}