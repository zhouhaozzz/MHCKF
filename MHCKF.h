#pragma once
#include "function.h"
// ��ȡ����

class MHCKF
{
public:
	int rank = 0;
	int size = 1;
	int heat_num;						// the number of the heaters
	int datax_num;						// the number of the data points in the x direction
	std::vector<double> input_data;		// the input data
	std::vector<double> datax;			// the x direction data points
	std::vector<std::vector<double>> M;	// the response function of the electrical heaters  (m*n)
	std::vector<double> H;				// a series of the heat fluxes generated by the heaters (1*n)
	std::vector<double> C;				// the mirror initial deformation caused by the processing, clamping, and gravity, etc. (m*1)
	std::vector<double> K;				// the deformation in the meridional direction caused by the X-ray power (m*1)	
	std::vector<double> F;				// the actual deformation generated by the three left terms (m*1)
	std::vector<double> init_face;		// the initial face of the mirror = C + K


	MHCKF(int rank, int size);
	~MHCKF();

	void Initialization(int heat_num);
	void readData(const std::string& filename, std::vector<double>& data, int num);
	void writeData(const std::string& filename);
};

