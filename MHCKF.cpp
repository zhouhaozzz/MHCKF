#include "MHCKF.h"

MHCKF::MHCKF(int rank, int size) : rank(rank), size(size)
{

}

MHCKF::~MHCKF()
{
    M.clear();
    H.clear();
    C.clear();
    K.clear();
    F.clear();
    init_face.clear();
}

void MHCKF::Initialization(int heat_num)
{
    this->heat_num = heat_num;
    this->datax = {};
    if (rank == 0)
    {
        for (int i = 1; i <= heat_num; ++i) {
            std::string filename = "data/" + std::to_string(i) + ".txt";
            readData(filename, this->datax, i-1);

            if (i == 1) {
				this->datax_num = this->datax.size() / 2;
			}
        }

        readData("data/heat source.txt", this->datax, 1);
    }

    
    MPI_Bcast(&this->datax_num, 1, MPI_INT, 0, MPI_COMM_WORLD);

    this->input_data = std::vector<double>((heat_num + 2) * this->datax_num);
    MPI_Scatter(this->datax.data(), (heat_num + 2) * this->datax_num, MPI_DOUBLE, this->input_data.data(), (heat_num + 2) * this->datax_num, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // 初始化数据
    this->datax = std::vector<double>(this->datax_num);
    this->M = std::vector<std::vector<double>>(this->datax_num, std::vector<double>(this->heat_num));
    for (int i = 0; i < this->datax_num; i++) {
        this->datax[i] = this->input_data[i] / 1.0e6;
    }

    for (int i = 0; i < this->heat_num; i++) {
        for (int j = 0; j < this->datax_num; j++)
        {
            this->M[j][i] = this->input_data[(i + 1) * this->datax_num + j] * 1.0e9;
        }
	}



    this->K = std::vector<double>(this->datax_num);
    this->init_face = std::vector<double>(this->datax_num, 0);
    for (int i = 0; i < this->datax_num; i++) {
        //this->K[i] = -this->init_face[i] - this->input_data[heat_num][i][1] * 1.0e9;
        this->K[i] = -this->init_face[i] - this->input_data[(heat_num + 1) * this->datax_num + i] * 1.0e9;
    }


}

void MHCKF::writeData(const std::string& filename)
{
    std::vector<double> face_computed = multiply(this->M, this->H);

    std::string filename1 = "data/suface_output_" + filename + ".txt";
    std::ofstream file(filename1);
    if (!file.is_open()) 
    {
        std::cerr << "Error: failed to open file data/suface_output.txt" << std::endl;
        return;
    }

    for (int i = 0; i < datax_num; i++)
    {
        file << std::setprecision(10) << this->datax[i] << " " << this->K[i] << " " << face_computed[i] << std::endl;
    }

    file.close();

    filename1 = "data/power_distribution_" + filename + ".txt";
    std::ofstream file1(filename1);
    if (!file1.is_open())
    {
        std::cerr << "Error: failed to open file data/suface_output.txt" << std::endl;
        return;
    }

    for (int i = 0; i < this->H.size(); i++)
    {
        file1 << std::setprecision(10) << this->H[i] << std::endl;
    }

    file1.close();
}

void MHCKF::readData(const std::string& filename, std::vector<double>& data, int num)
{
    std::vector<std::vector<double>> temp;
    std::string line;

    std::ifstream file(filename);
    if (!file.is_open()) {
		std::cerr << "Error: failed to open file " << filename << std::endl;
		return;
	}

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> row;
        double value;
        while (ss >> value) {
            row.push_back(value);
        }
        if (num == 0) {
            temp.push_back(row);
        }
        else {
            data.push_back(row[1]);
        }
    }

    if (num == 0) {
        for (int i = 0; i < temp.size(); i++) {
            data.push_back(temp[i][0]);
        }
        for (int i = 0; i < temp.size(); i++) {
            data.push_back(temp[i][1]);
        }
	}

    file.close();
}


