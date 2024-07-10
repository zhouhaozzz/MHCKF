#include "MHCKF.h"

MHCKF::MHCKF()
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
    this->input_data = {};
    for (int i = 1; i <= heat_num; ++i) {
        std::string filename = "data/" + std::to_string(i) + ".txt";
        readData(filename, this->input_data);
    }

    readData("data/heat source.txt", this->input_data);

    // 初始化数据
    this->datax_num = this->input_data[0].size();
    this->datax = std::vector<double>(this->datax_num);
    this->M = std::vector<std::vector<double>>(this->datax_num, std::vector<double>(this->heat_num));
    for (int i = 0; i < this->datax_num; i++) {
        this->datax[i] = this->input_data[0][i][0] / 1.0e6;

        for (int j = 0; j < this->heat_num; j++) {
            this->M[i][j] = this->input_data[j][i][1] * 1.0e9;
        }
    }

    this->K = std::vector<double>(this->datax_num);
    this->init_face = std::vector<double>(this->datax_num, 0);
    for (int i = 0; i < this->datax_num; i++) {
        this->K[i] = -this->init_face[i] - this->input_data[heat_num][i][1] * 1.0e9;
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

void MHCKF::readData(const std::string& filename, std::vector<std::vector<std::vector<double>>>& data)
{
    std::vector<std::vector<double>> data1 = {};
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
        data1.push_back(row);
    }

    file.close();

    data.push_back(data1);
}


