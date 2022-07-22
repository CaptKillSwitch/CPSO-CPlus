//
// Created by rh18v on 2022-07-09.
//

#ifndef CPSO_CPLUS_DATA_PROVIDER_H
#define CPSO_CPLUS_DATA_PROVIDER_H

#include "../data_types.h"
#include <random>
#include <string>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

using namespace Eigen;
using std::string;
using std::ifstream;
using std::getline;
using std::stringstream;

class data_provider {
private:
    MatrixXd replace_missing_values(string line);

    MatrixXd normalize(MatrixXd matrix);

public:
    static Data getIrisData() {
        NN_config config(4, 4, 3);
        Raw_data data;
        string line;
        ifstream reader;
        reader.open("./data/iris.data");
        while (getline(reader, line)) {
            stringstream values(line);
            string value;
            MatrixXd input, output;
            input.resize(5, 1);
            output.resize(3, 1);
            output.fill(0);
            input(0, 0) = 1;
            for (int i = 1; i < 5; i++) {
                getline(values, value, ',');
                input(i, 0) = std::stof(value);
            }
            getline(values, value, ',');
            output(std::stoi(value),0) = 1;
            std::cout << "input" << std::endl << input << std::endl << "output" << std::endl << output << std::endl
                      << "--------------" << std::endl;
            data.values.emplace_back(input, output);
        }
        return {data, config};
    };
};


#endif //CPSO_CPLUS_DATA_PROVIDER_H
