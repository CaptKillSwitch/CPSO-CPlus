//
// Created by rh18v on 2022-07-10.
//

#ifndef CPSO_CPLUS_DATA_TYPES_H
#define CPSO_CPLUS_DATA_TYPES_H

#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include <random>

using namespace Eigen;
using std::tuple;
using std::vector;

struct NN_config {
    NN_config() {
        this->input_nodes = 0;
        this->hidden_nodes = 0;
        this->output_nodes = 0;
    }

    NN_config(int i, int h, int o) {
        this->input_nodes = i;
        this->hidden_nodes = h;
        this->output_nodes = o;
    }

    int input_nodes;
    int hidden_nodes;
    int output_nodes;
};

struct Raw_data {
    vector<tuple<MatrixXd, MatrixXd>> values;
};

struct Data{
    Data(const Raw_data &data, NN_config c) {
        this->config = c;
        vector<tuple<MatrixXd, MatrixXd>> data_vector = data.values;
        std::shuffle(data_vector.begin(), data_vector.end(), std::mt19937(std::random_device()()));
        int split_index = int(0.60 * int(data_vector.size()));
        this->train_data = vector<tuple<MatrixXd, MatrixXd>>(data_vector.begin(), data_vector.end() - split_index);
        this->test_data = vector<tuple<MatrixXd, MatrixXd>>(data_vector.end() - split_index, data_vector.end());
    }

    Data() {}

    void shuffle_train() {
        std::shuffle(train_data.begin(), train_data.end(), std::mt19937(std::random_device()()));
    }

    void shuffle_test() {
        std::shuffle(test_data.begin(), test_data.end(), std::mt19937(std::random_device()()));
    }

    void shuffle() {
        shuffle_train();
        shuffle_test();
    }

    vector<tuple<MatrixXd, MatrixXd>> train_data;
    vector<tuple<MatrixXd, MatrixXd>> test_data;
    NN_config config;
};


#endif //CPSO_CPLUS_DATA_TYPES_H
