//
// Created by rh18v on 2022-07-08.
//

#ifndef CPSO_CPLUS_NEURALNETWORK_H
#define CPSO_CPLUS_NEURALNETWORK_H

#include <tuple>
#include <array>
#include <utility>
#include <vector>
#include <Eigen/Dense>

#include <iostream>
#include <utility>
#include "components/data_types.h"

using std::tuple;
using std::array;
using std::vector;
using std::make_tuple;
using namespace Eigen;

class NeuralNetwork {
private:
    Data data;
    float lambda = 0.001f;
    int batchSize = 100;
    array<MatrixXd, 2> nn;

    static void relu(MatrixXd &z) {
        for (int i = 0; i < z.rows(); i++) {
            for (int j = 0; j < z.cols(); j++) {
                if (z(i, j) < 0) {
                    z(i, j) = 0;
                }
            }
        }
    }

    double do_feedforward_train(bool full_size = false, bool show_output = false) const {
        double mse = 0;
        for (int i = 0; i < batchSize || (full_size && i < data.train_data.size()); ++i) {
            MatrixXd x, y;
            std::tie(x, y) = data.train_data[i];
            MatrixXd z = nn[0] * x;
            for (int l = 1; l < nn.size(); l++) {
                std::cout << z.transpose() << std::endl;
                relu(z);
                std::cout << z.transpose() << std::endl;
                z = nn[l] * z;
            }
            relu(z);
            if (show_output) {
                std::cout << z.transpose() << " : " << y.transpose() << std::endl;
            }
        }
        return mse;
    }

    double do_feedforward_test(bool show_output = false) const {
        double mse = 0;
        for (const auto &tuple: data.test_data) {
            MatrixXd x, y;
            std::tie(x, y) = tuple;
            MatrixXd z = nn[0] * x;
            for (int l = 1; l < nn.size(); l++) {
                std::cout << z << std::endl;
                relu(z);
                std::cout << z << std::endl;
                z = nn[l] * z;
            }
            relu(z);
            if (show_output) {
                std::cout << z << " : " << y << std::endl;
            }
        }
        return mse;
    }

public:

    explicit NeuralNetwork(Data &data) {
        this->nn[0] = MatrixXd::Random(data.config.input_nodes, data.config.hidden_nodes + 1);
        this->nn[1] = MatrixXd::Random(data.config.hidden_nodes + 1, data.config.output_nodes);
        this->data = data;
        if (this->batchSize > data.train_data.size()) {
            this->batchSize = data.train_data.size();
        }
        for (int l = 0; l < nn.size(); l++) {
            for (int w_j = 0; w_j < nn[l].rows(); w_j++) {
                this->nn[l](w_j, 0) = 0.01; // bias
            }
        }
        std::cout << nn[0] << std::endl;
        std::cout << nn[1] << std::endl;
    };


    tuple<bool, double> test(int layer, int node_i, int v_j, float test_weight, bool is_global_test) {
        this->data.shuffle_train();
        double current_value = this->nn[layer](node_i, v_j);
        double mse_c = this->do_feedforward_train();
        this->nn[layer](node_i, v_j) = test_weight;
        double mse_t = this->do_feedforward_train();
        if (mse_c > mse_t) {
            this->nn[layer](node_i, v_j) = current_value;
            return {false, mse_t};
        }
        return {true, mse_t};
    };

    tuple<double, double> get_results(bool show_output = false) {
        return {this->do_feedforward_train(true,show_output), this->do_feedforward_test(show_output)};
    }

    double get(int layer, int node_i, int v_j) {
        return this->nn[layer](node_i, v_j);
    }

};


#endif //CPSO_CPLUS_NEURALNETWORK_H
