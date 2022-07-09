//
// Created by rh18v on 2022-07-08.
//

#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(array<int, 3> nn_config, int seed) {
    this->nn_configuration = nn_config;
}

tuple<bool, int>
NeuralNetwork::test(array<int, 4> test_parameters, float current_fitness, int is_global_test) {
    tuple<bool, int> result = {true, 1};

    return result;
}

