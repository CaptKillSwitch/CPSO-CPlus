//
// Created by rh18v on 2022-07-08.
//

#ifndef CPSO_CPLUS_NEURALNETWORK_H
#define CPSO_CPLUS_NEURALNETWORK_H
#include <tuple>
#include <array>

using std::tuple;
using std::array;


class NeuralNetwork {
    private:
        float lambda = 0.001f;
    public:
        /** same as NeuralNetwork constructor configuration */
        array<int,3> nn_configuration{};
        /*
         * nn_config[0]: number of input nodes,
         * nn_config[1]: number of hidden nodes,
         * nn_config[2]: number of output nodes
         * dataset variable is still pending*/
        NeuralNetwork(array<int,3> nn_config, int seed);
        /*
         * test_parameter[0]: layer_number
         * test_parameter[1]: node_number
         * test_parameter[2]: connected_node_number
         * test_parameter[3]: weight*/
        tuple<bool,int> test(array<int,4> test_parameters, float current_fitness, int is_global_test);




};


#endif //CPSO_CPLUS_NEURALNETWORK_H
