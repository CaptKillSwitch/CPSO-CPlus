//
// Created by rh18v on 2022-07-08.
//
#include <iostream>
#include <Eigen/Dense>
#include "./components/neural net/NeuralNetwork.h"
#include "./components/data loader/data_provider.h"

using namespace std;
using namespace Eigen;

int main(){
    srand(1000);
    Data data = data_provider::getIrisData();
    NeuralNetwork nn(data);
    nn.get_results(true);
    return 0;
}