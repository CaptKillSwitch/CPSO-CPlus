//
// Created by rh18v on 2022-07-08.
//

#ifndef CPSO_CPLUS_OPTIMIZER_H
#define CPSO_CPLUS_OPTIMIZER_H


#include "components/neural net/NeuralNetwork.h"

class optimizer {
    private:
        virtual void generate_swarm()=0;
        float max_velocity = 0.14286f;
        float c1 = 1.49618f;
        float c2 = 1.49618f;
        float inertia = 0.729844;
    public:
        optimizer(NeuralNetwork *network, int seed);

        virtual void optimize()=0;
};


#endif //CPSO_CPLUS_OPTIMIZER_H
