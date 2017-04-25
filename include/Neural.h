#ifndef NEURAL_H
#define NEURAL_H

#include <iostream>
#include <stdio.h>
#include <vector>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

using namespace std;

class Neural{
    public:
        Neural();
        virtual ~Neural();

        Neural* addDescriptor(vector<double> x, double y);
        Neural* setModel(vector<double> w);
        vector<double> getModel();

        bool StartTraining(int maxIteration);
        bool StartTraining();
        double Humbral();
        void clearDescriptors();

        bool Clasificate(vector<double> x);

    protected:
    private:
        double ClasificatorFunction(int descriptorNumber);
        double LogisticFunction(int descriptorNumber);

        vector< vector<double> > descriptors;
        vector<double> descriptorAnswer;
        vector<double> model;
        double T;
        double landa;
        double alpha;
};

#endif // NEURAL_H
