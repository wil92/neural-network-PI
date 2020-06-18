#include "../include/Neural.h"

Neural::Neural()
{
    T = 0.5;
    landa = 0.;
    alpha = 1.;
}

Neural::~Neural()
{
}

/** \brief Add a descriptor to the neuronal web.
 *
 * \param new descriptor.
 * \param result of the descriptor.
 * \return return the instance of the class.
 */
Neural *Neural::addDescriptor(vector<double> x, double y)
{
    descriptorAnswer.push_back(y);
    vector<double> descriptor = vector<double>(x.size() + 1);
    descriptor[0] = 1.;
    for (int i = x.size(); i > 0; i--)
    {
        descriptor[i] = x[i - 1];
    }
    descriptors.push_back(descriptor);

    if (model.size() != descriptor.size())
    {
        srand(time(0));
        while (model.size() < descriptor.size())
            model.push_back((double)(rand() % 1000));
        while (model.size() > descriptor.size())
            model.pop_back();
    }

    return this;
}

Neural *Neural::setModel(vector<double> w)
{
    model = w;
    return this;
}

vector<double> Neural::getModel()
{
    return model;
}

void Neural::clearDescriptors()
{
    descriptors.clear();
}

/**
 * Start the training the neural network with 1 million maximum of iteration by default
 */
bool Neural::startTraining()
{
    return startTraining(1000000);
}

/**
 * Start the training the neural network given the maximum number of iterations
 */
bool Neural::startTraining(int maxIteration)
{
    double logisticArray[descriptors.size()];
    double clasificationArray[descriptors.size()];

    char digitos[100];

    for (int i = 0; i <= maxIteration; i++)
    {

        // For show porcent of iteration in the training
        if (i)
        {
            for (int i = strlen(digitos); i >= 0; i--)
                printf("\r");
            sprintf(digitos, "%d / %d", i, maxIteration);
            printf("%s", digitos);
        }
        else
        {
            sprintf(digitos, "%d / %d", i, maxIteration);
            printf("%s", digitos);
        }

        bool needTraining = 0;
        for (int i = 0; i < descriptors.size(); i++)
        {
            logisticArray[i] = logisticFunction(i);
            clasificationArray[i] = clasificatorFunction(i);

            double humbralValue = humbral();
            if ((clasificationArray[i] > humbralValue && descriptorAnswer[i] == 0) ||
                (clasificationArray[i] < humbralValue && descriptorAnswer[i] == 1))
            {
                needTraining = 1;
            }
        }
        if (!needTraining)
        {
            cout << "\nConverge on iteration " << i << endl;
            return i;
        }

        // Calculating the heuristics error
        vector<double> newModel;
        newModel.push_back(model[0]);
        for (int i = 1; i < model.size(); i++)
        {
            double acurrance = 0, M = descriptors.size();
            for (int j = 0; j < descriptors.size(); j++)
            {
                acurrance += (logisticArray[j] - descriptorAnswer[j]) * descriptors[j][i] + landa * model[i];
            }
            acurrance *= alpha;
            acurrance /= M;
            newModel.push_back(model[i] - acurrance);
        }

        model = newModel;
    }
    return 1;
}

double Neural::logisticFunction(int descriptorNumber)
{
    double logistic = 1. / (1. + exp(-clasificatorFunction(descriptorNumber)));
    return logistic;
}

double Neural::clasificatorFunction(int descriptorNumber)
{
    double clasificator = 0;
    for (int i = 0; i < model.size(); i++)
    {
        clasificator += model[i] * descriptors[descriptorNumber][i];
    }
    return clasificator;
}

/**
 * Given the descriptor, return true if it calsify or false otherwise
 */
bool Neural::clasificate(vector<double> descriptor)
{
    double clasificator = model[0];

    for (int i = 0; i < model.size() - 1; i++)
    {
        clasificator += model[i + 1] * descriptor[i];
    }

    return clasificator > humbral();
}

double Neural::humbral()
{
    return log(T / (1 - T));
}
