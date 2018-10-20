#pragma once
#include <vector>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cstdlib>
#include <ctime>

template <
    typename DATA_TYPE
  , int NUM_FEATURE
  , int NUM_LABEL
  , int NUM_LAYER // including input and ouput layer
  , int NUM_NEURON_PER_LAYER
  , int BIAS
>
class NeuroNet
{
  public:
    // constructor
    NeuroNet()
    : layers(NUM_LAYER)
    , no_act_layers(NUM_LAYER)
    , errors(NUM_LAYER)
    , weights(NUM_LAYER) // only use 1~NUM_LAYER-1
    , gradients(NUM_LAYER)
    , learn_rate(5)
    {
        // init size of layers, no_act_layers, errors
        for(int i=0; i<NUM_LAYER; i++)
        {
            if(i == 0) // input layer
            {
                layers[i].resize(NUM_FEATURE+1); // plus bias term
                no_act_layers[i].resize(NUM_FEATURE+1); 
                errors[i].resize(NUM_FEATURE+1); 

                weights[i].resize(NUM_FEATURE+1,
                        std::vector<double>(NUM_NEURON_PER_LAYER+1)); 
                gradients[i].resize(NUM_FEATURE+1,
                        std::vector<double>(NUM_NEURON_PER_LAYER+1)); 

            }
            else if(i == NUM_LAYER-1) // output layer
            {
                layers[i].resize(NUM_LABEL+1); // plus begining padding
                no_act_layers[i].resize(NUM_LABEL+1); 
                errors[i].resize(NUM_LABEL+1); 

                weights[i].resize(NUM_LABEL+1,
                        std::vector<double>(NUM_NEURON_PER_LAYER+1)); 
                gradients[i].resize(NUM_LABEL+1, 
                        std::vector<double>(NUM_NEURON_PER_LAYER+1)); 
            }
            else // hidden layer
            {
                layers[i].resize(NUM_NEURON_PER_LAYER+1); // plus bias term
                no_act_layers[i].resize(NUM_NEURON_PER_LAYER+1); 
                errors[i].resize(NUM_NEURON_PER_LAYER+1); 

                weights[i].resize(NUM_NEURON_PER_LAYER+1, 
                        std::vector<double>(NUM_NEURON_PER_LAYER+1)); 
                gradients[i].resize(NUM_NEURON_PER_LAYER+1, 
                        std::vector<double>(NUM_NEURON_PER_LAYER+1)); 
            }
        }

        // adjust first hidden layer matrix size
        for(int i=0; i<weights[1].size(); i++)
            weights[1][i].resize(NUM_FEATURE+1);

        // init bias term on layers to be 1
        for(int i=0; i<layers.size(); i++)
            layers[i][0] = 1;

        // init weight randomly, and bias 
        std::srand(time(NULL));
        for(int i=0; i<weights.size(); i++)
            for(int j=1; j<weights[i].size(); j++)
                for(int k=0; k<weights[i][j].size(); k++)
                    if(k == 0)
                        weights[i][j][k] = BIAS;
                    else
                        weights[i][j][k] = (double)(rand()%10) * 0.01; 
    }

    // train: batch version
    void train(std::vector<DATA_TYPE> train_data)
    {
        for(int n=0; !stop_criterion(n); n++)
        {
            std::cout << "epoch: " << n+1 << std::endl; //debug
            for(auto data: train_data)
            {
                // forward pass: evaluate the output layer
                classify(data);

                // backward pass and accumulate all gradients
                back_prop(data);
            }

            print_data_member(); //debug

            // update all the weight
            for(int i=0; i<weights.size(); i++)
                for(int j=0; j<weights[i].size(); j++)
                    // k=0 is bias
                    for(int k=1; k<weights[i][j].size(); k++)
                        weights[i][j][k] -= gradients[i][j][k] * learn_rate;
        }
    }

    // validate, return error rate
    double validate(std::vector<DATA_TYPE> vali_data)
    {
        int error = 0;
        for(auto data: vali_data)
            if(data.label != classify(data))
                error++;

        return (double)error / (double)vali_data.size();
    }

    // stop criterion: number of times
    bool stop_criterion(int n)
    {
        return n>1;
    }

    // classify input data, return label index
    int classify(DATA_TYPE data)
    {
        // set input from 1 (0 is bias)
        for(int i=1; i<layers[0].size(); i++)
            layers[0][i] = data.features[i];

        // forward pass: 
        // traverse all layer except layer 0 (input layer)
        for(int i=1; i<layers.size(); i++)
            // start from 1st neuron, 0 is bias
            for(int j=1; j<layers[i].size(); j++)
            {
                // single neuron evaluation
                no_act_layers[i][j] = matrix_mul(weights[i][j], layers[i-1]);
                layers[i][j] = act_func(no_act_layers[i][j]);
            }

        // find the max of discrimination func as the classify result
        int result = 1;
        for(int i=2; i<layers.back().size(); i++)
            if(layers.back()[i] > layers.back()[i-1])
                result = i;

        
        return result-1; // minus 1 to transfer to label index
    }

    // back propogation: accumulate all gradients
    void back_prop(DATA_TYPE data)
    {
        // set label vector (expected ouput)
        // add padding at the end, begin is 1
        std::vector<double> v_label(NUM_LABEL+1, 0);
        v_label[data.label+1] = 1;

        // backward, ommit input layer
        for(int i=gradients.size()-1; i>=1; i--) 
            for(int j=0; j<gradients[i].size(); j++)
            {
                // evaluate error
                if(i == NUM_LAYER-1) // output layers
                    // output layers error term
                    errors[i][j] = (-1.0) * (v_label[j] - layers[i][j]);
                else
                {
                    // k=0 is bias
                    for(int k=1; k<layers[i+1].size(); k++)
                        errors[i][j] += 
                            errors[i+1][k] 
                          * de_act_func(no_act_layers[i+1][k])
                          * weights[i+1][k][j];
                }

                // evaluate gradients
                for(int k=1; k<gradients[i][j].size(); k++)
                    gradients[i][j][k] += 
                        errors[i][j]
                      * de_act_func(no_act_layers[i][j])
                      * layers[i-1][k]; 
            }
    }

    // print all data member, for debug purpose
    void print_data_member()
    {
        // print layers
        std::cout << "[layers]" << std::endl;
        for(auto row: layers)
        {
            for(auto element: row)
                std::cout << element << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;

        // print no_act_layers
        std::cout << "[no_act_layers]" << std::endl;
        for(auto row: no_act_layers)
        {
            for(auto element: row)
                std::cout << element << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;

        // print errors
        std::cout << "[errors]" << std::endl;
        for(auto row: errors)
        {
            for(auto element: row)
                std::cout << element << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;

        // print weights
        std::cout << "[weights]" << std::endl;
        for(auto layer: weights)
            std::cout << layer.size() << " ";
        std::cout << std::endl;
        for(auto row: weights[1])
        {
            for(auto element: row)
                std::cout << element << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;

        // print gradients
        std::cout << "[gradients]" << std::endl;
        for(auto layer: gradients)
            std::cout << layer.size() << " ";
        std::cout << std::endl;
        for(auto row: gradients[0])
        {
            for(auto element: row)
                std::cout << element << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

  private:
    // matrix multiplication
    double matrix_mul(std::vector<double> lhs, std::vector<double> rhs)
    {
        // check if matrix size match
        if(lhs.size() != rhs.size())
            throw std::logic_error{
                "matrix multiplication error: size don't match\n"
            };

        double sum = 0;
        for(int i=0; i<lhs.size(); i++)
            sum += lhs[i] * rhs[i];

        return sum;
    }

    // neurons
    std::vector<std::vector<double>> layers;

    // neurons before activation func
    std::vector<std::vector<double>> no_act_layers;

    // errors: de(J)/de(y)
    std::vector<std::vector<double>> errors;

    // weights
    std::vector<std::vector<std::vector<double>>> weights;

    // gradients: de(J)/de(w)
    std::vector<std::vector<std::vector<double>>> gradients;



    // activation function
    double act_func(double value)
    {
        // use f(x) = max(0, x) as activation function
        return std::max(0.0, value);
    }

    // derivitive of activation function
    double de_act_func(double value)
    {
        return (value > 0.0) ?1 :0;
    }

    // learning rate
    double learn_rate;
};

