#include <fstream>
#include "include/data_set.hpp"
#include "include/neuro_net.hpp"
#include "include/iris.hpp"

// number of folds for cross-validation
#define NUM_FOLD 10

int main(int argc, char** argv)
{
    // check argument list
    if(argc != 2)
    {
        std::cout << "usage: ./program data_set" << std::endl;
        return -1;
    }

    // create data object and init with file stream
    std::ifstream ifs(argv[1]);
    DataSet<Iris> data(ifs);
    ifs.close();

    // shuffle the data set
    data.shuffle();

    // chose classifier parameter
    constexpr int num_feature = 4;
    constexpr int num_label = 3;
    constexpr int num_layer = 2+1;
    constexpr int num_neuron_per_layer = 4;
    constexpr int bias = 1;
    NeuroNet<
        Iris
      , num_feature
      , num_label
      , num_layer
      , num_neuron_per_layer
      , bias
    > classifier;

    // train the classifier
    double error_rate;
    std::vector<Iris> train_data, vali_data;

    // training 
    std::cout << "training..." << std::endl;
    train_data = data.get_train_data(0, NUM_FOLD);
    classifier.train(train_data);
    std::cout << "training finished" << std::endl;

    /*
    // validation and collect result
    std::cout << "validating..." << std::endl;
    vali_data = data.get_vali_data(0, NUM_FOLD);
    error_rate = classifier.validate(vali_data);
    std::cout << "validating finished" << std::endl;
    */
    // validation and collect result
    std::cout << "validating..." << std::endl;
    error_rate = classifier.validate(train_data);
    std::cout << "validating finished" << std::endl;

    // output the result
    double pos_predict_rate = 1-error_rate;
    std::cout << "correctness: " << pos_predict_rate << std::endl;


    return 0;
}
