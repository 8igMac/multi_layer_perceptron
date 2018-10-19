#include <iostream>
#include <fstream>

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
    DataSet data(ifs);
    ifs.close();

    // shuffle the data set
    data.shuffle()

    // train the classifier using cross-validation
    // and collect all the validation result
    double error_sum = 0;
    NeuroNet classifier;
    for(int i=0; i<NUM_FOLD; i++)
    {
        //TODO: how to pass train data
        // training 
        train_data = data.get_train:
        classifier.train(train_data);

        //TODO: how to pass vali data
        // validation and collect result
        error_sum += classifier.validate(vali_data);
    }

    // output the result
    double avg_error = error_sum/NUM_FOLD;
    double pos_predict_rate = 1-avg_error;
    std::cout << "correctness: " << pos_predict_rate << std::endl;

    return 0;
}
