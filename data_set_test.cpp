#include <iostream>
#include <fstream>
#include "include/iris.hpp"
#include "include/data_set.hpp"

#define NUM_FOLDS 10

int main(int argc, char** argv)
{
    std::ifstream ifs(argv[1]);

    DataSet<Iris> data(ifs);
    data.shuffle();

    std::vector<Iris> train_data = data.get_train_data(9, 10);
    std::vector<Iris> vali_data = data.get_vali_data(9, 10);

    // print valid data
    for(auto item: vali_data)
        std::cout << item.to_string() << std::endl;

    std::cout << "\n\n";

    // print train data
    for(auto item: train_data)
        std::cout << item.to_string() << std::endl;

    return 0;
}
