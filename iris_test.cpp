#include <iostream>
#include <fstream>
#include "include/iris.hpp"
#include "include/data_set.hpp"

int main(int argc, char** argv)
{
    std::ifstream ifs(argv[1]);

    DataSet<Iris> data(ifs);

    data.print_data();

    return 0;
}
