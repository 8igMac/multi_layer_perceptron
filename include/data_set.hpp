#pragma once
#include <sstream>
#include <iostream>

template <typename DATA_TYPE>
class DataSet
{
  public:
    // ifstream constructor
    DataSet(std::istream &is)
    {
        std::string str;
        while(std::getline(is, str))
            v_data.emplace_back(str);
    }

    // random shuffle
    void shuffle()
    {

    }

    // TODO:data retrieval

    // print v_data
    void print_data()
    {
        for(auto item: v_data)
            std::cout << item.to_string() << std::endl;
    }

  private:
    // data
    std::vector<DATA_TYPE> v_data;
};
