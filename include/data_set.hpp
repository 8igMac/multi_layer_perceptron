#pragma once
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm>

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
        
        // set rand seed
        std::srand(time(NULL));
    }

    // random shuffle
    void shuffle()
    {
        // kruth shuffle
        int j;
        for(int i=0; i<v_data.size()-1; i++)
        {
            j = i+rand()%(v_data.size()-i);
            std::swap(v_data[i], v_data[j]);
        }
    }

    // retrieve training data
    std::vector<DATA_TYPE> get_train_data(int idx, int num_fold)
    {
        std::vector<DATA_TYPE> v_train;
        if(idx == num_fold)
        {
            std::cout << "get_train_data() error: idx too big" << std::endl;
            return v_train;
        }
            
        int length = v_data.size() / num_fold;
        for(int i=0; i<v_data.size(); i++)
            if(i<i*length || i>(i+1)*length)
                v_train.push_back(v_data[i]);

        return v_train;
    }

    // retrieve validation data
    std::vector<DATA_TYPE> get_vali_data(int idx, int num_fold)
    {
        std::vector<DATA_TYPE> v_vali;
        if(idx == num_fold)
        {
            std::cout << "get_vali_data() error: idx too big" << std::endl;
            return v_vali;
        }
            
        int length = v_data.size() / num_fold;
        for(int i=0; i<length; i++)
            v_vali.push_back(v_data[idx*length+i]);

        return v_vali;
    }

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
