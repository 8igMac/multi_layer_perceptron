#pragma once
#include <string>
#include <vector>
#include <charconv>

struct Iris
{
    // constructor, init iris with string input
    Iris(std::string str)
    : features(4), label(0)
    {
        set_data(str);
    }

    // set iris data member
    void set_data(std::string str)
    {
        size_t begin, end = 0;

        // set feature
        for(int i=0; i<4; i++)
        {
            begin = str.find_first_not_of(',', end);
            end = str.find_first_of(',', begin);
            features[i] = std::stof(str.substr(begin, end-begin));
        }
        
        // set label
        if(str.substr(end+1, 7).compare("Iris-ve") == 0)
            label = 1;
        else if(str.substr(end+1, 7).compare("Iris-vi") == 0)
            label = 2;
    }

    // output data member to string
    std::string to_string()
    {
        std::string name;
        switch(label)
        {
            case 0: name = "Iris-sentosa"; break;
            case 1: name = "Iris-versicolor"; break;
            case 2: name = "Iris-virginica"; break;
            default: name = "Iris-sentosa"; break;
        }

        return 
            std::to_string(features[0]) + "," +
            std::to_string(features[1]) + "," +
            std::to_string(features[2]) + "," + 
            std::to_string(features[3]) + "," +
            name;
    }

    // Iris label
    // label0: Iris-sentosa
    // label1: Iris-versicolor
    // label2: Iris-verginica
    int label;

    // 4 features
    std::vector<double> features;
};
