// file_loader.cpp
#include "file_loader.h"

 float* load_conv_weight(const std::string& name, int len) {
  auto file_name = "../model/resnet50_weight/resnet50_" + name + "_weight.txt";
  return load_data_from_file<float>(file_name, len, true);
}

 int* load_conv_param(const std::string& name, int len) {
  auto file_name = "../model/resnet50_weight/resnet50_" + name + "_param.txt";
  return load_data_from_file<int>(file_name, len, false);
}