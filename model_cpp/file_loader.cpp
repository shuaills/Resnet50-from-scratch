// file_loader.cpp
#include "file_loader.h"

template <typename T>
T* load_data_from_file(const std::string& file_name, int len, bool is_float) {
  T* data = (T*)malloc(len * sizeof(T));
  FILE* fp = fopen(file_name.c_str(), "r");
  // std::cout << "file_name = " << file_name << ", fp = " << fp << std::endl;
  for (auto i = 0; i < len; i++) {
    float x = 0;
    auto d = fscanf(fp, "%f", &x);
    data[i] = is_float ? x : (int)x;
  }
  fclose(fp);
  return data;
}

 float* load_conv_weight(const std::string& name, int len) {
  auto file_name = "../model/resnet50_weight/resnet50_" + name + "_weight.txt";
  return load_data_from_file<float>(file_name, len, true);
}

 int* load_conv_param(const std::string& name, int len) {
  auto file_name = "../model/resnet50_weight/resnet50_" + name + "_param.txt";
  return load_data_from_file<int>(file_name, len, false);
}