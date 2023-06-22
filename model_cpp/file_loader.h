#ifndef FILE_LOADER_H
#define FILE_LOADER_H

#include <string>

template <typename T>
T* load_data_from_file(const std::string& file_name, int len, bool is_float);

float* load_conv_weight(const std::string& name, int len);

int* load_conv_param(const std::string& name, int len);

#endif  // FILE_LOADER_H
