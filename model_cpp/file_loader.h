#ifndef FILE_LOADER_H
#define FILE_LOADER_H
#include <string>

template <typename T>
T* load_data_from_file(const std::string& file_name, int len, bool is_float) {
    T* data = (T*)malloc(len * sizeof(T));
    FILE* fp = fopen(file_name.c_str(), "r");
    for (auto i = 0; i < len; i++) {
        float x = 0;
        auto d = fscanf(fp, "%f", &x);
        data[i] = is_float ? x : (int)x;
    }
    fclose(fp);
    return data;
}

float* load_conv_weight(const std::string& name, int len);

int* load_conv_param(const std::string& name, int len);

#endif  // FILE_LOADER_H
