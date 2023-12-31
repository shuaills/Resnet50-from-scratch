#ifndef CONV_LAYERS_H
#define CONV_LAYERS_H


// Convolutional 2D layer function
float* my_conv2d(float* img,
                        float* weight,
                        int hi,
                        int wi,
                        int& ho,
                        int& wo,
                        int ci,
                        int co,
                        int kernel,
                        int stride,
                        int pad,
                        bool is_free_img = true);

// Fully connected layer function
float* my_fc(float* img, float* weight, float* bias);

// Max pooling layer function
float* my_max_pool(float* img);

// Average pooling layer function
float* my_avg_pool(float* img);

// Batch normalization layer function
float* my_bn(float* img, 
                    float* mean, 
                    float* var, 
                    float* gamma, 
                    float* bias, 
                    int h, 
                    int w, 
                    int c);

// Relu activation function
float* compute_relu_layer(float* img, int len);

#endif // CONV_LAYERS_H
