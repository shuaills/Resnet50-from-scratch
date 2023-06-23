// conv_layers.cpp
#include "conv_layers.h"
#include <cmath>
#include <cstdint>
#include "debug.h"
#include <iostream>




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
                        bool is_free_img) {
#if DEBUG_SHOW
  printf("conv in: (%d, %d, %d)\n", hi, wi, ci);
#endif
  ho = (hi + 2 * pad - kernel) / stride + 1;
  wo = (wi + 2 * pad - kernel) / stride + 1;
  float* out = (float*)malloc(ho * wo * co * sizeof(float));

  for (int co_idx = 0; co_idx < co; co_idx++) {
    for (int ho_idx = 0; ho_idx < ho; ho_idx++) {
      const int in_h_origin = ho_idx * stride - pad;
      for (int wo_idx = 0; wo_idx < wo; wo_idx++) {
        const int in_w_origin = wo_idx * stride - pad;
        const int filter_h_start = std::max(0, -in_h_origin);
        const int filter_w_start = std::max(0, -in_w_origin);
        const int filter_h_end = std::min(kernel, hi - in_h_origin);
        const int filter_w_end = std::min(kernel, wi - in_w_origin);
        float acc = 0;
        for (int kh_idx = filter_h_start; kh_idx < filter_h_end; kh_idx++) {
          const int hi_index = in_h_origin + kh_idx;
          for (int kw_idx = filter_w_start; kw_idx < filter_w_end; kw_idx++) {
            const int wi_index = in_w_origin + kw_idx;
            for (int ci_ = 0; ci_ < ci; ci_++) {
              auto in_data = img[hi_index * wi * ci + wi_index * ci + ci_];
              auto weight_data =
                  weight[co_idx * kernel * kernel * ci + kh_idx * kernel * ci + kw_idx * ci + ci_];
              acc += in_data * weight_data;
            }
          }
        }
        out[ho_idx * wo * co + wo_idx * co + co_idx] = acc;
      }
    }
  }

  if (is_free_img) {
    free(img);
  }
  free(weight);
#if DEBUG_SHOW
  printf("conv out: (%d, %d, %d)\n", ho, wo, co);
#endif
  return out;
}

float* my_fc(float* img, float* weight, float* bias) {
#if DEBUG_SHOW
  printf("fc in: (1000, 2048)\n");
  printf("fc out: (1000)\n");
#endif
  float* out = (float*)malloc(1000 * sizeof(float));
  for (int i = 0; i < 1000; i++) {
    float sum_x = float(0);
    for (int j = 0; j < 2048; j++) {
      auto l = img[j];
      auto r = weight[i * 2048 + j];
      sum_x += l * r;
    }
    out[i] = sum_x + bias[i];
  }
  free(img);
  free(weight);
  free(bias);
  return out;
}

float* my_max_pool(float* img) {
  auto hi = 112;
  auto wi = 112;
  auto channel = 64;
  auto pad = 1;
  auto stride = 2;
  auto kernel = 3;
  auto ho = (hi + 2 * pad - kernel) / stride + 1;
  auto wo = (wi + 2 * pad - kernel) / stride + 1;
#if DEBUG_SHOW
  printf("maxpool in: (%d, %d, %d)\n", hi, wi, channel);
#endif
  float* out = (float*)malloc(ho * wo * channel * sizeof(float));

  for (auto c_ = 0; c_ < channel; c_++) {
    for (auto ho_idx = 0; ho_idx < ho; ho_idx++) {
      int in_h_origin = ho_idx * stride - pad;
      for (auto wo_idx = 0; wo_idx < wo; wo_idx++) {
        int in_w_origin = wo_idx * stride - pad;
        auto filter_h_start = std::max(0, -in_h_origin);
        auto filter_w_start = std::max(0, -in_w_origin);
        auto filter_h_end = std::min(kernel, hi - in_h_origin);
        auto filter_w_end = std::min(kernel, wi - in_w_origin);
        float max_x = float(0);
        for (auto kh_idx = filter_h_start; kh_idx < filter_h_end; kh_idx++) {
          auto hi_index = in_h_origin + kh_idx;
          for (auto kw_idx = filter_w_start; kw_idx < filter_w_end; kw_idx++) {
            auto wi_index = in_w_origin + kw_idx;
            auto in_data = img[hi_index * wi * channel + wi_index * channel + c_];
            max_x = std::max(in_data, max_x);
          }
        }
        out[ho_idx * wo * channel + wo_idx * channel + c_] = max_x;
      }
    }
  }
  free(img);
#if DEBUG_SHOW
  printf("maxpool out: (%d, %d, %d)\n", ho, wo, channel);
#endif
  return out;
}

float* my_avg_pool(float* img) {
  auto hi = 7;
  auto wi = 7;
  auto channel = 2048;
  auto pad = 0;
  auto stride = 1;
  auto kernel = 7;
  auto ho = (hi + 2 * pad - kernel) / stride + 1;
  auto wo = (wi + 2 * pad - kernel) / stride + 1;
  float* out = (float*)malloc(ho * wo * channel * sizeof(float));
#if DEBUG_SHOW
  printf("avgpool in: (%d, %d, %d)\n", hi, wi, channel);
#endif

  for (auto c_ = 0; c_ < channel; c_++) {
    for (auto ho_idx = 0; ho_idx < ho; ho_idx++) {
      int in_h_origin = ho_idx * stride - pad;
      for (auto wo_idx = 0; wo_idx < wo; wo_idx++) {
        int in_w_origin = wo_idx * stride - pad;
        auto filter_h_start = std::max(0, -in_h_origin);
        auto filter_w_start = std::max(0, -in_w_origin);
        auto filter_h_end = std::min(kernel, hi - in_h_origin);
        auto filter_w_end = std::min(kernel, wi - in_w_origin);
        float sum = float(0);
        for (auto kh_idx = filter_h_start; kh_idx < filter_h_end; kh_idx++) {
          auto hi_index = in_h_origin + kh_idx;
          for (auto kw_idx = filter_w_start; kw_idx < filter_w_end; kw_idx++) {
            auto wi_index = in_w_origin + kw_idx;
            auto in_data = img[hi_index * wi * channel + wi_index * channel + c_];
            sum += in_data;
          }
        }
        out[ho_idx * wo * channel + wo_idx * channel + c_] = sum / (kernel * kernel);
      }
    }
  }
  free(img);
#if DEBUG_SHOW
  printf("avgpool out: (%d, %d, %d)\n", ho, wo, channel);
#endif
  return out;
}

float* my_bn(
    float* img, float* mean, float* var, float* gamma, float* bias, int h, int w, int c) {
#if DEBUG_SHOW
  printf("bn in : (%d, %d, %d)\n", h, w, c);
#endif
  float* out = (float*)malloc(h * w * c * sizeof(float));
  for (auto c_ = 0; c_ < c; c_++) {
    auto m = mean[c_];
    auto v = var[c_];
    auto gm = gamma[c_];
    auto bi = bias[c_];
    for (auto hw = 0; hw < h * w; hw++) {
      auto data = img[hw * c + c_];
      auto data_ = (data - m) / sqrt(v + 1e-5);
      data_ = data_ * gm + bi;
      out[hw * c + c_] = data_;
    }
  }
  free(img);
  free(mean);
  free(var);
  free(gamma);
  free(bias);

#if DEBUG_SHOW
  printf("bn out: (%d, %d, %d)\n", h, w, c);
#endif
  return out;
}

float* compute_relu_layer(float* img, int len) {
#if DEBUG_SHOW
  printf("-- compute relu with %d\n", len);
#endif
  for (int i = 0; i < len; i++) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
  return img;
}