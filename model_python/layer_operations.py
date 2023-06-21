import numpy as np

# Load the weights from the text files into a list.

#Conv2d

def my_conv2d(img, weight, hi, wi, ci, co, kernel, stride, pad):
  ho = (hi + 2 * pad - kernel) // stride + 1
  wo = (wi + 2 * pad - kernel) // stride + 1
  
  weight = np.array(weight).reshape(co, kernel, kernel, ci)
  img_pad = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), 'constant')
  img_out = np.zeros((ho, wo, co))

  for co_ in range(co):
    for ho_ in range(ho):
      in_h_origin = ho_ * stride - pad
      for wo_ in range(wo):
        in_w_origin = wo_ * stride - pad
        filter_h_start = max(0, -in_h_origin)
        filter_w_start = max(0, -in_w_origin)
        filter_h_end = min(kernel, hi - in_h_origin)
        filter_w_end = min(kernel, wi - in_w_origin)
        acc = float(0)
        for kh_ in range(filter_h_start, filter_h_end):
          hi_index = in_h_origin + kh_
          for kw_ in range(filter_w_start, filter_w_end):
            wi_index = in_w_origin + kw_
            # use vdot to optimize MAC operation
            acc += np.vdot(img[hi_index][wi_index], weight[co_][kh_][kw_])
            #for ci_ in range(ci):
            #  in_data = img[hi_index][wi_index][ci_]
            #  weight_data = weight[co_][kh_][kw_][ci_]
            #  acc = acc + in_data * weight_data
        img_out[ho_][wo_][co_] = acc
  return img_out

def my_fc(img, weight, bias):
  '''
  fc compute [2048] * [1000, 2048] = [1000]
  img : [1, 1, 2048] from last layer
  weight: need reshpe to [1000, 2048]
  bias: [1000]
  '''
  img_new = img.reshape(2048)
  weight_new = np.array(weight).reshape([1000, 2048])
  bias_new = np.array(bias).reshape(1000)
  out = np.zeros(1000)

  for i in range(1000):
    # use vdot to optimize MAC operation
    sum_x = np.vdot(img_new, weight_new[i])
    out[i] = sum_x + bias_new[i]
    #sum_x = float(0)
    #for j in range(2048):
    #  l = img_new[j]
    #  r = weight_new[i][j]
    #  sum_x = sum_x + l * r
    #out[i] = sum_x + bias_new[i]
  return out

def my_max_pool(img):
  hi  = img.shape[0]
  wi = img.shape[1]
  channel = img.shape[2]
  pad = 1
  stride = 2
  kernel = 3
  ho = (hi + 2 * pad - kernel) // stride + 1
  wo = (wi + 2 * pad - kernel) // stride + 1
  
  img_pad = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), 'constant')
  img_out = np.zeros((ho, wo, channel))

  for c_ in range(channel):
    for ho_ in range(ho):
      in_h_origin = ho_ * stride - pad
      for wo_ in range(wo):
        in_w_origin = wo_ * stride - pad
        filter_h_start = max(0, -in_h_origin)
        filter_w_start = max(0, -in_w_origin)
        filter_h_end = min(kernel, hi - in_h_origin)
        filter_w_end = min(kernel, wi - in_w_origin)
        max_x = float(0)
        for kh_ in range(filter_h_start, filter_h_end):
          hi_index = in_h_origin + kh_
          for kw_ in range(filter_w_start, filter_w_end):
            wi_index = in_w_origin + kw_
            in_data = img[hi_index][wi_index][c_]
            max_x = max(in_data, max_x)
        img_out[ho_][wo_][c_] = max_x 
  return img_out

def my_avg_pool(img):
  hi  = img.shape[0]
  wi = img.shape[1]
  channel = img.shape[2]
  pad = 0
  stride = 1
  kernel = 7
  ho = (hi + 2 * pad - kernel) // stride + 1
  wo = (wi + 2 * pad - kernel) // stride + 1

  img_pad = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), 'constant')
  img_out = np.zeros((ho, wo, channel))

  for c_ in range(channel):
    for ho_ in range(ho):
      in_h_origin = ho_ * stride - pad
      for wo_ in range(wo):
        in_w_origin = wo_ * stride - pad
        filter_h_start = max(0, -in_h_origin)
        filter_w_start = max(0, -in_w_origin)
        filter_h_end = min(kernel, hi - in_h_origin)
        filter_w_end = min(kernel, wi - in_w_origin)
        sum_x = float(0)
        for kh_ in range(filter_h_start, filter_h_end):
          hi_index = in_h_origin + kh_
          for kw_ in range(filter_w_start, filter_w_end):
            wi_index = in_w_origin + kw_
            in_data = img[hi_index][wi_index][c_]
            sum_x = sum_x + in_data
        img_out[ho_][wo_][c_] = sum_x / (kernel * kernel)
  return img_out

def my_bn(img, mean, var, gamma, bias):
  h = img.shape[0]
  w = img.shape[1]
  c = img.shape[2]

  for c_ in range(c):
    data = img[:, :, c_]
    data_ = (data - mean[c_]) / (pow(var[c_] + 1e-5, 0.5))
    data_ = data_ * gamma[c_]
    data_ = data_ + bias[c_]
    img[:, :, c_] = data_
  return img
    
def compute_relu_layer(img):
  print("-- compute relu")
  res = np.maximum(0, img)
  print(res.shape)
  return res