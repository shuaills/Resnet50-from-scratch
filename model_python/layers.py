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

