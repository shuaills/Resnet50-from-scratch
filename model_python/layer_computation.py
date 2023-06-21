from data_preparation import load_conv_weight, load_conv_param, load_data_from_file
from layer_operations import my_conv2d, my_fc, my_bn, compute_relu_layer, my_avg_pool, my_max_pool

def compute_conv_layer(in_data, layer_name):
  print("-- compute " + layer_name)
  weight = load_conv_weight(layer_name)
  param = load_conv_param(layer_name)
  # ci, co, kernel, stride, pad
  hi = in_data.shape[0]
  wi = in_data.shape[1]
  ci = param[0]
  co = param[1]
  kernel = param[2]
  stride = param[3]
  pad = param[4]
  res = my_conv2d(in_data, weight, hi, wi, ci, co, kernel, stride, pad)
  print(res.shape)
  return res

def compute_fc_layer(in_data, layer_name):
  print("-- compute " + layer_name)
  weight_file_name = "../model/resnet50_weight/resnet50_" + layer_name + "_weight.txt"
  bias_file_name = "../model/resnet50_weight/resnet50_" + layer_name + "_bias.txt"
  weight = load_data_from_file(weight_file_name)
  bias = load_data_from_file(bias_file_name)
  res = my_fc(in_data, weight, bias)
  print(res.shape)
  return res

def compute_bn_layer(in_data, layer_name):
  print("-- compute " + layer_name)
  weight = load_conv_weight(layer_name)
  weight_file_name = "../model/resnet50_weight/resnet50_" + layer_name + "_weight.txt"
  bias_file_name = "../model/resnet50_weight/resnet50_" + layer_name + "_bias.txt"
  mean_file_name = "../model/resnet50_weight/resnet50_" + layer_name + "_running_mean.txt"
  var_file_name = "../model/resnet50_weight/resnet50_" + layer_name + "_running_var.txt"
  weight = load_data_from_file(weight_file_name)
  bias = load_data_from_file(bias_file_name)
  mean = load_data_from_file(mean_file_name)
  var = load_data_from_file(var_file_name)
  res = my_bn(in_data, mean, var, weight, bias)  
  print(res.shape)
  return res

def compute_maxpool_layer(in_data):
  print("-- compute maxpool")
  res = my_max_pool(in_data)  
  print(res.shape)
  return res

def compute_avgpool_layer(in_data):
  print("-- compute avgpool")
  res = my_avg_pool(in_data)
  print(res.shape)
  return res

def compute_bottleneck(in_data, bottleneck_layer_name, down_sample = False):
  print("compute " + bottleneck_layer_name)
  out = compute_conv_layer(in_data, bottleneck_layer_name + "_conv1")
  out = compute_bn_layer(out, bottleneck_layer_name + "_bn1")
  out = compute_relu_layer(out)
  out = compute_conv_layer(out, bottleneck_layer_name + "_conv2")
  out = compute_bn_layer(out, bottleneck_layer_name + "_bn2")
  out = compute_relu_layer(out)
  out = compute_conv_layer(out, bottleneck_layer_name + "_conv3")
  bn_out = compute_bn_layer(out, bottleneck_layer_name + "_bn3")

  if down_sample == True:
    conv_out= compute_conv_layer(in_data, bottleneck_layer_name + "_downsample_conv2d")
    short_cut_out = compute_bn_layer(conv_out, bottleneck_layer_name + "_downsample_batchnorm")
    bn_out = bn_out + short_cut_out
  else:
    bn_out = bn_out + in_data
  return compute_relu_layer(bn_out)