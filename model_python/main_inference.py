from data_preparation import getPicList
from layer_computation import (compute_conv_layer, compute_bn_layer, compute_relu_layer, compute_maxpool_layer,
compute_bottleneck, compute_avgpool_layer, compute_fc_layer)
from data_preparation import preprocess


pic_to_predice = getPicList()

for filename in pic_to_predice:
  print("begin predice with " + filename)
  out = preprocess(filename)

  out = compute_conv_layer(out, "conv1")
  out = compute_bn_layer(out, "bn1")
  # print("-----------------------bn -------------")
  # print(out)
  # exit()

  out = compute_relu_layer(out)
  out = compute_maxpool_layer(out)

  # layer1 
  out = compute_bottleneck(out, "layer1_bottleneck0", down_sample = True)
  out = compute_bottleneck(out, "layer1_bottleneck1", down_sample = False)
  out = compute_bottleneck(out, "layer1_bottleneck2", down_sample = False)

  # layer2
  out = compute_bottleneck(out, "layer2_bottleneck0", down_sample = True)
  out = compute_bottleneck(out, "layer2_bottleneck1", down_sample = False)
  out = compute_bottleneck(out, "layer2_bottleneck2", down_sample = False)
  out = compute_bottleneck(out, "layer2_bottleneck3", down_sample = False)

  # layer3
  out = compute_bottleneck(out, "layer3_bottleneck0", down_sample = True)
  out = compute_bottleneck(out, "layer3_bottleneck1", down_sample = False)
  out = compute_bottleneck(out, "layer3_bottleneck2", down_sample = False)
  out = compute_bottleneck(out, "layer3_bottleneck3", down_sample = False)
  out = compute_bottleneck(out, "layer3_bottleneck4", down_sample = False)
  out = compute_bottleneck(out, "layer3_bottleneck5", down_sample = False)
  
  # layer4
  out = compute_bottleneck(out, "layer4_bottleneck0", down_sample = True)
  out = compute_bottleneck(out, "layer4_bottleneck1", down_sample = False)
  out = compute_bottleneck(out, "layer4_bottleneck2", down_sample = False)

  # avg pool
  out = compute_avgpool_layer(out)
  # Linear
  out = compute_fc_layer(out, "fc")

  # find inference result
  out_res = list(out)
  max_value = max(out_res)
  index = out_res.index(max_value)
  
  print("\npredict picture: " + filename)
  print("      max_value: " + str(max_value))
  print("          index: " + str(index))
  
  # Read the categories
  with open("imagenet_classes.txt", "r") as f:
      categories = [s.strip() for s in f.readlines()]
      print("         result: " + categories[index])