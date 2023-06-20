
import torch
import torchvision
import numpy as np
from torchvision import models
resnet50 = models.resnet50(pretrained=True)
print(resnet50)

def save(data, file):
  d = np.array(data.weight.data.cpu().numpy())
  np.savetxt(file+str(".txt"), d.reshape(-1, 1))
  
save(resnet50.conv1, "resnet50_conv1")
save(resnet50.bn1, "resnet50_bn1")

def save_bottle_neck(layer, layer_index):
  bottle_neck_idx = 0
  layer_name = "resnet50_layer" + str(layer_index) + "_bottleneck"
  for bottleNeck in layer:
    save(bottleNeck.conv1, layer_name + str(bottle_neck_idx) + "_conv1")
    save(bottleNeck.bn1, layer_name + str(bottle_neck_idx) + "_bn1")
    save(bottleNeck.conv2, layer_name + str(bottle_neck_idx) + "_conv2")
    save(bottleNeck.bn2, layer_name + str(bottle_neck_idx) + "_bn2")
    save(bottleNeck.conv3, layer_name + str(bottle_neck_idx) + "_conv3")
    save(bottleNeck.bn3, layer_name + str(bottle_neck_idx) + "_bn3")
    if bottleNeck.downsample:
      save(bottleNeck.downsample[0], layer_name + str(bottle_neck_idx) + "_downsample_conv2d")
      save(bottleNeck.downsample[1], layer_name + str(bottle_neck_idx) + "_downsample_batchnorm")
    bottle_neck_idx = bottle_neck_idx + 1

save_bottle_neck(resnet50.layer1, 1)
save_bottle_neck(resnet50.layer2, 2)
save_bottle_neck(resnet50.layer3, 3)
save_bottle_neck(resnet50.layer4, 4)

save(resnet50.fc, "resnet50.fc")