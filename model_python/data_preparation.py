import os
from PIL import Image
from torchvision import transforms
import numpy as np

# load data from txt
def load_data_from_file(file_name, is_float = True):
  k = []
  with open(file_name, 'r') as f_:
    lines = f_.readlines()
    k = [float(l) for l in lines ]
    if is_float == False:
      k = [int(l) for l in k]
    #if is_float == True:
    #  for l in lines:
    #    k.append(float(l))
    #else:
    #  for l in lines:
    #    k.append(int(float(l)))
  return k

def load_conv_weight(name):
  name = "../model/resnet50_weight/resnet50_" + name + "_weight.txt"
  return load_data_from_file(name, is_float = True)

def load_conv_param(name):
  name = "../model/resnet50_weight/resnet50_" + name + "_param.txt"
  param = load_data_from_file(name, is_float = False)
  return param

def getPicList():
  import os
  pic_dir = "../pics/"
  file_to_predict = [pic_dir + f for f in os.listdir(pic_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
  return file_to_predict

# pre-process for pictures
def preprocess(filename):
  from PIL import Image
  from torchvision import transforms
  img = Image.open(filename)
  preprocess = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
  input_tensor = preprocess(img)
  input_batch = input_tensor.unsqueeze(0)
  
  out = np.array(input_batch)
  out = np.transpose(out, (0, 2, 3, 1))
  out = np.reshape(out, (224, 224, 3))
  return out