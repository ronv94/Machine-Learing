import argparse

import numpy as np
import pandas as pd

import time
import json

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

from PIL import Image
import time
import json
from collections import OrderedDict

parser = argparse.ArgumentParser()

parser.add_argument('--top_k', dest="top_k", type=int, default=5)
parser.add_argument ('--save_dir', dest="save_dir", type=str, default="./checkpoint.pth")
parser.add_argument('--image_path', dest="image_path", type=str, default="flowers/test/100/image_07899.jpg")
parser.add_argument('--arch', dest="arch", type=str, default="vgg16")
parser.add_argument ('--category_names', dest="category_names", type=str, default="cat_to_name.json")
parser.add_argument('--gpu', dest="gpu", type=str, default="gpu")

args = parser.parse_args()

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

checkpoint = torch.load(args.save_dir)

# Establish model template
trained_model = args.arch
model = getattr(models,trained_model)(pretrained=True)

model.classifier = checkpoint['classifier']
model.load_state_dict(checkpoint['state_dict'])
model.class_to_idx = checkpoint['mapping']
    
for param in model.parameters(): 
    param.requires_grad = False 

def process_image(path):
    
    #size = 256, 256
    im = Image.open(path) #loading image
    width, height = im.size #original size

    if width > height: 
        height = 256
        im.thumbnail ((50000, height), Image.ANTIALIAS)
    else: 
        width = 256
        im.thumbnail ((width,50000), Image.ANTIALIAS)

    width, height = im.size #new size of im
    #crop 224x224 in the center
    reduce = 224
    left = (width - reduce)/2 
    top = (height - reduce)/2
    right = left + 224 
    bottom = top + 224
    im = im.crop ((left, top, right, bottom))

    #preparing numpy array
    np_image = np.array (im)/255 #to make values from 0 to 1
    np_image -= np.array ([0.485, 0.456, 0.406]) 
    np_image /= np.array ([0.229, 0.224, 0.225])

    np_image= np_image.transpose ((2,0,1))
    return np_image

# Implement the code to predict the class from an image file
# Use CPU instead of GPU
model.to("cpu")

#process a desired image
image = process_image(args.image_path)

#converting a numpy array based image to a tensor
img_tensor = torch.from_numpy(image).type(torch.FloatTensor)

#make torch of right size
img_tensor = img_tensor.unsqueeze(dim=0)

with torch.no_grad():
    output = model.forward(img_tensor)
    output_probability = torch.exp(output) 

    probs, indeces = output_probability.topk(args.top_k)

    #converting probs and indeces to a numpy array
    probs = probs.numpy() 
    indeces = indeces.numpy() 

    #converting probs and indeces to a list
    probs = probs.tolist()[0]
    indeces = indeces.tolist()[0]


    mapping = {val: key for key, val in
               model.class_to_idx.items()
              }

    classes = [mapping[item] for item in indeces]
    classes = np.array(classes)

names = []
for i in classes:
    names += [cat_to_name[i]]
    

# Print the top predicted flowers with their probabilities
print(f"The top {args.top_k} flowers and their probabilities are: ")
for i in range(len(names)):
    print(f"{names[i]}\t\t{probs[i]}")

print()

# Final result based on the prediction
print(f"This flower predicted to be a '{names[0]}' with a probability of {round(probs[0]*100, 2)}% ")