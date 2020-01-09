import argparse

import numpy as np
import pandas as pd
import time
import json
import torch
import torch.nn.functional as F
import torch.utils.data

from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

from PIL import Image
from collections import OrderedDict

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', dest="data_dir", help='Path for the dataset', type=str, default="./flowers")
parser.add_argument('--save_dir', dest="save_dir", help='Path where the model will be saved', type=str, default="./checkpoint.pth")
parser.add_argument('--arch', dest="arch", help='Architecture for the neural network', type=str, choices=['vgg11', 'vgg16', 'densenet121', 'alexnet'], default="vgg16")
parser.add_argument('--learning_rate', dest="learning_rate", help='Learning rate of the model', type=float, default=0.0001)
parser.add_argument('--dropout', dest="dropout", help='Dropout rate of the model', type=float, default=0.05)
parser.add_argument('--hidden_units', dest="hidden_units", help='Number of hidden units this NN will have', type=int, default=500)
parser.add_argument('--input_units', dest="input_units", help='Number of input units this NN will have', type=int, default=25088)
parser.add_argument('--output_units', dest="output_units", help='Number of output units this NN will have', type=int, default=102)
parser.add_argument('--epochs', dest="epochs", help='Number of epochs to run', type=int, default=10)
parser.add_argument('--gpu', dest="gpu", help='The hardware the neural network will use to run', type=str, default="gpu")

args = parser.parse_args()

data_dir = args.data_dir #"./flowers" #args.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


train_data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_data_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

test_data_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


#Load the datasets with ImageFolder
train_image_datasets = datasets.ImageFolder(train_dir, transform=train_data_transforms)
valid_image_datasets = datasets.ImageFolder(valid_dir, transform=valid_data_transforms)
test_image_datasets = datasets.ImageFolder(test_dir, transform=test_data_transforms)

#Using the image datasets and the trainforms, define the dataloaders
train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size=60, shuffle=True)
valid_dataloaders = torch.utils.data.DataLoader(valid_image_datasets, batch_size=60, shuffle=False)
test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size=60, shuffle=False)
                    
#if no model is chosen, set it by default to alexnet
if args.arch == "":
    print("You didn't choose a model, so it is being set to alexnet by default!")
    args.arch = "alexnet"

# Load pretrained model
trained_model = args.arch
model = getattr(models,trained_model)(pretrained=True)

# freeze parameters of the model
for parameters in model.parameters():
    parameters.required_grad = False
    
#setting the number of input units based on the architecture
if trained_model == "vgg16":
    args.input_units = 25088
elif trained_model == "densenet121":
    args.input_units = 1024
else:
    args.input_units = 9216
    
# untrained feed-forward network as a classifier, using ReLU activations and dropout
classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(args.input_units, args.hidden_units, bias=True)),
                              ('relu1', nn.ReLU()),
                              ('dropout1', nn.Dropout(p=args.dropout)),
                              ('fc2', nn.Linear(args.hidden_units, args.output_units, bias=True)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
                    
# feedforward network for use as a classifier using the features as input
model.classifier = classifier

images, labels = next(iter(test_dataloaders))

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

epochs = args.epochs
steps = 0
                    
#show the model chosen
print(trained_model)
                    
if torch.cuda.is_available() and args.gpu.lower() == 'gpu':
    device = torch.device('cuda')
else:
    device = 'cpu'

model.to(device)                 

train_losses, test_losses = [], []
for e in range(args.epochs):
    running_loss = 0
    
    for images, labels in train_dataloaders:
        optimizer.zero_grad()
        
        images, labels = images.to(device), labels.to(device)
        
        log_ps = model.forward(images)
        
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    else:
        test_loss = 0
        accuracy = 0
        
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            for ii, (images, labels) in enumerate(test_dataloaders):
                
                images, labels = images.to(device), labels.to(device)
                
                log_ps = model.forward(images)
                test_loss += criterion(log_ps, labels)
                
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                #equals = top_class == labels.view(*top_class.shape)
                equals = (labels.data == ps.max(dim=1)[1])
                accuracy += torch.mean(equals.type(torch.FloatTensor))
                 
        train_losses.append(running_loss/len(train_dataloaders))
        test_losses.append(test_loss/len(test_dataloaders))

        print("Epoch: {}/{}.. ".format(e+1, args.epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/len(train_dataloaders)),
                  "Test Loss: {:.3f}.. ".format(test_loss/len(test_dataloaders)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(test_dataloaders)))
        
#Do validation on the test set
correct = 0
total = 0
model.to(device)

with torch.no_grad():
    for data in test_dataloaders:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        # Get probabilities
        outputs = model(images)
        # Turn probabilities into predictions
        _, predicted_outcome = torch.max(outputs.data, 1)
        # Total number of images
        total += labels.size(0)
        # Count number of cases in which predictions are correct
        correct += (predicted_outcome == labels).sum().item()

print(f"Test accuracy of model: {round(100 * correct / total,3)}%")

#Save model
model.class_to_idx = train_image_datasets.class_to_idx 

checkpoint = {'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'mapping':    model.class_to_idx,
              'arch': args.arch
             }        

torch.save(checkpoint, 'checkpoint.pth')
