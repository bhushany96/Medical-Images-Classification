# Importing Libraries

from __future__ import print_function
import argparse
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pickle
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from torchvision.models import resnet34
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import PIL
import random
import logging
logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

# WandB – Import the wandb library
import wandb

# Training Loop:

def train(args, model, device, train_loader, optimizer, epoch):
    num_correct = 0  # keep track of correctly classified Images
    train_loss = 0
    model.train()  # setting model to train mode
    for imgs, labels in train_loader:  # trainloader returns a tuple -> (batch of images, corresponding vector of labels)
        # Feed-forward Section
        imgs = imgs.to(device)  # shift images to GPU for faster training
        labels = labels.to(device)  # shift labels to GPU for faster training
        outputs = model(imgs)  # output of feed-forward neural network before softmax layer
        pred = outputs.max(1, keepdim=True)[1]  # extract digit index with the highest probability
        num_correct += pred.eq(
            labels.view_as(pred)).sum().item()
        # Back-propagation Section
        loss = F.cross_entropy(outputs, labels)  # calculate the softmax output and loss per batch of the images
        train_loss+=loss.item()
        optimizer.zero_grad()  # set the gradients matrix to zero before calculating the gradients for every batch
        loss.backward()  # calculate the gradients through differentiation (dL/dW)
        optimizer.step()  # update of weights (w = w - dL/dW)

    wandb.log({
        "Train Accuracy": 100. * num_correct / len(train_loader.dataset),
        "Train Loss": train_loss})
# Testing Loop :

def test(args, model, device, test_loader, classes):
    num_correct = 0  # keep track of correctly classified Images
    test_loss = 0  # keep track of test loss
    model.eval()  # set model in evaluation mode for test accuracy calculation

    example_images = []
    with torch.no_grad():  # no gradient calculations required during testing
        for imgs, labels in test_loader:
            imgs = imgs.to(device)  # shift images to GPU
            labels = labels.to(device)  # shift labels to GPU
            scores = model(imgs)  # predictions vector containing probability of each digit
            test_loss += F.cross_entropy(scores, labels, reduction='mean').item()
            pred = scores.max(1, keepdim=True)[1]  # extract digit index with the highest probability
            num_correct += pred.eq(
                labels.view_as(pred)).sum().item()  # calculating correctly classified images from the batch
            # WandB – Log images in your test dataset automatically, along with predicted and true labels by passing pytorch tensors with image data into wandb.Image
            # example_images.append(wandb.Image(
            #     imgs[:16], caption="Pred: {} Truth: {}".format([classes[pred[i].item()] for i in pred[:16]], [classes[labels[i]] for i in labels[:16]])))
    # WandB – wandb.log(a_dict) logs the keys and values of the dictionary passed in and associates the values with a step.
    # You can log anything by passing it to wandb.log, including histograms, custom matplotlib objects, images, video, text, tables, html, pointclouds and other 3D objects.
    # Here we use it to log test accuracy, loss and some test images (along with their true and predicted labels).
    wandb.log({
        # "Examples": example_images,
        "Test Accuracy": 100. * num_correct / len(test_loader.dataset),
        "Test Loss": test_loss})

if __name__ == '__main__':
    # First we define the transformations to apply to our images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # load datasets
    train_dataset = ImageFolder("Image Classification Data/data/train", transform)
    test_dataset = ImageFolder("Image Classification Data/data/test", transform)

    # WandB – Initialize a new run
    wandb.init(entity="vsgrover", project="pytorch-cell-classification")
    wandb.watch_called = False  # Re-run the model without restarting the runtime, unnecessary after our next release

    # WandB – Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config  # Initialize config
    config.batch_size = 32  # input batch size for training (default: 64)
    config.test_batch_size = 1000  # input batch size for testing (default: 1000)
    config.epochs = 30  # number of epochs to train (default: 10)
    config.lr = 0.001  # learning rate (default: 0.01)
    # config.momentum = 0.1          # SGD momentum (default: 0.5)
    config.no_cuda = False  # disables CUDA training
    config.seed = 42  # random seed (default: 42)
    config.log_interval = 10  # how many batches to wait before logging training status


    def main():
        use_cuda = not config.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

        # Set random seeds and deterministic pytorch for reproducibility
        # random.seed(config.seed)       # python random seed
        torch.manual_seed(config.seed)  # pytorch random seed
        # numpy.random.seed(config.seed) # numpy random seed
        torch.backends.cudnn.deterministic = True

        # Now we load our training and test datalaoders
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=True, **kwargs)

        classes = train_dataset.classes

        ## loading pretrained resnet18 model
        model = resnet34(pretrained=True).to(device)

        ## setting grad requirement for weights to false to avoid backpropagation through those layers
        for param in model.parameters():
            param.requires_grad = False

        ## grabbing number of neurons in the second last layer
        hidden_units_second_last = model.fc.in_features
        model.fc = nn.Linear(hidden_units_second_last, len(classes)).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        # WandB – wandb.watch() automatically fetches all layer dimensions, gradients, model parameters and logs them automatically to your dashboard.
        # Using log="all" log histograms of parameter values in addition to gradients
        wandb.watch(model, log="all")

        for epoch in range(1, config.epochs + 1):
            train(config, model, device, train_loader, optimizer, epoch)
            test(config, model, device, test_loader, classes)

        # WandB – Save the model checkpoint. This automatically saves a file to the cloud and associates it with the current run.
        torch.save(model.state_dict(), "model_resnet34_cf.h5")
        wandb.save('model_resnet34_cf.h5')


    main()