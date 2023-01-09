#!/usr/bin/env python

# Deep Learning Homework 2

import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
import numpy as np

import utils

class CNN(nn.Module):
    
    def __init__(self, dropout_prob):
        """
        The __init__ should be used to declare what kind of layers and other
        parameters the module has. For example, a CNN module has convolution,
        max pooling, activation, linear, and other types of layers. For an 
        idea of how to us pytorch for this have a look at
        https://pytorch.org/docs/stable/nn.html
        """
        super(CNN, self).__init__()

        #conv1 with 8 output channels, kernel of size 5*5, stride of 1 
        #padding: (2 x Padding + N - Kernel)/Stride + 1 = 28 <=> Padding = 2
        self.conv1 = nn.Conv2d(1, 8, 5, padding=2)
        self.max_pool = nn.MaxPool2d(2,2)

        #conv2 with 16 output channels, kernel of size 3x3, stride of 1 
        self.conv2 = nn.Conv2d(8, 16, 3, padding = 0)

        # input features = #output_channels x output_width x output_height
        #output_width = (input_width + padding_right + padding_left - kernel_width) / Stride + 1
        #output_height = (input_height + padding_height_top + padding_height_bottom - kernel_height) / Stride + 1
        self.fc1 = nn.Linear(6*6*16, 600)
        self.dropout = nn.Dropout2d(p=dropout_prob)
        self.fc2 = nn.Linear(600, 120)
        self.fc3 = nn.Linear(120, 10)    

                
    def forward(self, x):
        """
        x (batch_size x n_channels x height x width): a batch of training 
        examples

        Every subclass of nn.Module needs to have a forward() method. forward()
        describes how the module computes the forward pass. This method needs 
        to perform all the computation needed to compute the output from x. 
        This will include using various hidden layers, pointwise nonlinear 
        functions, and dropout. Don't forget to use logsoftmax function before 
        the return

        One nice thing about pytorch is that you only need to define the
        forward pass -- this is enough for it to figure out how to do the
        backward pass.
        """
        # 3d: [batch_size, channels, num_features (aka: H * W)]
        # 4d: [batch_size, channels, height, width]  --->  x.shape = [1, 1, 8, 784]
        x = x.view(x.shape[0], 1, 28, 28)    
        x = F.relu(self.max_pool(self.conv1(x)))
        # CONV:
        # (28-5+2*2)/1 + 1 = 28 
        # (28-5+2*2)/1 + 1 = 28  
        #   ---> x.shape = [N, 8, 28, 28]
        # POOL:
        # (28-2+2*0)/2 + 1 = 14   
        # (28-2+2*0)/2 + 1 = 14 
        #   ---> x.shape = [N, 8, 14, 14]        

        x = F.relu(self.max_pool(self.conv2(x)))
        # CONV:
        # (14-3+2*0)/1 + 1 = 12 
        # (14-3+2*0)/1 + 1 = 12 
        #   ---> x.shape = [N, 16, 12, 12] 
        # POOL:
        # (12-2+2*0)/2 + 1 = 6 
        # (12-2+2*0)/2 + 1 = 6
        #   ---> x.shape = [N, 16, 6, 6] 

        #flatten the output from previous layer and slide it through only set of fully connected - relu layer
        x = torch.flatten(x, 1) 
        x = x.view(-1, 6*6*16)        
        x = self.fc1(x)
        x = self.dropout(F.relu(x))
        x =  F.log_softmax(self.fc3(F.relu(self.fc2(x))), dim=1)

        return x

def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    model: a PyTorch defined model
    optimizer: optimizer used in gradient step
    criterion: loss function

    To train a batch, the model needs to predict outputs for X, compute the
    loss between these predictions and the "gold" labels y using the criterion,
    and compute the gradient of the loss with respect to the model parameters.

    Check out https://pytorch.org/docs/stable/optim.html for examples of how
    to use an optimizer object to update the parameters.

    This function should return the loss (tip: call loss.item()) to get the
    loss as a numerical value that is not part of the computation graph.
    """
    scores = model(X)
         
    # initializing
    optimizer.zero_grad()

    #print("scores: ",scores.shape, "y: ", y.shape)
    #computing the loss 
    loss = criterion(scores, y)
    
    # compute the gradients
    loss.backward()
    
    # updating the weights
    optimizer.step()
   
    return loss.item()

def predict(model, X):
    """X (n_examples x n_features)"""
    scores = model(X)  # (n_examples x n_classes)
    predicted_labels = scores.argmax(dim=-1)  # (n_examples)
    return predicted_labels


def evaluate(model, X, y):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    """
    model.eval()
    y_hat = predict(model, X)
    n_correct = (y == y_hat).sum().item()
    n_possible = float(y.shape[0])
    model.train()
    return n_correct / n_possible


def plot(epochs, plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def plot_feature_maps(model, train_dataset):
    
    model.conv1.register_forward_hook(get_activation('conv1'))
    
    data, _ = train_dataset[4]
    data.unsqueeze_(0)
    output = model(data)

    plt.imshow(data.reshape(28,-1)) 
    plt.savefig('original_image.pdf')

    k=0
    act = activation['conv1'].squeeze()
    fig,ax = plt.subplots(2,4,figsize=(12, 8))
    
    for i in range(act.size(0)//3):
        for j in range(act.size(0)//2):
            ax[i,j].imshow(act[k].detach().cpu().numpy())
            k+=1  
            plt.savefig('activation_maps.pdf') 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-batch_size', default=8, type=int,
                        help="Size of training batch.")
    parser.add_argument('-learning_rate', type=float, default=0.01,
                        help="""Learning rate for parameter updates""")
    parser.add_argument('-l2_decay', type=float, default=0)
    parser.add_argument('-dropout', type=float, default=0.8)
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='sgd')
    
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    data = utils.load_classification_data()
    dataset = utils.ClassificationDataset(data)
    train_dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True)
    dev_X, dev_y = dataset.dev_X, dataset.dev_y
    test_X, test_y = dataset.test_X, dataset.test_y

    # initialize the model
    model = CNN(opt.dropout)
    
    # get an optimizer
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay
    )
    
    # get a loss criterion
    criterion = nn.NLLLoss()
    
    # training loop
    epochs = np.arange(1, opt.epochs + 1)
    train_mean_losses = []
    valid_accs = []
    train_losses = []
    for ii in epochs:
        print('Training epoch {}'.format(ii))
        for X_batch, y_batch in train_dataloader:
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion)
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % (mean_loss))

        train_mean_losses.append(mean_loss)
        valid_accs.append(evaluate(model, dev_X, dev_y))
        print('Valid acc: %.4f' % (valid_accs[-1]))

    print('Final Test acc: %.4f' % (evaluate(model, test_X, test_y)))
    # plot
    config = "{}-{}-{}-{}".format(opt.learning_rate, opt.dropout, opt.l2_decay, opt.optimizer)

    plot(epochs, train_mean_losses, ylabel='Loss', name='CNN-training-loss-{}'.format(config))
    plot(epochs, valid_accs, ylabel='Accuracy', name='CNN-validation-accuracy-{}'.format(config))
    
    plot_feature_maps(model, dataset)

if __name__ == '__main__':
    main()
