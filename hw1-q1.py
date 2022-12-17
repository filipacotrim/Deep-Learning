#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import random
import os

import numpy as np
import matplotlib.pyplot as plt

import utils

import warnings #runtime erros ?? TODO: fix 


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def CrossEntropy(output, y):
    output -= np.max(output) # to prevent NaN values
    # softmax transformation. 
    warnings.filterwarnings('ignore')

    probs = np.exp(output) / np.sum(np.exp(output))
    loss = np.dot(-y,np.log(probs))
    return loss  

def ReLU(x):
    #returns 0 if output is negative, otherwise return output as it is 
    temp = [max(0,value) for value in x]
    return np.array(temp, dtype=float)


def forward(x, weights, biases):
    num_layers = len(weights)
    hiddens = []
    for i in range(num_layers):
        h = x if i == 0 else hiddens[i-1]
        z = weights[i].dot(h) + biases[i]
        if i < num_layers-1:  # Assume the output layer has no activation.
            hiddens.append(ReLU(z))

    output = z
    return output, hiddens

def backward(weights, x, y, output, hiddens, num_layers):
    #num_layers = len(weights)
    z = output
 
    probs = np.exp(output) / np.sum(np.exp(output))
    grad_z = probs - y  # Grad of loss wrt last z.

    grad_weights = []
    grad_biases = []

    g = ReLU(z)
    for i in range(num_layers-1, -1, -1):
        # Gradient of hidden parameters.
        h = x if i == 0 else hiddens[i-1]
        grad_weights.append(grad_z[:, None].dot(h[:, None].T))
        grad_biases.append(grad_z)

        # Gradient of hidden layer below.
        grad_h = weights[i].T.dot(grad_z)

        # Gradient of hidden layer below before activation.
        #assert(g == ReLU(z)) 
        grad_z = grad_h * (1-h**2)   # Grad of loss wrt z3.


    grad_weights.reverse()
    grad_biases.reverse()
    return grad_weights, grad_biases


def update_parameters(weights, biases, grad_weights, grad_biases, learning_rate):
    num_layers = len(weights)
    for i in range(num_layers):
        weights[i] -= learning_rate*grad_weights[i]
        biases[i] -= learning_rate*grad_biases[i]

def predict_label(output):
    y_hat = np.zeros_like(output)
    y_hat[np.argmax(output)] = 1
    return y_hat


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
        
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        rate = 1
        
        y_pred = np.argmax(self.W.dot(x_i))
        
        if y_pred != y_i:
            self.W[y_i,:] +=  rate * x_i
            self.W[y_pred,:] -= rate * x_i
        
class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Label scores according to the model (num_labels x 1).
        label_scores = self.W.dot(x_i)[:, None]
        
        # One-hot vector with the true label (num_labels x 1).
        y_one_hot = np.zeros((np.size(self.W, 0), 1))
        y_one_hot[y_i] = 1
        
        # Softmax function.
        # This gives the label probabilities according to the model (num_labels x 1).
        label_probabilities = np.exp(label_scores) / np.sum(np.exp(label_scores))
        # SGD update. W is num_labels x num_features.
        self.W += learning_rate * (y_one_hot - label_probabilities) * x_i[None, :]


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size, layers):
        # Initialize an MLP with a single hidden layer.
        self.W = [np.random.normal(0.1,0.1**2,(hidden_size, n_features)),np.random.normal(0.1,0.1**2,(n_classes, hidden_size))]
        self.B = [np.zeros(hidden_size),np.zeros(n_classes)]

        #self.learning_rate = 0.001
        #raise NotImplementedError

    def predict(self, X):
        # forward
        predicted_labels = []
        for x in X:
            z = []
            output, hiddens = forward(x, self.W, self. B)

            # compute loss
            probs = np.exp(output) / np.sum(np.exp(output),keepdims=True)
            
            # predict
            y_hat = np.argmax(probs)
            predicted_labels.append(y_hat)
        predicted_labels = np.array(predicted_labels)
        print("predicted: ",predicted_labels)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


    def train_epoch(self, X, y, learning_rate = 0.001):
        total_loss = 0
        count = 0

        for x, y_i in zip(X, y):
            count += 1

            # forward
            output, hiddens = forward(x, self.W, self.B)

            # compute loss
            loss = CrossEntropy(output, y_i)
            total_loss += loss

            num_layers = len(self.W)
            # backward
            grad_weights, grad_biases = backward(self.W, x, y_i, output, hiddens, num_layers)

            # update parameters
            update_parameters(self.W, self.B, grad_weights, grad_biases, learning_rate)

def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.show()
    plt.savefig('question_1.2.b.png')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-layers', type=int, default=1,
                        help="""Number of hidden layers (needed only for MLP,
                        not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_classification_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size  # 10
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size, opt.layers)

    # TODO: Encode labels as one-hot vectors. ???

    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []
    for i in epochs:
        print('---- Training epoch {} ----'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        model.train_epoch(
            train_X,
            train_y,
            learning_rate=opt.learning_rate
        )
        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))
        print(valid_accs, test_accs)

    print(model.W)
    
    # plot
    plot(epochs, valid_accs, test_accs)


if __name__ == '__main__':
    main()
