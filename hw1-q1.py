#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import random
import os

import numpy as np
import matplotlib.pyplot as plt

import utils


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


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
    def __init__(self, n_classes, n_features, hidden_size):
        self.W = [np.random.normal(loc=0.1,scale=0.1,size=(hidden_size, n_features)),
                  np.random.normal(loc=0.1,scale=0.1,size=(n_classes, hidden_size))]
        self.B = [np.zeros(hidden_size), np.zeros(n_classes)]

    def ReLU(self, x):
        return (x > 0) * x 

    def derivateReLu(self, x):
        return (x > 0) * 1
        
    def compute_label_probabilities(self, output):
        # softmax transformation.
        probs = np.exp(output) / np.sum(np.exp(output))
        return probs

    def forward(self, x):
        hiddens = []
        num_layers = len(self.W)

        for i in range(num_layers):
            h = x if i == 0 else hiddens[i-1]
            z = self.W[i].dot(h) + self.B[i]
            if i < num_layers-1:  
                hiddens.append(self.ReLU(z))
        
        return z, hiddens

    def backward(self,x, y, output, hiddens):
        output -= output.max()
        probs = self.compute_label_probabilities(output)

        y_one_hot_vector = np.zeros((10,))
        y_one_hot_vector[y] = 1
        
        grad_z = probs - y_one_hot_vector # Grad of loss w.r.t. last z

        grad_weights = []
        grad_biases = []

        num_layers = len(self.W)
        for i in range(num_layers-1, -1, -1):
            # dL/dW = dL/dz . h^T
            h = x if i == 0 else hiddens[i-1]
            grad_weights.append(grad_z[:, None].dot(h[:, None].T))
            
            # dL/db = dL/dz
            grad_biases.append(grad_z)

            # dL/dh[i-1] = W[i]^T . dL/dz[i] 
            grad_h = self.W[i].T.dot(grad_z)
            # dL /dz = dL/dh * dh/dz
            grad_z = grad_h * self.derivateReLu(h)

        grad_weights.reverse()
        grad_biases.reverse()
        return grad_weights, grad_biases

    def predict_label(self, output):
        # The most probable label is also the label with the largest logit.
        y_hat = np.zeros_like(output)
        y_hat[np.argmax(output)] = 1
        return y_hat

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        predicted_labels = np.empty(X.shape[0])
        for i in range(X.shape[0]):
            output, _ = self.forward(X[i])
            y_hat = np.argmax(self.predict_label(output))
            predicted_labels[i] = y_hat
        return predicted_labels  

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def update_parameters(self, grad_weights, grad_biases, learning_rate):
        num_layers = len(self.W)
        for i in range(num_layers):
            self.W[i] -= learning_rate*grad_weights[i]
            self.B[i] -= learning_rate*grad_biases[i]
    
    def train_epoch(self, X, y, learning_rate=0.001):
        for x, y in zip(X, y):
            output, hiddens = self.forward(x)
            grad_weights, grad_biases = self.backward(x, y, output, hiddens)
            self.update_parameters(grad_weights, grad_biases, learning_rate)
   
def plot(epochs, valid_accs, test_accs, model):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    #plt.show()
    if model == 'perceptron':
        plt.savefig('results_1/question_1.1.a.png')
    elif model == 'logistic_regression':
        plt.savefig('results_1/question_1.1.b.png')
    else:
        plt.savefig('results_1/question_1.2.b.png')

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
        # model = MLP(n_classes, n_feats, opt.hidden_size, opt.layers)
        model = MLP(n_classes, n_feats, opt.hidden_size)

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

    print("Final validation accuracy: ", valid_accs[-1])
    print("Final testing accuracy: ", test_accs[-1])
    # plot
    plot(epochs, valid_accs, test_accs, opt.model)


if __name__ == '__main__':
    main()
