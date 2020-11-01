'''
Feed-forward multi-layer neural network with backpropagation with momentum. Able 
to handle any number of hidden layers and hidden neurons (specify at run-time).

Author: Omar Alsayed (alsayeoy@mail.uc.edu).
Date: 10-25-2020
'''
import dataset

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd

from sklearn.metrics import confusion_matrix

mnist = dataset.get_sets()

class MLNN:
    def __init__(self, layers, neurons, epoch=0, epochs=500, learning_rate=0.3, min_learning_rate=0.0025, momentum=0.9):
        self.neurons = neurons # List of nerons for MLNN
        self.epoch = epoch
        self.epochs = epochs
        self.alpha = learning_rate
        self.min_alpha = min_learning_rate
        self.momentum = momentum
        self.layers = layers
        self.hyperparameters = self.initialize_dictionary()

    def initialize_dictionary(self):
        hyperparameters = {}

        input_layer = self.neurons[0]
        output_layer = self.neurons[len(self.neurons) - 1]
        layer_sizes = self.neurons[1:len(self.neurons) - 1] # Extract hidden layers

        hidden = { hidden: [] for hidden in range(self.layers) } 
        for x in range(len(layer_sizes)):
            hidden[x] = layer_sizes[x]
        
        hyperparameters['w1'] = np.random.randn(hidden[0], input_layer) * np.sqrt(1. / hidden[0])
        for x in range(self.layers-1):
            hyperparameters['w' + str(x + 2)] = np.random.randn(hidden[x + 1], hidden[x]) * np.sqrt(1. / hidden[x + 1])
        hyperparameters['w' + str(self.layers + 1)] = np.random.randn(output_layer, hidden[self.layers - 1]) * np.sqrt(1. / output_layer)

        return hyperparameters

    def compute_sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)
        return 1 / (1 + np.exp(-x))

    def compute_softmax(self, x, derivative=False):
        e = np.exp(x - x.max())
        if derivative:
            return e / np.sum(e, axis=0) * (1 - e / np.sum(e, axis=0))
        return e / np.sum(e, axis=0)

    def forward(self, x_train):
        hyperparameters = self.hyperparameters

        # Activation for input layer
        hyperparameters['a0'] = x_train

        # Traverse through the MLNN
        for i in range(self.layers + 1):
            hyperparameters['z' + str(i + 1)] = np.dot(hyperparameters['w' + str(i + 1)], hyperparameters['a' + str(i)])
            hyperparameters['a' + str(i + 1)] = self.compute_sigmoid(hyperparameters['z' + str(i + 1)])
        return hyperparameters['a' + str(self.layers + 1)]

    def backward(self, y_train, output):
        hyperparameters = self.hyperparameters

        error = 2 * (output - y_train) / output.shape[0] * self.compute_softmax(hyperparameters['z' + str(self.layers + 1)], derivative=True)

        w = {}
        w['w' + str(self.layers + 1)] = np.outer(error, hyperparameters['a' + str(self.layers)])
        
        for x in range(self.layers, 0, -1):
            error = np.dot(hyperparameters['w' + str(x + 1)].T, error) * self.compute_sigmoid(hyperparameters['z' + str(x)], derivative=True)
            w['w' + str(x)] = np.outer(error, hyperparameters['a' + str(x - 1)])
        return w

    def update_hyperparameters(self, w):
        for key, value in w.items():
            if self.epoch == 0:
                self.hyperparameters[key] -= (self.alpha * value)
            else:
                # Regularization with momentum
                self.hyperparameters[key] -= (self.momentum * self.alpha * value)

    # Winner-take-all approach
    def compute_accuracy(self, xp, yp):
        predictions = []
        for x, y in zip(xp, yp):
            output = self.forward(x)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))
        return np.mean(predictions)

    def compute_error(self, xp, yp):
        return 1 - self.compute_accuracy(xp, yp)

    def plot_metrics(self, accuracies, errors, xp, yp):
        '''
        Confusion Matrix
        '''
        n_predictions = []
        ground_truths = []

        for x, y in zip(xp, yp):
            output = self.forward(x)
            ground_truths.append(np.argmax(y))
            n_predictions.append(np.argmax(output))

        train_confusion_matrix = confusion_matrix(ground_truths, n_predictions)
        df = pd.DataFrame(train_confusion_matrix, index = [i for i in ['y = 0','y = 1','y = 2','y = 3','y = 4','y = 5','y = 6','y = 7','y = 8','y = 9']], 
            columns = [c for c in ['ŷ = 0','ŷ = 1','ŷ = 2','ŷ = 3','ŷ = 4','ŷ = 5','ŷ = 6','ŷ = 7','ŷ = 8','ŷ = 9']])
        sn.heatmap(df, annot=True, fmt='g')
        plt.axes().set_title('Test Set Confusion Matrix')
        plt.show()

        span = np.arange(0, self.epochs, 10)

        '''
        Accuracy Plot
        '''
        plt.plot(span, accuracies)
        plt.title('Balanced Accuracy Over 500 Epochs')
        plt.ylabel('Balanced Accuracy')
        plt.xlabel('Epochs')
        plt.show()

        '''
        Error Plot
        '''
        plt.plot(span, errors)
        plt.title('Training Error Over 500 Epochs')
        plt.ylabel('Error')
        plt.xlabel('Epochs')
        plt.show()

    def train_model(self, x_train, y_train, x_test, y_test):
        t_accuracies = []
        train_errors = []

        for epoch in range(self.epochs):
            for x, y in zip(x_train, y_train):
                output = self.forward(x)
                w = self.backward(y, output)
                self.update_hyperparameters(w)

            ''' 
            Reduce learning rate as we move forward to reduce oscillation. min_learning_rate is used 
            to ensure that the learning rate does not shrink too much to guarantee that the neural 
            net keeps learning at high number of epochs, just at a slower rate.
            '''
            if self.alpha > self.min_alpha:
                self.alpha -= 0.0001 * self.alpha
            
            accuracy = self.compute_accuracy(x_train, y_train)
            tr_error = self.compute_error(x_train, y_train)

            # Save accuracy value at the beginning, and then at every tenth epoch
            if (self.epoch % 10 == 0):
                t_accuracies.append(accuracy)
                train_errors.append(tr_error)
            self.epoch += 1

            print('Epoch: {0} ... Accuracy:'.format(epoch + 1), round(accuracy * 100, 2))

        self.plot_metrics(t_accuracies, train_errors, x_test, y_test)

def run():
    '''
    User can specify number of hidden layers at run-time. Input and output neurons are predetermined
    since each image (input) is 24 x 24 pixels and the output layer contains a neuron for each digit.

    Sample input:
        Hidden layers: 2
        Number of neurons (seperate using space): 256 128
    '''
    hidden_layers = int(input('Hidden layers: ')) 
    neurons = list(map(int, input('Number of neurons (seperate using space): ').strip().split()))[:hidden_layers]

    neurons_list = [784]     # Input layer dimensionality
    for layer in range(0, hidden_layers):
        neurons_list.append(neurons[layer])
    neurons_list.append(10)  # Output layer dimensionality

    mlnn = MLNN(layers=hidden_layers, neurons=neurons_list)
    mlnn.train_model(mnist.x_train, mnist.y_train, mnist.x_test, mnist.y_test)

# run()