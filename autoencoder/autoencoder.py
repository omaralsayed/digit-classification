'''
Autoencoder network using the dataset and hyperparameters used in the MLP neural network. The task of this autoencoder 
is to explore interesting features in the hidden layers and to reconstruct the input, which is a 28 x 28 pixels image.
'''
import dataset

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd

mnist = dataset.get_sets()

class Autoencoder:
    def __init__(self, layers, neurons, epoch=0, epochs=500, learning_rate=0.3, min_learning_rate=0.0025, momentum=0.9):
        self.neurons = neurons # List of nerons for Autoencoder
        self.epoch = epoch
        self.epochs = epochs
        self.alpha = learning_rate
        self.min_alpha = min_learning_rate
        self.momentum = momentum
        self.layers = layers
        self.output = []
        self.hyperparameters = self.initialize_dictionary()

    def initialize_dictionary(self):
        hyperparameters = {}
        input_layer = self.neurons[0]; output_layer = self.neurons[len(self.neurons) - 1]
        layer_sizes = self.neurons[1:len(self.neurons) - 1] # Extract hidden layers
        hidden = { hidden: [] for hidden in range(self.layers) } 
        for x in range(len(layer_sizes)):
            hidden[x] = layer_sizes[x]
        hyperparameters['W1'] = np.random.randn(hidden[0], input_layer) * np.sqrt(1. / hidden[0])
        for x in range(self.layers-1):
            hyperparameters['W' + str(x + 2)] = np.random.randn(hidden[x + 1], hidden[x]) * np.sqrt(1. / hidden[x + 1])
        hyperparameters['W' + str(self.layers + 1)] = np.random.randn(output_layer, hidden[self.layers - 1]) * np.sqrt(1. / output_layer)
        return hyperparameters

    def forward_pass(self, x_train):
        hyperparameters = self.hyperparameters
        hyperparameters['A0'] = x_train
        for i in range(self.layers + 1):
            hyperparameters['Z' + str(i + 1)] = np.dot(hyperparameters['W' + str(i + 1)], hyperparameters['A' + str(i)])
            hyperparameters['A' + str(i + 1)] = self.compute_sigmoid(hyperparameters['Z' + str(i + 1)])
        return hyperparameters['A' + str(self.layers + 1)]

    def backward_pass(self, y_train, output):
        hyperparameters = self.hyperparameters; w = {}
        error = 2 * (output - y_train) / output.shape[0] * self.compute_sigmoid(hyperparameters['Z' + str(self.layers + 1)], derivative=True)
        w['W' + str(self.layers + 1)] = np.outer(error, hyperparameters['A' + str(self.layers)])
        for x in range(self.layers, 0, -1):
            error = np.dot(hyperparameters['W' + str(x + 1)].T, error) * self.compute_sigmoid(hyperparameters['Z' + str(x)], derivative=True)
            w['W' + str(x)] = np.outer(error, hyperparameters['A' + str(x - 1)])
        return w

    def update_w(self, w):
        for wi, wv in w.items():
            if self.epoch == 0:
                self.hyperparameters[wi] -= (self.alpha * wv)
            else:
                self.hyperparameters[wi] -= (self.momentum * self.alpha * wv)

    def save_w(self):
        w = open('weights.txt', 'w')
        for wi in self.hyperparameters['W1']:
            np.savetxt(w, wi)
        w.close()

    def compute_sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)
        return 1 / (1 + np.exp(-x))

    def compute_accuracy(self, xp, output, labels):
        losses = []; cumulative_losses = []
        itr_loss = 0; total_loss = 0

        l0 = []; l1 = []; l2 = []; l3 = []; l4 = []
        l5 = []; l6 = []; l7 = []; l8 = []; l9 = []
        
        ln = []

        epoch = 0
        for xp, _ in zip(xp, output):
            output = self.forward_pass(xp)
            if (self.epoch == self.epochs - 1):
                self.output.append(output)
            for i in range(self.neurons[len(self.neurons) - 1]):
                itr_loss += (xp[i] - output[i]) ** 2
            losses.append(0.5 * itr_loss)
            if self.epoch == self.epochs:
                if labels[epoch] == '0':
                    l0.append(0.5 * itr_loss) 
                elif labels[epoch] == '1':
                    l1.append(0.5 * itr_loss)
                elif labels[epoch] == '2':
                    l2.append(0.5 * itr_loss)
                elif labels[epoch] == '3':
                    l3.append(0.5 * itr_loss)
                elif labels[epoch] == '4':
                    l4.append(0.5 * itr_loss)
                elif labels[epoch] == '5':
                    l5.append(0.5 * itr_loss)
                elif labels[epoch] == '6':
                    l6.append(0.5 * itr_loss)
                elif labels[epoch] == '7':
                    l7.append(0.5 * itr_loss)
                elif labels[epoch] == '8':
                    l8.append(0.5 * itr_loss)
                else:
                    l9.append(0.5 * itr_loss)
                ln.append(0.5 * itr_loss)
            epoch += 1

        if self.epoch == self.epochs:
            digits = [l0, l1, l2, l3, l4, l5, l6, l7, l8, l9]
            for digit in digits:
                cumulative_losses.append(100 - np.mean(digit) / len(xp))
            total_loss = 100 - np.mean(ln) / len(xp)
        return [np.mean(losses) / len(xp), cumulative_losses, total_loss]

    def train_model(self, x_train, y_train, x_test, y_test, y_train_, y_test_):
        losses = []
        for epoch in range(self.epochs):
            for x, y in zip(x_train, y_train):
                output = self.forward_pass(x)
                w = self.backward_pass(y, output)
                self.update_w(w)

            if self.alpha > self.min_alpha:
                self.alpha -= 0.0001 * self.alpha
            
            # Save accuracy value at the beginning, and then at every tenth epoch
            itr_loss = self.compute_accuracy(x_test, x_test, y_test)[0]
            if (self.epoch % 10 == 0):
                losses.append(itr_loss)
            self.epoch += 1

            print('Epoch: {0} ... Loss:'.format(epoch + 1), round(itr_loss, 2))

        itr_loss, losses_test, total_loss_test = (self.compute_accuracy(x_test, x_test, y_test_)[0], 
            self.compute_accuracy(x_test, x_test, y_test_)[1], self.compute_accuracy(x_test, x_test, y_test_)[2])
        itr_loss, losses_train, total_loss_train = (self.compute_accuracy(x_train, x_train, y_train_)[0], 
            self.compute_accuracy(x_train, x_train, y_train_)[1], self.compute_accuracy(x_train, x_train, y_train_)[2])
        self.show_plots(losses, losses_train, losses_test, total_loss_train, total_loss_test)

    def show_plots(self, accuracy, losses_train, losses_test, total_loss_train, total_loss_test):
        span = np.arange(0, self.epochs, 10)

        '''
        Overall Accuracy Plot
        '''
        plt.figure(0)
        plt.plot(span, accuracy)
        plt.title('Overall Accuracy Over 500 Epochs')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')

        '''
        Accuracy Per Digit Chart
        '''
        plt.figure(1)
        digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        interval = np.arange(10)
        bar_train = [x + 0.15 for x in interval]
        bar_test = [x + 0.15 for x in bar_train]
        plt.bar(bar_train, losses_train, width = 0.15, label='Train')
        plt.bar(bar_test, losses_test, width = 0.15, label='Test')
        plt.xticks([r + 0.15 for r in range(len(losses_train))], digits)
        plt.legend()
        plt.title('Accuracy Per Digit Over 500 Epochs')
        plt.xlabel('Digits')
        plt.ylabel('Accuracy')

        '''
        Overall Accuracy Chart
        '''
        plt.figure(2)
        bar_train = 0.15; bar_test = 0.30
        plt.bar(bar_train, total_loss_train, width = 0.15, label='Train')
        plt.bar(bar_test, total_loss_test, width = 0.15, label='Test')
        plt.legend()
        plt.title('Train and Test Accuracy Over 500 Epochs')
        plt.xlabel('Data Labels')
        plt.ylabel('Accuracy')

    def show_image_reconstruction(self, xp):
        figure, axes = plt.subplots(nrows=2, ncols=8, sharex=True, sharey=True)
        for images, rows in zip([xp, self.output[:8]], axes):
            for image, row in zip(images, rows):
                row.imshow(image.reshape((28, 28)), cmap='gray')
                row.get_xaxis().set_visible(False)
                row.get_yaxis().set_visible(False)
        figure.tight_layout(pad=0.1)

    def show_random_hidden_neurons(self):
        neurons = self.hyperparameters['W1']
        neuron = []
        for i in range(len(neurons)):
            neuron.append(neurons[i])
        figure, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True)
        for images, rows in zip([neuron[:10], neuron[20:30]], axes):
            for image, row in zip(images, rows):
                row.imshow(image.reshape((28, 28)), cmap='gray')
                row.get_xaxis().set_visible(False)
                row.get_yaxis().set_visible(False)
        figure.tight_layout(pad=0.1)

def main():
    '''
    User can specify number of hidden layers at run-time. Input and output neurons are predetermined
    since each image (input) is 24 x 24 pixels and the output layer is the reconstructed input.

    Sample input:
        Hidden layers: 2
        Number of neurons (seperate using space): 256 128
    '''
    hidden_layers = int(input('Hidden layers: ')) 
    neurons = list(map(int, input('Number of neurons (seperate using space): ').strip().split()))[:hidden_layers]

    neurons_list = [784]      # Input layer dimensionality
    for layer in range(0, hidden_layers):
        neurons_list.append(neurons[layer])
    neurons_list.append(784)  # Output layer dimensionality

    auto = Autoencoder(layers=hidden_layers, neurons=neurons_list)
    auto.train_model(mnist.x_train, mnist.x_train, mnist.x_test, mnist.x_test, mnist.y_train_, mnist.y_test_)

    # Store final weights
    auto.save_w()

    # Generate plots
    auto.show_image_reconstruction(mnist.x_test[:8])
    auto.show_random_hidden_neurons()
    plt.show()

if __name__ == '__main__':
    main()