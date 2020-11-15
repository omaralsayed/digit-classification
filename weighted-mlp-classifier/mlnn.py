'''
Feed-forward multi-layer neural network with backpropagation with momentum. Able 
to handle any number of hidden layers and hidden neurons (specify at run-time).
'''
import dataset

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd

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
        output_layer = self.neurons[len(self.neurons) - 1]; layer_sizes = self.neurons[1:len(self.neurons) - 1] # Extract hidden layers
        hidden = { hidden: [] for hidden in range(self.layers) } 
        for x in range(len(layer_sizes)):
            hidden[x] = layer_sizes[x]
        # Load weights from Autoencoder
        hyperparameters['W1'] = np.loadtxt('weights-regular-autoencoder.txt').reshape(100, 784)
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
        error = 2 * (output - y_train) / output.shape[0] * self.compute_softmax(hyperparameters['Z' + str(self.layers + 1)], derivative=True)
        w['W' + str(self.layers + 1)] = np.outer(error, hyperparameters['A' + str(self.layers)])
        for x in range(self.layers, 1, -1):
            error = np.dot(hyperparameters['W' + str(x + 1)].T, error) * self.compute_sigmoid(hyperparameters['Z' + str(x)], derivative=True)
            w['W' + str(x)] = np.outer(error, hyperparameters['A' + str(x - 1)])
        return w

    def update_w(self, w):
        for wi, wv in w.items():
            if self.epoch == 0:
                self.hyperparameters[wi] -= (self.alpha * wv)
            else:
                self.hyperparameters[wi] -= (self.momentum * self.alpha * wv)

    def compute_sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)
        return 1 / (1 + np.exp(-x))

    def compute_softmax(self, x, derivative=False):
        e = np.exp(x - x.max())
        if derivative:
            return e / np.sum(e, axis=0) * (1 - e / np.sum(e, axis=0))
        return e / np.sum(e, axis=0)

    def compute_accuracy(self, xp, yp):
        predictions = []
        for x, y in zip(xp, yp):
            output = self.forward_pass(x)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))
        return np.mean(predictions)

    def compute_error(self, xp, yp):
        return 1 - self.compute_accuracy(xp, yp)

    def compute_confusion_matrix(self, actual, predicted):
        classes = np.unique(actual)
        matrix = np.zeros((len(classes), len(classes)))
        for i in range(len(classes)):
            for j in range(len(classes)):
                matrix[i, j] = np.sum((actual == classes[i]) & (predicted == classes[j]))      
        return matrix

    def train_model(self, x_train, y_train, x_test, y_test):
        accuracies = []; errors = []

        for epoch in range(self.epochs):
            for x, y in zip(x_train, y_train):
                output = self.forward_pass(x)
                w = self.backward_pass(y, output)
                self.update_w(w)

            if self.alpha > self.min_alpha:
                self.alpha -= 0.0001 * self.alpha
            
            accuracy = self.compute_accuracy(x_train, y_train)
            tr_error = self.compute_error(x_train, y_train)

            # Save accuracy value at the beginning, and then at every tenth epoch
            if (self.epoch % 10 == 0):
                accuracies.append(accuracy)
                errors.append(tr_error)
            self.epoch += 1

            print('Epoch: {0} ... Accuracy:'.format(epoch + 1), round(accuracy * 100, 2))
        self.show_plots(accuracies, errors, x_train, y_train, x_test, y_test)

    def show_plots(self, accuracies, errors, x_train, y_train, x_test, y_test):
        '''
        Train Confusion Matrix
        '''
        actual = []; predicted = []
        for x, y in zip(x_train, y_train):
            output = self.forward_pass(x)
            actual.append(np.argmax(y))
            predicted.append(np.argmax(output))
        train_confusion_matrix = self.compute_confusion_matrix(actual, predicted)
        df = pd.DataFrame(train_confusion_matrix, index = [i for i in ['y = 0','y = 1','y = 2','y = 3','y = 4','y = 5','y = 6','y = 7','y = 8','y = 9']], 
            columns = [c for c in ['ŷ = 0','ŷ = 1','ŷ = 2','ŷ = 3','ŷ = 4','ŷ = 5','ŷ = 6','ŷ = 7','ŷ = 8','ŷ = 9']])
        sn.heatmap(df, annot=True, fmt='g', cmap="YlGnBu")
        plt.title('Train Set Confusion Matrix')
        plt.show()

        '''
        Test Confusion Matrix
        '''
        actual = []; predicted = []
        for x, y in zip(x_test, y_test):
            output = self.forward_pass(x)
            actual.append(np.argmax(y))
            predicted.append(np.argmax(output))
        test_confusion_matrix = self.compute_confusion_matrix(actual, predicted)
        df = pd.DataFrame(test_confusion_matrix, index = [i for i in ['y = 0','y = 1','y = 2','y = 3','y = 4','y = 5','y = 6','y = 7','y = 8','y = 9']], 
            columns = [c for c in ['ŷ = 0','ŷ = 1','ŷ = 2','ŷ = 3','ŷ = 4','ŷ = 5','ŷ = 6','ŷ = 7','ŷ = 8','ŷ = 9']])
        sn.heatmap(df, annot=True, fmt='g', cmap="YlGnBu")
        plt.title('Test Set Confusion Matrix')

        span = np.arange(0, self.epochs, 10)

        '''
        Accuracy Plot
        '''
        plt.figure(3)
        plt.plot(span, accuracies)
        plt.title('Balanced Accuracy Over 500 Epochs')
        plt.ylabel('Balanced Accuracy')
        plt.xlabel('Epochs')

        '''
        Error Plot
        '''
        plt.figure(4)
        plt.plot(span, errors)
        plt.title('Training Error Over 500 Epochs')
        plt.ylabel('Error')
        plt.xlabel('Epochs')

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
    mlnn.show_random_hidden_neurons()
    plt.show()

if __name__ == '__main__':
    main()