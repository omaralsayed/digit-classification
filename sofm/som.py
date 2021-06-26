'''
Self-Organizing Feature Map representing the provided MNIST dataset. 
The weights are stored in a map which is written to som.txt.
'''
import dataset

import numpy as np
import matplotlib.pyplot as plt

mnist = dataset.get_sets()

def compute_euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2) 

def compute_closest_node(data, c, n_map, rows, cols):
    vert = (0, 0)
    dist = 100 # Initial dist
    for i in range(rows):
        for j in range(cols):
            distance = compute_euclidean_distance(n_map[i][j], data[c])
            if distance < dist:
                dist = distance
                vert = (i, j)
    return vert

def main():
    l_rate = 0.05
    epochs = 10000
    neurons = 784
    rows = 12
    cols = 12

    x_train= mnist.x_train
    x_test = mnist.x_test
    y_test = mnist.y_test_

    np.random.seed(50)
    n_map = np.random.random_sample(size=(rows, cols, neurons))
    for epoch in range(epochs):
        print('Epoch ', str(epoch))
        remaining = 1.0 - ((epoch * 1.0) / epochs)
        updated_range = (int)(remaining * (rows + cols))
        updated_lrate = remaining * l_rate
        c = np.random.randint(len(x_train))
        bmu_i, bmu_j = compute_closest_node(x_train, c, n_map, rows, cols)
        for i in range(rows):
            for j in range(cols):
                if np.abs(bmu_i - i) + np.abs(bmu_j - j) < updated_range:
                    n_map[i][j] = n_map[i][j] + updated_lrate * (x_train[c] - n_map[i][j])

    # Write weights to text file
    file = open('som.txt', 'w')
    for x in range(rows):
        for y in range(cols):
            np.savetxt(file, n_map[x][y])
    file.close()

    # Zip 144 neurons and plot on grayscale
    neurons = n_map; neuron = []
    for x in range(rows):
        for y in range(cols):
            neuron.append(neurons[x][y])
    _, axes = plt.subplots(nrows=12, ncols=12, sharex=True, sharey=True, figsize=(20, 4))

    n_144 = ([neuron[:12], neuron[12:24], neuron[24:36], neuron[36:48], neuron[48:60], neuron[60:72], 
        neuron[72:84], neuron[84:96], neuron[96:108], neuron[108:120], neuron[120:132], neuron[132:144]])
    for img, row in zip(n_144, axes):
        for img, ax in zip(img, row):
            ax.imshow(img.reshape((28, 28)), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()

    n_00 = {}; n_01 = {}; n_02 = {}; n_03 = {}; n_04 = {}
    n_05 = {}; n_06 = {}; n_07 = {}; n_08 = {}; n_09 = {}
    for x in range(len(x_test)):
        bmu_i, bmu_j = compute_closest_node(x_test, x, n_map, rows, cols)
        if y_test[x] == '0':
            if str(bmu_i) + ' ' + str(bmu_j) not in n_00.keys():  
                n_00[str(bmu_i) + ' ' + str(bmu_j)] = 1
            else: 
                n_00[str(bmu_i) + ' ' + str(bmu_j)] += 1
        elif y_test[x] == '1':
            if str(bmu_i) + ' ' + str(bmu_j) not in n_01.keys():  
                n_01[str(bmu_i) + ' ' + str(bmu_j)] = 1
            else:
                n_01[str(bmu_i) + ' ' + str(bmu_j)] += 1
        elif y_test[x] == '2':
            if str(bmu_i) + ' ' + str(bmu_j) not in n_02.keys():  
                n_02[str(bmu_i) + ' ' + str(bmu_j)] = 1
            else:
                n_02[str(bmu_i) + ' ' + str(bmu_j)] += 1
        elif y_test[x] == '3':
            if str(bmu_i) + ' ' + str(bmu_j) not in n_03.keys():  
                n_03[str(bmu_i) + ' ' + str(bmu_j)] = 1
            else:
                n_03[str(bmu_i) + ' ' + str(bmu_j)] += 1
        elif y_test[x] == '4':
            if str(bmu_i) + ' ' + str(bmu_j) not in n_04.keys():  
                n_04[str(bmu_i) + ' ' + str(bmu_j)] = 1
            else:
                n_04[str(bmu_i) + ' ' + str(bmu_j)] += 1
        elif y_test[x] == '5':
            if str(bmu_i) + ' ' + str(bmu_j) not in n_05.keys():  
                n_05[str(bmu_i) + ' ' + str(bmu_j)] = 1
            else:
                n_05[str(bmu_i) + ' ' + str(bmu_j)] += 1
        elif y_test[x] == '6':
            if str(bmu_i) + ' ' + str(bmu_j) not in n_06.keys():  
                n_06[str(bmu_i) + ' ' + str(bmu_j)] = 1
            else:
                n_06[str(bmu_i) + ' ' + str(bmu_j)] += 1
        elif y_test[x] == '7':
            if str(bmu_i) + ' ' + str(bmu_j) not in n_07.keys():  
                n_07[str(bmu_i) + ' ' + str(bmu_j)] = 1
            else:
                n_07[str(bmu_i) + ' ' + str(bmu_j)] += 1
        elif y_test[x] == '8':
            if str(bmu_i) + ' ' + str(bmu_j) not in n_08.keys():  
                n_08[str(bmu_i) + ' ' + str(bmu_j)] = 1
            else:
                n_08[str(bmu_i) + ' ' + str(bmu_j)] += 1
        elif y_test[x] == '9':
            if str(bmu_i) + ' ' + str(bmu_j) not in n_09.keys():  
                n_09[str(bmu_i) + ' ' + str(bmu_j)] = 1
            else:
                n_09[str(bmu_i) + ' ' + str(bmu_j)] += 1

    classes = []
    n_total = [n_00, n_01, n_02, n_03, n_04, n_05, n_06, n_07, n_08, n_09]
    for i in range(len(set(y_test))):
        classes.append(compute_mapping(n_total[i], rows, cols))

    # Show plots
    show_plots(classes)
    plt.show()

def compute_mapping(class_, rows, cols):
    ret_map = np.zeros(shape=(rows, cols), dtype=int)
    for x in range(rows):
        for y in range(cols):
            for neuron, value in class_.items():
                if neuron == (str(x) + ' ' + str(y)):
                    ret_map[x][y] = value
    return ret_map

# 10 activity matrices
def show_plots(classes):
    count = 0
    _, axes = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, figsize=(20, 4))
    for img, row in zip([classes[:5], classes[5:]], axes):
        for img, ax in zip(img, row):
            ax.imshow(img.reshape((12, 12)), cmap='gray')
            ax.set_title('Class ' + str(count))
            count += 1

if __name__ == '__main__':
    main()