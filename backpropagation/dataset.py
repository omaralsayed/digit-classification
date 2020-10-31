'''
Parse dataset to useful format.

Each row in MNISTnumImages5000_balanced.txt has 784 values
representing the intensities of the image for one digit between 0 and 9.

Each row in MNISTnumLabels5000_balanced.txt has a label for
the respective image (a value between 0 and 9).

Author: Omar Alsayed (alsayeoy@mail.uc.edu).
Date: 10-25-2020
'''
import random
import numpy as np

class MNIST():
    x_train = [None] * 4000
    y_train = [None] * 4000
    x_test  = [None] * 1000
    y_test  = [None] * 1000

def get_sets():
    sets = MNIST()

    mnist_images = open('MNISTnumImages5000_balanced.txt')
    mnist_labels = open('MNISTnumLabels5000_balanced.txt')
    
    image_file = mnist_images.readlines()
    label_file = mnist_labels.readlines()

    images = list()
    for line in image_file:
        images.append(line.split())

    labels = list()
    for line in label_file:
        labels.append(line.strip())

    sets.x_train = (images[:399] + images[499:899]   + images[999:1399] 
        + images[1499:1899]      + images[1999:2399] + images[2499:2899]
        + images[2999:3399]      + images[3499:3899] + images[3999:4399]
        + images[4499:4900])

    sets.y_train = (labels[:399] + labels[499:899]   + labels[999:1399]  
        + labels[1499:1899]      + labels[1999:2399] + labels[2499:2899] 
        + labels[2999:3399]      + labels[3499:3899] + labels[3999:4399]
        + labels[4499:4900])

    data = list(zip(sets.x_train, sets.y_train))
    random.seed(50)
    random.shuffle(data) # Shuffle training set
    sets.x_train, sets.y_train = zip(*data)

    sets.x_train = np.array(sets.x_train).astype('float32')
    sets.y_train = np.array(sets.y_train).astype(int)

    sets.x_test = (images[399:499] + images[899:999]   + images[1399:1499] 
        + images[1899:1999]        + images[2399:2499] + images[2899:2999] 
        + images[3399:3499]        + images[3899:3999] + images[4399:4499] 
        + images[4899:4999])

    sets.y_test = (labels[399:499] + labels[899:999]   + labels[1399:1499] 
        + labels[1899:1999]        + labels[2399:2499] + labels[2899:2999] 
        + labels[3399:3499]        + labels[3899:3999] + labels[4399:4499] 
        + labels[4899:4999])

    data = list(zip(sets.x_test, sets.y_test))
    random.seed(50)
    random.shuffle(data) # Shuffle testing set
    sets.x_test, sets.y_test = zip(*data)

    sets.x_test = np.array(sets.x_test).astype('float32')
    sets.y_test = np.array(sets.y_test).astype(int)

    sets.y_train = encode(10, sets.y_train)
    sets.y_test  = encode(10, sets.y_test)

    mnist_images.close()
    mnist_labels.close()

    return sets

# One-hot encoding
def encode(digits, labels):
    return np.eye(digits)[labels]