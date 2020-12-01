'''
Self-Organizing Feature Map representing the provided MNIST dataset. 
'''
import numpy as np
import matplotlib.pyplot as plt
import dataset

mnist = dataset.get_sets()

def closest_node(data, t, map, m_rows, m_cols):
  result = (0,0)
  small_dist = 1.0e20
  for i in range(m_rows):
    for j in range(m_cols):
      ed = euc_dist(map[i][j], data[t])
      if ed < small_dist:
        small_dist = ed
        result = (i, j)
  return result

def euc_dist(v1, v2):
  return np.linalg.norm(v1 - v2) 

def manhattan_dist(r1, c1, r2, c2):
  return np.abs(r1-r2) + np.abs(c1-c2)

def most_common(lst, n):
  if len(lst) == 0: 
      return -1
  counts = np.zeros(shape=n, dtype=np.int)
  for i in range(len(lst)):
    counts[lst[i]] += 1
  return np.argmax(counts)

def main():
  np.random.seed(50)

  Dim = 784
  Rows = 12; Cols = 12
  RangeMax = Rows + Cols
  LearnMax = 0.05
  StepsMax = 1000000

  data_x = mnist.x_train
  x_test = mnist.x_test
  y_test = mnist.y_test_

  map = np.random.random_sample(size=(Rows,Cols,Dim))
  for s in range(StepsMax):
    if s % (StepsMax/10) == 0: print("step = ", str(s))
    pct_left = 1.0 - ((s * 1.0) / StepsMax)
    curr_range = (int)(pct_left * RangeMax)
    curr_rate = pct_left * LearnMax
    t = np.random.randint(len(data_x))
    (bmu_row, bmu_col) = closest_node(data_x, t, map, Rows, Cols)
    for i in range(Rows):
      for j in range(Cols):
        if manhattan_dist(bmu_row, bmu_col, i, j) < curr_range:
          map[i][j] = map[i][j] + curr_rate * (data_x[t] - map[i][j])

  file = open("sofm.txt", "w")
  for x in range(Rows):
      for y in range(Cols):
          np.savetxt(file, map[x][y])
  file.close()

  # Plot 144 neurons
  neurons = map
  neuron = []
  for x in range(Rows):
      for y in range(Cols):
          neuron.append(neurons[x][y])
  _, axes = plt.subplots(nrows=12, ncols=12, sharex=True, sharey=True, figsize=(20, 4))
  for images, row in zip([neuron[:12], neuron[12:24], neuron[24:36], neuron[36:48], neuron[48:60], neuron[60:72], neuron[72:84], neuron[84:96], neuron[96:108], neuron[108:120], neuron[120:132], neuron[132:144]], axes):
      for img, ax in zip(images, row):
          ax.imshow(img.reshape((28, 28)), cmap='gray')
          ax.get_xaxis().set_visible(False)
          ax.get_yaxis().set_visible(False)
  plt.show()
  zero = {}
  one = {}
  two = {}
  three = {}
  four = {}
  five = {}
  six = {}
  seven = {}
  eight = {}
  nine = {}
  for x in range(len(x_test)):
    (bmu_row, bmu_col) = closest_node(x_test, x, map, Rows, Cols)
    if y_test[x] == '0':
        if str(bmu_row) + " " + str(bmu_col) not in zero.keys():  
            zero[str(bmu_row) + " " + str(bmu_col)] = 1
        else: 
            zero[str(bmu_row) + " " + str(bmu_col)] += 1
    elif y_test[x] == '1':
        if str(bmu_row) + " " + str(bmu_col) not in one.keys():  
            one[str(bmu_row) + " " + str(bmu_col)] = 1
        else:
            one[str(bmu_row) + " " + str(bmu_col)] += 1
    elif y_test[x] == '2':
        if str(bmu_row) + " " + str(bmu_col) not in two.keys():  
            two[str(bmu_row) + " " + str(bmu_col)] = 1
        else:
            two[str(bmu_row) + " " + str(bmu_col)] += 1
    elif y_test[x] == '3':
        if str(bmu_row) + " " + str(bmu_col) not in three.keys():  
            three[str(bmu_row) + " " + str(bmu_col)] = 1
        else:
            three[str(bmu_row) + " " + str(bmu_col)] += 1
    elif y_test[x] == '4':
        if str(bmu_row) + " " + str(bmu_col) not in four.keys():  
            four[str(bmu_row) + " " + str(bmu_col)] = 1
        else:
            four[str(bmu_row) + " " + str(bmu_col)] += 1
    elif y_test[x] == '5':
        if str(bmu_row) + " " + str(bmu_col) not in five.keys():  
            five[str(bmu_row) + " " + str(bmu_col)] = 1
        else:
            five[str(bmu_row) + " " + str(bmu_col)] += 1
    elif y_test[x] == '6':
        if str(bmu_row) + " " + str(bmu_col) not in six.keys():  
            six[str(bmu_row) + " " + str(bmu_col)] = 1
        else:
            six[str(bmu_row) + " " + str(bmu_col)] += 1
    elif y_test[x] == '7':
        if str(bmu_row) + " " + str(bmu_col) not in seven.keys():  
            seven[str(bmu_row) + " " + str(bmu_col)] = 1
        else:
            seven[str(bmu_row) + " " + str(bmu_col)] += 1
    elif y_test[x] == '8':
        if str(bmu_row) + " " + str(bmu_col) not in eight.keys():  
            eight[str(bmu_row) + " " + str(bmu_col)] = 1
        else:
            eight[str(bmu_row) + " " + str(bmu_col)] += 1
    elif y_test[x] == '9':
        if str(bmu_row) + " " + str(bmu_col) not in nine.keys():  
            nine[str(bmu_row) + " " + str(bmu_col)] = 1
        else:
            nine[str(bmu_row) + " " + str(bmu_col)] += 1
  zero = arrange_classes(zero, Rows, Cols)
  one = arrange_classes(one, Rows, Cols)
  two = arrange_classes(two, Rows, Cols)
  three = arrange_classes(three, Rows, Cols)
  four = arrange_classes(four, Rows, Cols)
  five = arrange_classes(five, Rows, Cols)
  six = arrange_classes(six, Rows, Cols)
  seven = arrange_classes(seven, Rows, Cols)
  eight = arrange_classes(eight, Rows, Cols)
  nine = arrange_classes(nine, Rows, Cols)
  digits = []
  digits.append(zero)
  digits.append(one)
  digits.append(two)
  digits.append(three)
  digits.append(four)
  digits.append(five)
  digits.append(six)
  digits.append(seven)
  digits.append(eight)
  digits.append(nine)
  plot_classes(digits)
  plt.show()
def arrange_classes(digit, Rows, Cols):
    mapping = np.zeros(shape=(Rows, Cols), dtype=int)
    for x in range(Rows):
        for y in range(Cols):
            for neuron, value in digit.items():
                if neuron == (str(x) + " " + str(y)):
                    mapping[x][y] = value
    return mapping
def plot_classes(digits):
    # plot all 10 activity matrices
    count = 0
    _, axes = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, figsize=(20, 4))
    for images, row in zip([digits[:5], digits[5:]], axes):
        for img, ax in zip(images, row):
            ax.imshow(img.reshape((12, 12)), cmap='gray')
            ax.set_title("Class " + str(count))
            count += 1
    print(count)
if __name__=="__main__":
    main()