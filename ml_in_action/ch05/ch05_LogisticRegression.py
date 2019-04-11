# -*- coding: utf-8 -*-
import numpy as np

def load_data_set():
    data_mat, label_mat = list(), list()
    with open("./ml_in_action/ch05/testSet.txt") as f:
        for line in f:
            line_arr = line.strip().split()
            data_mat.append([1.0, float(line_arr[0])], float(line_arr[1]))
            label_mat.append(int(line_arr[2]))
    return data_mat, label_mat

def sigmoid(inX):
    return 1.0 / (1+np.exp(-inX))

def gradient_ascent(data_matrix_in, class_labels):
    data_matrix = np.mat(data_matrix_in)
    label_matrix = np.mat(class_labels)
    m, n = np.shape(data_matrix)
    alpha = 10e-3
    max_cycle = 500
    weights = np.ones((n, 1))
    for i in range(max_cycle):
        h = sigmoid(data_matrix * weights)
        error = (label_matrix - h)
        weights += alpha * data_matrix.transpose() * error
    return error

if __name__ == "__main__":
    print(load_data_set())


