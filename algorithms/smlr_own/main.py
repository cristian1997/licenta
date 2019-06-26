import numpy as np
import sys

from math import exp

def read_file(filename, num_of_classes):
	with open(filename) as f:
		data_X, data_Y = [], []
		for line in f:
			v = list(map(float, line.split()))
			y = [0] * (num_of_classes - 1)
			
			v[-1] -= 1 # for satimage
			
			# assumes classes are numbered 0, 1, ..., num_of_classes - 1
			if v[-1] < num_of_classes - 1:
				y[int(v[-1])] = 1
			
			data_X.append(v[:-1])
			data_Y.append(y)
		
		return np.asmatrix(data_X), np.asmatrix(data_Y)

def get_prob_vector(instance, w, num_of_classes, num_of_features):
	p = []
	for label in range(num_of_classes - 1):
		wl = w[label * num_of_features : (label + 1) * num_of_features]
		p.append(exp(instance @ wl))
	
	p.append(1)
	
	return np.array(p) / sum(p)

def predict(instance, w, num_of_classes):
	p = get_prob_vector(instance, w, num_of_classes, instance.shape[1])
	
	return np.argmax(p)

def get_accuracy(test_x, test_y, w, num_of_classes):
	correct = 0
	for i in range(test_x.shape[0]):
		if predict(test_x[i], w, num_of_classes) == np.argmax(test_y[i]):
			correct += 1

	return correct / test_x.shape[0] * 100

def train(train_x, train_y, num_of_classes, nr_iterations = 10, test_x = None, test_y = None, test_error_file = None):
	num_of_features = train_x[0].shape[1]
	
	v1 = np.array([1] * (num_of_classes - 1)).T
	identity = np.identity(num_of_features * (num_of_classes - 1))
	
	lhs = -(identity - (v1 @ v1.T) / num_of_classes) / 2
	rhs = sum([vx * vx.T for vx in train_x])
	
	b = np.kron(lhs, rhs)
	invb = np.linalg.inv(b)
	
	w = np.array([1 / num_of_classes] * (num_of_features * (num_of_classes - 1)))
	w = w.reshape((w.shape[0], 1))
	
	print("Training started")
	for i in range(nr_iterations):
		g = np.zeros(((num_of_features * (num_of_classes - 1)), 1))
		for j in range(train_x.shape[0]):
			# pj = []
			# for label in range(num_of_classes - 1):
			# 	wl = w[label * num_of_features : (label + 1) * num_of_features]
			# 	# print(train_x[j])
			# 	# print(wl)
			# 	# print(train_x[j] @ wl)
			# 	pj.append(exp(train_x[j] @ wl))
			
			# pj = np.array(pj) / sum(pj)
			pj = get_prob_vector(train_x[j], w, num_of_classes, num_of_features)[:-1]
			
			aux = train_y[j] - pj
			g += np.kron(aux.T, train_x[j].T)
		
		# print(invb.shape)
		# print(g.shape)
		# print((invb @ g).shape)
		# print(w.shape)
		w = w - (invb @ g)
		# print(w.shape)
		# print(w)
		
		if test_error_file is not None:
			test_error_file.write(str(get_accuracy(test_x, test_y, w, num_of_classes)) + "\n")
	
	print("Train ended")
	
	return w

dataset_name = sys.argv[1]
nr_iterations = int(sys.argv[2])
num_of_classes = 7

test_error_file = open("out/" + dataset_name + "/" + str(nr_iterations) + "-testerrorplot.txt", "w")

train_x, train_y = read_file("../datasets/" + dataset_name + "/train.txt", num_of_classes = num_of_classes)
test_x, test_y = read_file("../datasets/" + dataset_name + "/test.txt", num_of_classes = num_of_classes)

# print(test_x.shape)
w = train(train_x, train_y, num_of_classes = num_of_classes, test_x = test_x, test_y = test_y,
	nr_iterations = nr_iterations, test_error_file = test_error_file)
