import sys
import time
import numpy as np

from sklearn.ensemble import AdaBoostClassifier

def read_file(filename):
	with open(filename) as f:
		data_X, data_Y = [], []
		for line in f:
			v = list(map(float, line.split()))
			
			v[-1] = (-1 if v[-1] <= 0.5 else 1) # for banknote
			# v[-1] = (-1 if v[-1] <= 1.5 else 1) # for satimage
			# v[-1] = (-1 if v[-1] <= 5.5 else 1) # for mnist
			
			data_X.append(v[:-1])
			data_Y.append(v[-1])
		
		return np.asmatrix(data_X), np.asarray(data_Y)

dataset_name = sys.argv[1]
nr_iterations = int(sys.argv[2])

train_x, train_y = read_file("../datasets/" + dataset_name + "/train.txt")
test_x, test_y = read_file("../datasets/" + dataset_name + "/test.txt")

clf = AdaBoostClassifier(n_estimators = nr_iterations, algorithm = 'SAMME')

train_time_start = time.time()
clf.fit(train_x, train_y)
train_time_stop = time.time()

train_error = 100 * (1 - clf.score(train_x, train_y))
test_error = 100 * (1 - clf.score(test_x, test_y))

# print("Feature importance: ", clf.feature_importances_)
print("Train error:", train_error)
print("Test error:", test_error)
# print(clf.get_params(deep = True))
# print(clf.estimator_weights_)

# print("Train staged score", list(clf.staged_score(train_x, train_y)))
# print("Test staged score", list(clf.staged_score(test_x, test_y)))

train_margins = clf.decision_function(train_x) * train_y
test_margins = clf.decision_function(test_x) * test_y

# print("Staged predict")
# print(list(clf.staged_predict(np.array([[5, 8]]))))


###############################################################################################
# SAVE OUTPUT

folder_name = "out/" + dataset_name
prefix = str(nr_iterations)

with open(folder_name + "/" + prefix + "-summary.txt", 'w') as f:
	f.write("Dataset name: " + dataset_name + '\n')
	f.write("Number of iterations: " + str(nr_iterations) + '\n')
	f.write("Loss function: exponential\n")
	
	f.write("Train set size: " + str(len(train_x)) + '\n')
	f.write("Train error: " + str(train_error) + '\n')
	
	f.write("Test set size: " + str(len(test_x)) + '\n')
	f.write("Test error: " + str(test_error) + '\n')
	
	f.write("Training duration: " + str(train_time_stop - train_time_start) + '\n')

with open(folder_name + "/" + prefix + "-trainerrorplot.txt", 'w') as f:
	f.write('\n'.join(list(map(lambda x : str(100 * (1 - x)), clf.staged_score(train_x, train_y)))))

with open(folder_name + "/" + prefix + "-testerrorplot.txt", 'w') as f:
	f.write('\n'.join(list(map(lambda x : str(100 * (1 - x)), clf.staged_score(test_x, test_y)))))

with open(folder_name + "/" + prefix + "-trainmargins.txt", 'w') as f:
	f.write('\n'.join(list(map(str, train_margins))))

with open(folder_name + "/" + prefix + "-testmargins.txt", 'w') as f:
	f.write('\n'.join(list(map(str, test_margins))))