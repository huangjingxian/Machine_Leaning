from rbf_svm import LSSVM
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

data_train = np.genfromtxt("./svm_dataset/train.csv", delimiter=',')[1:]
data_test = np.genfromtxt("./svm_dataset/test.csv", delimiter=',')[1:]
X_train, y_train = data_train[:,:2], data_train[:,2]
X_test, y_test = data_test[:,:2], data_test[:,2]

# build cg pairs
cgpairs = []
C = 0.1
gamma = 0.001
for i in range(5):
	C *= 10
	for i in range(5):
		gamma *= 10
		cgpairs.append((C, gamma))
	gamma = 0.001

# train and compare
test_data = [[],[],[],[],[]]
train_data = [[],[],[],[],[]]
for cg in cgpairs:
	C = cg[0]
	gamma = cg[1]

	# lssvm
	lssvm = LSSVM(C, gamma)
	model = lssvm.train(X_train, y_train)
	y_pred_train = lssvm.predict(X_train)
	y_pred_test = lssvm.predict(X_test)
	loss_train = lssvm.error(y_pred_train, y_train)
	loss_test = lssvm.error(y_pred_test, y_test)
	test_data[0].append(loss_test)
	train_data[0].append(loss_train)

	# different kernels with svm
	kernels = ['linear', 'poly', 'rbf', 'sigmoid']
	for i in range(len(kernels)):
		kernel = kernels[i]
		svm = SVC(C = C, kernel = kernel, gamma = gamma)
		print(kernel, "----", cg, "----")
		svm.fit(X_train, y_train)
		y_pred_train = svm.predict(X_train)
		y_pred_test = svm.predict(X_test)
		loss_train = np.sum(y_pred_train != y_train)/y_train.shape[0]
		loss_test = np.sum(y_test != y_pred_test)/y_test.shape[0]
		test_data[i + 1].append(loss_test)
		train_data[i + 1].append(loss_train)

test_1, test_2, test_3, test_4, test_5 = test_data[0], test_data[1], test_data[2], test_data[3], test_data[4]
labels = 'LSSVM-rbf','SVM-linear','SVM-poly','SVM-rbf','SVM-sigmoid'
plt.boxplot([test_1, test_2, test_3, test_4, test_5],labels = labels)
plt.ylabel("error")
plt.title("test")
plt.savefig('test.png')
plt.close("all")
train_1, train_2, train_3, train_4, train_5 = train_data[0], train_data[1], train_data[2], train_data[3], train_data[4]
plt.boxplot([train_1, train_2, train_3, train_4, train_5], labels = labels)
plt.ylabel("error")
plt.title("train")
plt.savefig('train.png')
