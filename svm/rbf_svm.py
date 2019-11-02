import numpy as np

class LSSVM():
	def __init__(self, C, gamma):
		self.C = C
		self.gamma = gamma
		self.model = {'alpha': 0, 'b': 0, 'sv': None}

	def kernel(self, X, sv):
		n = X.shape[0]
		m = sv.shape[0]
		K = np.zeros((m, n))
		for i in range(m):
			for j in range(n):
				xi, xj = sv[i], X[j]
				K[i, j] = np.exp(-self.gamma * np.dot((xi - xj).T, (xi - xj)))
		return K

	def train(self, X, y):
		n = X.shape[0]
		# solve the matrix
		R = np.concatenate((np.array([0]), y), axis = 0)
		L_B = self.kernel(X, X) + (1/self.C)* np.eye(n)
		R_B = np.array([np.ones((n))])
		B = np.concatenate((R_B.T, L_B), axis = 1)
		U = np.array([np.ones((n+1))])
		U[0][0] = 0
		M = np.concatenate((U, B),axis = 0)
		b_a = np.linalg.solve(M, R)
		self.model['alpha'] = b_a[1:]
		self.model['b'] = b_a[0]
		self.model['sv'] = X
		return self.model

	def predict(self, X):
		n = X.shape[0]
		k = self.kernel(X, self.model['sv'])
		pred = self.model['b'] + np.dot(k.T, self.model['alpha'])
		pred[pred >= 0] = 1
		pred[pred < 0] = -1
		return pred

	def error(self,pred, y):
		return np.sum(y != pred)/y.shape[0]


# data_train = np.genfromtxt("./svm_dataset/train.csv", delimiter=',')[1:]
# data_test = np.genfromtxt("./svm_dataset/test.csv", delimiter=',')[1:]
# X_train, y_train = data_train[:,:2], data_train[:,2]
# X_test, y_test = data_test[:,:2], data_test[:,2]
# svm1 = LSSVM(C = 1, gamma = 1)
# model1 = svm1.train(X_train, y_train)
# y_pred = svm1.predict(X_test)
# y_train_pred = svm1.predict(X_train)
# print(y_train_pred)
# print(np.sum(y_train_pred != y_train))
# print(svm1.error(y_pred, y_test))