import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

data_train = np.genfromtxt("./svm_dataset/train.csv", delimiter=',')[1:]
X_train, y_train = data_train[:,:2], data_train[:,2]


gamma = [0.001, 0.1, 1, 10]
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
axs = [ax1, ax2, ax3, ax4]
for i in range(len(gamma)):
	svm = SVC(C = 1, kernel = 'rbf', gamma = gamma[i])
	svm.fit(X_train, y_train)
	pred_y = svm.predict(X_train)
	h = 0.01
	# axs[i] = plt.subplot(223)
	x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
	y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	cm = plt.cm.RdBu
	Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	axs[i].contourf(xx, yy, Z, cmap=cm)
	axs[i].contour(xx, yy, Z, colors=['#A9A9A9','#696969','#A9A9A9'], levels=[-1, 0, 1],linestyles=['--', 'dashdot', '--'])

plt.savefig('gamma.png')
