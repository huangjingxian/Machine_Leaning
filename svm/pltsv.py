import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

data_train = np.genfromtxt("./svm_dataset/train.csv", delimiter=',')[1:]
X_train, y_train = data_train[:,:2], data_train[:,2]

svm = SVC(C = 1, kernel = 'rbf', gamma = 1)
svm.fit(X_train, y_train)
pred_y = svm.predict(X_train)

# plot decision boundary
h = 0.01
ax = plt.subplot(111)
x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
cm = plt.cm.RdBu
Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap=cm)
ax.contour(xx, yy, Z, colors=['#A9A9A9','#696969','#A9A9A9'], levels=[-1, 0, 1],linestyles=['--', 'dashdot', '--'])

# plot vectors
x_sv = svm.support_vectors_
idsv = svm.support_
y_sv = y_train[idsv]
ps_sv = x_sv[y_sv==1]
neg_sv = x_sv[y_sv==-1]
x_nsv = np.delete(X_train, idsv, 0)
y_nsv = np.delete(y_train, idsv, 0)
ps_nsv = x_nsv[y_nsv == 1]
neg_nsv = x_nsv[y_nsv == -1]
ax.scatter(ps_nsv[:,0], ps_nsv[:,1], color = '#1E90FF', s = 7, label = 'nonsupport vectors')
ax.scatter(neg_nsv[:,0], neg_nsv[:,1], color = '#FA8072', s = 7)
ax.scatter(ps_sv[:,0], ps_sv[:,1], color = 'blue', s = 7, label = 'support vectors')
ax.scatter(neg_sv[:,0], neg_sv[:,1], color = 'red', s = 7)
ax.legend()
ax.set_title("Support vectors in rbf SVM")
plt.savefig('rbfsvm.png')