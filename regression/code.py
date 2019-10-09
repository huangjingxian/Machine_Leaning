import numpy as np
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def read_data():
    data = []
    with open('transfusion.data') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader) # take the header out
        for row in reader: # each row is a list
            data.append(row)
    data  = np.array(data, dtype = np.int32)
    X = data[:,:-1]
    y = data[:,-1]
    
    return X, y

X, y = read_data()

C = [0.1, 1, 10, 100] # Hyperparameter for regularization

split_X = []
split_y = []
i = 0
step =  X.shape[0]//5
for j in range(4):
	split_X.append(X[i:i + step])
	split_y.append(y[i:i + step])
	i += step
split_X.append(X[i:])
split_y.append(y[i:])

tt_f1 = np.zeros(5)
for i in range(5):
    test_X = split_X[i]
    test_y = split_y[i]
    val_train_X = split_X[:i] + split_X[i + 1:]
    val_train_y = split_y[:i] + split_y[i + 1:]
    C_score = {0.1:[], 1:[], 10:[], 100:[]}
    for j in range(4):
        val_X = val_train_X[j]
        val_y = val_train_y[j]
        train_X = np.concatenate(val_train_X[:i] + val_train_X[i+1:])
        train_y = np.concatenate(val_train_y[:i] + val_train_y[i+1:])
        for c in C:
            model = LogisticRegression(C=c)
            model.fit(train_X, train_y)
            val_y_pred = model.predict(val_X)
            f1 = f1_score(val_y, val_y_pred)
            C_score[c].append(f1)
    C_ave_score = {}
    for key in C_score.keys():
        ave = sum(C_score[key])/4
        C_ave_score[key] = ave
    tmpC = sorted(C_ave_score.items(), key = lambda item:item[1],reverse=True)
    choosen_c = tmpC[0][0]
    print(choosen_c)
    new_train_X = np.concatenate(val_train_X)
    new_train_y = np.concatenate(val_train_y)
    model = LogisticRegression(C=choosen_c)
    model.fit(new_train_X, new_train_y)
    test_y_pred = model.predict(test_X)
    f1 = f1_score(test_y, test_y_pred)
    tt_f1[i] = f1
print(np.sum(tt_f1), np.std(tt_f1))






