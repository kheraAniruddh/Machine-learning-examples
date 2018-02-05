# final train error: 0.0613333333333 p 0 T 100
# final test error: 0.0768269831355
# final train error: 0.054 p 0 T 200
# final test error: 0.0649594003748
# final train error: 0.0483333333333 p 0 T 500
# final test error: 0.0624609618988
# final train error: 0.0406666666667 p 0 T 1000
# final test error: 0.0655840099938

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
data =  pd.read_csv("spam.csv", header =None)

#Normalize data
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(data.head(3000))
train_data = (scaled_values)

scaled_values = scaler.fit_transform(data.tail(1601))
test_data = (scaled_values)
train_X = train_data[:,0:57]
train_Y = train_data[:,57]
train_Y[train_Y == 0] = -1 # make the targets -1,+1

test_X = test_data[:,0:57]
test_Y = test_data[:,57]
test_Y[test_Y == 0] = -1 # make the targets -1,+1


class AdaBoost:
    def __init__(self, M):
        self.M = M

    def fit(self, X, Y,p):
        self.models = []
        self.alphas = []

        N, _ = X.shape
        W = np.ones(N) / N

        for m in range(self.M):
            tree = DecisionTreeClassifier(max_depth=1)
            tree.fit(X, Y, sample_weight=W)
            P = tree.predict(X)

            err = W.dot(P != Y)
            alpha = 0.5*(np.log((1 - err)*(1-p)) - np.log((err)*(1+p)))
            z = 2 * (((1-err)*err)/(1-p**2))**0.5
            W = W*np.exp(-alpha*Y*P) # vectorized form
            W = W / z # normalize so it sums to 1

            self.models.append(tree)
            self.alphas.append(alpha)

    def predict(self, X):
        # NOT like SKLearn API
        # we want accuracy and exponential loss for plotting purposes
        N, _ = X.shape
        FX = np.zeros(N)
        for alpha, tree in zip(self.alphas, self.models):
            FX += alpha*tree.predict(X)
        return np.sign(FX), FX

    def score(self, X, Y):
        # NOT like SKLearn API
        # we want accuracy and exponential loss for plotting purposes
        P, FX = self.predict(X)
        L = np.exp(-Y*FX).mean()
        return np.mean(P == Y), L


if __name__ == '__main__':

 Ntrain = len(train_data)
 Xtrain, Ytrain = train_X, train_Y
 Xtest, Ytest = test_X, test_Y
 T = [100,200,500,1000]
 for pow_p in range(-10,0):
  p = 2**pow_p
  for t in T:

    model = AdaBoost(t)
    model.fit(Xtrain, Ytrain,p)
    acc, loss = model.score(Xtest, Ytest)
    acc_train, _ = model.score(Xtrain, Ytrain)
    train_errors = 1 - acc_train
    #    test_errors[num_trees] = 1 - acc
     #   test_losses[num_trees] = los
    print ("final train error:", train_errors,"p",pow_p,"T",t)
    print ("final test error:", 1 - acc)
