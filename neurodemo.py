#!/usr/bin/env python

# TODO: Maybe remove later
from sklearn.datasets import load_iris
from sknn.mlp import Classifier, Layer
import numpy as np

def split_train_test(X, y):
    assert X.shape[0] == y.shape[0]

    indices = np.random.permutation(X.shape[0])

    training_idx, test_idx = indices[:80], indices[80:]

    X_train = X[training_idx,:]
    y_train = y[training_idx]
    
    X_test = X[test_idx,:]
    y_test = y[test_idx]

    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    X = load_iris()['data']
    y = load_iris()['target']

    X_train, y_train, X_test, y_test = split_train_test(X, y)


    nn = Classifier(
        layers = [
            Layer("Maxout", units=100, pieces=2),
            Layer("Softmax")],
        learning_rate=0.001,
        n_iter=25)

    nn.fit(X_train, y_train)

    y_predict = nn.predict(X_test)
    y_predict = y_predict.reshape(y_predict.shape[0],)

    accuracy = float(np.sum(y_predict == y_test)) / len(y_test)

    print "Accuracy: %.2f %%" % (accuracy * 100)
