#!/usr/bin/env python

# TODO: Maybe remove later
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sknn.mlp import Classifier, Layer
import numpy as np
import matplotlib.pyplot as plt

def split_train_test(X, y):
    assert X.shape[0] == y.shape[0]

    indices = np.random.permutation(X.shape[0])

    # 80% train set, 20% test set
    splitpoint = int(X.shape[0] * 0.8)
    training_idx, test_idx = indices[:splitpoint], indices[splitpoint:]

    X_train = X[training_idx,:]
    y_train = y[training_idx]
    
    X_test = X[test_idx,:]
    y_test = y[test_idx]

    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    X = load_iris()['data']
    y = load_iris()['target']

    X_train, y_train, X_test, y_test = split_train_test(X, y)


    # Construct classifier object with one maxout hidden layer and softmax
    # output layer
    nn = Classifier(
        layers = [
            Layer("Maxout", units=100, pieces=2),
            Layer("Softmax")],
        learning_rate=0.001,
        n_iter=25)

    # Train the neural network
    nn.fit(X_train, y_train)

    # Predict using neural network
    y_predict = nn.predict(X_test)
    y_predict = y_predict.reshape(y_predict.shape[0],)

    accuracy = float(np.sum(y_predict == y_test)) / len(y_test)
    print "Accuracy: %.2f %%" % (accuracy * 100)

    # Using Principal Component Analysis, reduce points from test set to two
    # dimensions and visualize using matplotlib's scatter plot
    pca = PCA(n_components = 2)
    pca.fit(X)

    points = pca.transform(X_test)
    plt.scatter(points[:,0], points[:,1], c=y_predict)
    plt.show()
