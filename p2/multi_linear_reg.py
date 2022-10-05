import numpy as np
import copy
import math
import matplotlib.pyplot as plt
import public_tests as PublicTest

def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column

    Args:
      X (ndarray (m,n))     : input data, m examples, n features

    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    X_norm = np.zeros(X.shape)

    for j in range(X.shape[1]):
      mu = np.mean(X, axis=j)
      sigma = np.std(X, axis=j)
      X_norm[:, j] = (X - mu) / (sigma)


    return (X_norm, mu, sigma)


def compute_cost(X, y, w, b):
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
    Returns
      cost (scalar)    : cost
    """
    cost = 0

    for i in range(X.shape[0]):     
        cost += ((np.dot(w, X[i]) + b - y[i])**2)

    cost = cost/(2*X.shape[0])

    return cost


def compute_gradient(X, y, w, b):
    """
    Computes the gradient for linear regression 
    Args:
      X : (ndarray Shape (m,n)) matrix of examples 
      y : (ndarray Shape (m,))  target value of each example
      w : (ndarray Shape (n,))  parameters of the model      
      b : (scalar)              parameter of the model      
    Returns
      dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 
      dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b. 
    """
    m = X.shape[0]

    dj_db = 0
    dj_dw = np.zeros(X.shape[1])

    for i in range(m):
      dj_db += (np.dot(w, X[i]) + b - y[i])
      dj_dw += ((np.dot(w, X[i]) + b - y[i]))*X[i]

    

    return dj_db/m, dj_dw/m


def gradient_descent(X, y, w_in, b_in, cost_function,
                     gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha

    Args:
      X : (array_like Shape (m,n)    matrix of examples 
      y : (array_like Shape (m,))    target value of each example
      w_in : (array_like Shape (n,)) Initial values of parameters of the model
      b_in : (scalar)                Initial value of parameter of the model
      cost_function: function to compute cost
      gradient_function: function to compute the gradient
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (array_like Shape (n,)) Updated values of parameters of the model
          after running gradient descent
      b : (scalar)                Updated value of parameter of the model 
          after running gradient descent
      J_history : (ndarray): Shape (num_iters,) J at each iteration,
          primarily for graphing later
    """
    w = w_in
    b =  b_in
    J_history = []

    for n in range(num_iters):
      for i in range(X.shape[0]):
          w = np.subtract(w, alpha*gradient_function(X, y, w, b)[1])
          b = b - alpha*gradient_function(X, y, w, b)[0]
      J_history.append(cost_function)

    return w, b, J_history

def main():
  PublicTest.compute_cost_test(compute_cost)
  PublicTest.compute_gradient_test(compute_gradient)

  
  # data = np.loadtxt("./data/houses.txt", delimiter=',', skiprows=1)
  # X_train = data[:, :4]
  # y_train = data[:, 4]

  # X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']
  # fig, ax = plt.subplots(1, 4, figsize=(25, 5), sharey=True)
  # for i in range(len(ax)):
  #   ax[i].scatter(X_train[:, i], y_train)
  #   ax[i].set_xlabel(X_features[i])
  # ax[0].set_ylabel("Price (1000's)")

  # plt.show()

main()

