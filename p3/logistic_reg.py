import numpy as np
import copy
import math
import public_tests as PublicTest
import matplotlib.pyplot as plt

def plot_data(X, y, pos_label="y=1", neg_label="y=0"):
  positive = y == 1
  negative = y == 0

  plt.plot(X[positive, 0], X[positive, 1], 'k+', label=pos_label)
  plt.plot(X[negative, 0], X[negative, 1], 'yo', label=neg_label)

def sigmoid(z):
  """
  Compute the sigmoid of z

  Args:
      z (ndarray): A scalar, numpy array of any size.

  Returns:
      g (ndarray): sigmoid(z), with the same shape as z

  """
  g = 1 / (1 + np.exp(z*-1))

  return g


#########################################################################
# logistic regression
#
def compute_cost(X, y, w, b, lambda_=None):
  """
  Computes the cost over all examples
  Args:
    X : (ndarray Shape (m,n)) data, m examples by n features
    y : (array_like Shape (m,)) target value
    w : (array_like Shape (n,)) Values of parameters of the model
    b : scalar Values of bias parameter of the model
    lambda_: unused placeholder
  Returns:
    total_cost: (scalar)         cost
  """

  total_cost = 0

  for i in range(X.shape[0]):     
      total_cost += -y[i]*np.log(sigmoid(np.dot(w, X[i]) + b)) - (1-y[i])*np.log(1 - sigmoid(np.dot(w, X[i]) + b))

  total_cost /= X.shape[0]

  return total_cost


def compute_gradient(X, y, w, b, lambda_=None):
  """
  Computes the gradient for logistic regression

  Args:
    X : (ndarray Shape (m,n)) variable such as house size
    y : (array_like Shape (m,1)) actual value
    w : (array_like Shape (n,1)) values of parameters of the model
    b : (scalar)                 value of parameter of the model
    lambda_: unused placeholder
  Returns
    dj_db: (scalar)                The gradient of the cost w.r.t. the parameter b.
    dj_dw: (array_like Shape (n,1)) The gradient of the cost w.r.t. the parameters w.
  """

  m = X.shape[0]

  dj_dw = np.zeros(X.shape[1])
  dj_db = 0

  for i in range(m):
    dj_dw += (sigmoid(np.dot(w, X[i]) + b) - y[i])*X[i]
    dj_db += sigmoid(np.dot(w, X[i]) + b) - y[i]

  return dj_db, dj_dw


#########################################################################
# regularized logistic regression
#
# def compute_cost_reg(X, y, w, b, lambda_=1):
#   """
#   Computes the cost over all examples
#   Args:
#     X : (array_like Shape (m,n)) data, m examples by n features
#     y : (array_like Shape (m,)) target value 
#     w : (array_like Shape (n,)) Values of parameters of the model      
#     b : (array_like Shape (n,)) Values of bias parameter of the model
#     lambda_ : (scalar, float)    Controls amount of regularization
#   Returns:
#     total_cost: (scalar)         cost 
#   """

#   return total_cost


# def compute_gradient_reg(X, y, w, b, lambda_=1):
#   """
#   Computes the gradient for linear regression 

#   Args:
#     X : (ndarray Shape (m,n))   variable such as house size 
#     y : (ndarray Shape (m,))    actual value 
#     w : (ndarray Shape (n,))    values of parameters of the model      
#     b : (scalar)                value of parameter of the model  
#     lambda_ : (scalar,float)    regularization constant
#   Returns
#     dj_db: (scalar)             The gradient of the cost w.r.t. the parameter b. 
#     dj_dw: (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 

#   """

#   return dj_db, dj_dw


# #########################################################################
# # gradient descent
# #
# def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_=None):
#   """
#   Performs batch gradient descent to learn theta. Updates theta by taking 
#   num_iters gradient steps with learning rate alpha

#   Args:
#     X :    (array_like Shape (m, n)
#     y :    (array_like Shape (m,))
#     w_in : (array_like Shape (n,))  Initial values of parameters of the model
#     b_in : (scalar)                 Initial value of parameter of the model
#     cost_function:                  function to compute cost
#     alpha : (float)                 Learning rate
#     num_iters : (int)               number of iterations to run gradient descent
#     lambda_ (scalar, float)         regularization constant

#   Returns:
#     w : (array_like Shape (n,)) Updated values of parameters of the model after
#         running gradient descent
#     b : (scalar)                Updated value of parameter of the model after
#         running gradient descent
#     J_history : (ndarray): Shape (num_iters,) J at each iteration,
#         primarily for graphing later
#   """

#   return w, b, J_history


# #########################################################################
# # predict
# #
# def predict(X, w, b):
#   """
#   Predict whether the label is 0 or 1 using learned logistic
#   regression parameters w and b

#   Args:
#   X : (ndarray Shape (m, n))
#   w : (array_like Shape (n,))      Parameters of the model
#   b : (scalar, float)              Parameter of the model

#   Returns:
#   p: (ndarray (m,1))
#       The predictions for X using a threshold at 0.5
#   """

#   return p

def main():
  PublicTest.sigmoid_test(sigmoid)
  PublicTest.compute_cost_test(compute_cost)
  PublicTest.compute_gradient_test(compute_gradient)

main()
