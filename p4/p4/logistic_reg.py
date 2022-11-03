import numpy as np
import copy
import math
import public_tests as PublicTest
import utils as utils
import scipy.io as loadmat

def plot_data(X, y, pos_label="y=1", neg_label="y=0"):
  positive = y == 1
  negative = y == 0

  plt.plot(X[positive, 0], X[positive, 1], 'k+', label=pos_label)
  plt.plot(X[negative, 0], X[negative, 1], 'yo', label=neg_label)

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
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    for i in range(X.shape[0]):
      X_norm[i] = (X[i] - mu) / (sigma)


    return X_norm, mu, sigma

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

  f_dw = sigmoid(X @ w +b)

  a = np.dot(-y,np.log(f_dw))
  b = np.dot((1-y),np.log(1-f_dw))
  sum = np.sum(a, b)

  total_cost = sum/X.shape[0]

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

  f_dw = sigmoid(X @ w +b)
  dj_dw = (f_dw - y) @ X
  dj_db = np.sum(f_dw, y)

  return dj_db/m, dj_dw/m


#########################################################################
# regularized logistic regression
#
def compute_cost_reg(X, y, w, b, lambda_=1):
  """
  Computes the cost over all examples
  Args:
    X : (array_like Shape (m,n)) data, m examples by n features
    y : (array_like Shape (m,)) target value 
    w : (array_like Shape (n,)) Values of parameters of the model      
    b : (array_like Shape (n,)) Values of bias parameter of the model
    lambda_ : (scalar, float)    Controls amount of regularization
  Returns:
    total_cost: (scalar)         cost 
  """
  total_cost = 0
  cost_a = cost_b = 0
  m = X.shape[0]

  for i in range(m):     
    cost_a += -y[i]*np.log(sigmoid(np.dot(w, X[i]) + b)) - (1-y[i])*np.log(1 - sigmoid(np.dot(w, X[i]) + b))
  for j in range(X.shape[1]):     
    cost_b += w[j]**2

  total_cost = cost_a/m + cost_b*lambda_/(2*m)

  return total_cost


def compute_gradient_reg(X, y, w, b, lambda_=1):
  """
  Computes the gradient for linear regression 

  Args:
    X : (ndarray Shape (m,n))   variable such as house size 
    y : (ndarray Shape (m,))    actual value 
    w : (ndarray Shape (n,))    values of parameters of the model      
    b : (scalar)                value of parameter of the model  
    lambda_ : (scalar,float)    regularization constant
  Returns
    dj_db: (scalar)             The gradient of the cost w.r.t. the parameter b. 
    dj_dw: (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 

  """
  m = X.shape[0]

  dj_dw = np.zeros(X.shape[1])
  dj_db = 0

  for i in range(m):
    dj_db += sigmoid(np.dot(w, X[i]) + b) - y[i]
    dj_dw += (sigmoid(np.dot(w, X[i]) + b) - y[i])*X[i]

  dj_dw /= m
  dj_dw += (lambda_/m)*w

  return dj_db/m, dj_dw


# #########################################################################
# # gradient descent
# #
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_=None):
  """
  Performs batch gradient descent to learn theta. Updates theta by taking 
  num_iters gradient steps with learning rate alpha

  Args:
    X :    (array_like Shape (m, n)
    y :    (array_like Shape (m,))
    w_in : (array_like Shape (n,))  Initial values of parameters of the model
    b_in : (scalar)                 Initial value of parameter of the model
    cost_function:                  function to compute cost
    alpha : (float)                 Learning rate
    num_iters : (int)               number of iterations to run gradient descent
    lambda_ (scalar, float)         regularization constant

  Returns:
    w : (array_like Shape (n,)) Updated values of parameters of the model after
        running gradient descent
    b : (scalar)                Updated value of parameter of the model after
        running gradient descent
    J_history : (ndarray): Shape (num_iters,) J at each iteration,
        primarily for graphing later
  """
  w = w_in
  b = b_in
  J_history = []

  J_history += [cost_function(X, y, w, b)]
  for n in range(num_iters):
      w_aux, b_aux = gradient_function(X, y, w, b)
      w -= alpha*w_aux
      b -= alpha*b_aux
      J_history += [cost_function(X, y, w, b)]

  return w, b, J_history


# #########################################################################
# # predict
# #
def predict(X, w, b):
  """
  Predict whether the label is 0 or 1 using learned logistic
  regression parameters w and b

  Args:
  X : (ndarray Shape (m, n))
  w : (array_like Shape (n,))      Parameters of the model
  b : (scalar, float)              Parameter of the model

  Returns:
  p: (ndarray (m,1))
      The predictions for X using a threshold at 0.5
  """
  p = np.zeros((X.shape[0]))

  for i in range(X.shape[0]):#if mayor que 0.5 1 else 0
    p[i] = np.dot(w, X[i]) + b

  return p

#Practica 4 ahora mas que nunca
def oneVsAll(X, y, n_labels, lambda_):
  """
  Trains n_labels logistic regression classifiers and returns 
  each of these classifiers in a matrix all_theta, where the i-th 
  row of all_theta corresponds to the classifier for label i. 
  
  Parameters 
  ---------
  X : array_like 
  The input dataset of shape (m x n). m is the number of 
  data points, and n is the number of features. 
  
  y : array_like 
  The data labels. A vector of shape (m, ). 
  
  n_labels : int 
  Number of possible labels. 
  
  lambda_ : float 
  The logistic regularization parameter.
  Returns 
  ------
  all_theta:array_like 
  Thetrainedparametersforlogisticregressionforeachclass. 
  Thisisamatrixofshape(Kxn+1)whereKisnumberofclasses 
  (ie.`n_labels`)andnisnumberoffeatureswithoutthebias. 
  """

  all_theta = np.zeros(n_labels,X.shape[1])

  for i in range(n_labels):
    
   
  return all_theta



def gradient_descent_test():
  data = loadmat
  data = np.loadtxt("./data/ex2data2.txt", delimiter=',', skiprows=1)
  X = data[:, :2]
  y = data[:, 2]
  x, mu, sigma = zscore_normalize_features(X)
  b_in = -8
  w_in = [0,0]
  w, b, J_history = gradient_descent(x, y,w_in, b_in, compute_cost,compute_gradient, 0.001, 10000)

  t = 0
  for i in range(J_history.shape[0]):
    t += J_history[i]
  print(t/J_history.shape[0])

  utils.plot_data(X, y)
  utils.plot_decision_boundary(w,b,X,y)

def gradient_reg_test():
  data = np.loadtxt("./data/ex2data2.txt", delimiter=',', skiprows=1)
  X = data[:, :2]
  y = data[:, 2]
  x, mu, sigma = zscore_normalize_features(X)
  b_in = 1
  w_in = [0,0]
  lambda_ = 0.01
  w, b, J_history = gradient_descent(x, y,w_in, b_in, compute_cost,compute_gradient, 0.001, 10000)

  t = 0
  for i in range(J_history.shape[0]):
    t += J_history[i]
  print(t/J_history.shape[0])
  
  utils.plot_data(X, y)
  utils.plot_decision_boundary(w,b,X,y)
  

def main():
#   data = loadmat('data/ex3data1.mat', squeeze_me=True)
#   X = data['X']
#   y = data['y']

#   rand_indices = np.random.choice(X.shape[0], 100, replace=False) 
#   utils.displayData(X[rand_indices, :])

  PublicTest.sigmoid_test(sigmoid)
  PublicTest.compute_cost_test(compute_cost)
  PublicTest.compute_gradient_test(compute_gradient)
  #gradient_descent_test()

  PublicTest.predict_test(predict)
  PublicTest.compute_cost_reg_test(compute_cost_reg)
  PublicTest.compute_gradient_reg_test(compute_gradient_reg)
  gradient_reg_test()

