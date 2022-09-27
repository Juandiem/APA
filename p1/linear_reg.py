import numpy as np
import copy
import math
import public_tests as PublicTest


#########################################################################
# Cost function
#
def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities)
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model

    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    total_cost = 0
    #Sumatorio
    for i in range(x.shape[0]-1):
        cost = (w*x[i] + b) - y[i]
        #Cuadrado del sumatorio i
        total_cost += pow(cost,2)
    #Division
    total_cost = total_cost /(2*x.shape[0])

    return total_cost


#########################################################################
# Gradient function
#
def compute_gradient(x, y, w, b):
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray): Shape (m,) Input to the model (Population of cities) 
      y (ndarray): Shape (m,) Label (Actual profits for the cities)
      w, b (scalar): Parameters of the model  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    dj_dw = dj_db = 0
    
    #dj_dw
    for i in range(x.shape[0]-1):
        dj_dw += (w*x[i] + b) - y[i]
    # result
    dj_dw = dj_dw /x.shape[0]
    #dj_db
    for i in range(x.shape[0]-1):
        aux = (w*x[i] + b) - y[i]
        dj_db += aux * x[i]
    # result
    dj_db = dj_db /x.shape[0]


    return dj_dw, dj_db


#########################################################################
# gradient descent
#
def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha

    Args:
      x :    (ndarray): Shape (m,)
      y :    (ndarray): Shape (m,)
      w_in, b_in : (scalar) Initial values of parameters of the model
      cost_function: function to compute cost
      gradient_function: function to compute the gradient
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (ndarray): Shape (1,) Updated values of parameters of the model after
          running gradient descent
      b : (scalar) Updated value of parameter of the model after
          running gradient descent
      J_history : (ndarray): Shape (num_iters,) J at each iteration,
          primarily for graphing later
    """
    #Valores iniciales
    w = copy.deepcopy(w_in)
    b = b_in
    J_history = []
    #iteracion hasta convergencia
    for i in range(num_iters - 1):
        w = w - alpha * gradient_function(x,y,w,b)
        b = b - alpha * gradient_function(x,y,w,b)
        #historial coste j
        J_history.append(cost_function)

    return w, b, J_history


def main():
    PublicTest.compute_cost_test(compute_cost)
    PublicTest.compute_gradient_test(compute_gradient)

main()