from json import load
from tkinter.filedialog import LoadFileDialog
import numpy as np
import scipy.io as sio
import utils  as utils
import public_tests as PublicTest
import logistic_reg as log_reg

######################################################################### 
# one-vs-all
#
def loadData():
    data = sio.loadmat('data/ex3data1.mat', squeeze_me=True)
    X = data['X']
    y = data['y']
    rand_indices = np.random.choice(X.shape[0], 100, replace=False)
    utils.displayData(X[rand_indices, :])

    return X, y

def oneVsAll(X, y, n_labels, lambda_):
    """
     Trains n_labels logistic regression classifiers and returns
     each of these classifiers in a matrix all_theta, where the i-th
     row of all_theta corresponds to the classifier for label i.

     Parameters
     ----------
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
     -------
     all_theta : array_like
         The trained parameters for logistic regression for each class.
         This is a matrix of shape (K x n+1) where K is number of classes
         (ie. `n_labels`) and n is number of features without the bias.
     """
    all_theta = (n_labels, X.shape[1]+1)

    for i in range(n_labels):
        b = 0
        w = np.zeros(len(X[0]))
        y_aux = np.where(y == i, 1, 0)
        w_aux, b_aux, J_history = log_reg.gradient_descent(X, y_aux, w, b, log_reg.compute_cost_reg,log_reg.compute_gradient_reg, 0.001, 5000, lambda_)
        all_theta[i, 0] = b_aux
        all_theta[i, 1:] = w_aux


    return all_theta


def predictOneVsAll(all_theta, X):
    """
    Return a vector of predictions for each example in the matrix X. 
    Note that X contains the examples in rows. all_theta is a matrix where
    the i-th row is a trained logistic regression theta vector for the 
    i-th class. You should set p to a vector of values from 0..K-1 
    (e.g., p = [0, 2, 0, 1] predicts classes 0, 2, 0, 1 for 4 examples) .

    Parameters
    ----------
    all_theta : array_like
        The trained parameters for logistic regression for each class.
        This is a matrix of shape (K x n+1) where K is number of classes
        and n is number of features without the bias.

    X : array_like
        Data points to predict their labels. This is a matrix of shape 
        (m x n) where m is number of data points to predict, and n is number 
        of features without the bias term. Note we add the bias term for X in 
        this function. 

    Returns
    -------
    p : array_like
        The predictions for each data point in X. This is a vector of shape (m, ).
    """

    p = np.zeros(X.shape[0])

    p = np.argmax(log_reg.sigmoid(all_theta), axis = 0) #SUUUUS


    return p


#########################################################################
# NN
#
def predict(theta1, theta2, X):
    """
    Predict the label of an input given a trained neural network.

    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size)

    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size)

    X : array_like
        The image inputs having shape (number of examples x image dimensions).

    Return 
    ------
    p : array_like
        Predictions vector containing the predicted label for each example.
        It has a length equal to the number of examples.
    """

    return 0#p

def main():
    PublicTest.compute_cost_reg_test(log_reg.compute_cost_reg)
    PublicTest.compute_gradient_reg_test(log_reg.compute_gradient_reg)

    X,y = loadData()
    all_theta = oneVsAll(X, y, 10, 1)
    pred = predictOneVsAll(all_theta, X)
    
    for i in range(pred.shape[0]):
        

main()
