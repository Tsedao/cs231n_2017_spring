import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  # calculate the loss
  num_train = X.shape[0]
  num_class = W.shape[1]
  for i in range(0,num_train):
  	s = X[i,:].dot(W)
  	idx = y[i]
  	denominator = 0
  	for j in range(0,num_class):
  		denominator += np.exp(s[j])
  		if j == idx:
  			numerator = np.exp(s[j])
  	loss += np.log(numerator/denominator) * (-1)
  loss = 1./num_train * loss + reg*np.sum(W*W)

  # calculate the gradient
  dL=1
  for i in range(0, num_train):
  	s = X[i,:].dot(W)
  	dL_i = 1/num_train * dL
  	for j in range(0,num_class):
  		de_j = (1/np.sum(np.exp(s)))*dL_i
  		if j == y[i]:
  			de_j += (1/np.exp(s[j]))*(-1)*dL_i
  		ds_j = np.exp(s[j])*de_j
  		dW[:,j] += X[i,:] * ds_j
  dW += 2*reg*W 


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  # calculate the vectorized loss
  num_train = X.shape[0]
  num_class = W.shape[1]
  S = np.exp(X.dot(W))
  row_sum = np.sum(S,axis=1)
  target_S = S[range(num_train),y]
  loss = np.sum(-1*np.log(target_S / row_sum))/num_train 
  loss += reg*np.sum(W*W)

  # calculate the vectorized gradient
  dL = 1
  dL_i = 1/num_train * dL
  de_j = (1/row_sum).reshape(-1,1) * np.ones((num_train,num_class))*dL_i
  de_j[range(num_train),y] += (-1/S[range(num_train),y])*dL_i
  ds_j = S * de_j
  dW = X.T.dot(ds_j)
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

