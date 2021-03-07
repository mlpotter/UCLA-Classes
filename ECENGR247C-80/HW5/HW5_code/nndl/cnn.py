import numpy as np

from nndl.layers import *
from nndl.conv_layers import *
from cs231n.fast_layers import *
from nndl.layer_utils import *
from nndl.conv_layer_utils import *

import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.use_batchnorm = use_batchnorm
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Initialize the weights and biases of a three layer CNN. To initialize:
    #     - the biases should be initialized to zeros.
    #     - the weights should be initialized to a matrix with entries
    #         drawn from a Gaussian distribution with zero mean and 
    #         standard deviation given by weight_scale.
    # ================================================================ #
    self.params["W1"] = weight_scale * np.random.randn(num_filters,3,filter_size,filter_size)
    self.params["b1"] = np.zeros(num_filters,)

    
    flattened_size = (1 + (input_dim[1]-2)/2)
    
    self.params["W2"] = weight_scale * np.random.randn(int(flattened_size**2 * num_filters),hidden_dim)
    self.params["b2"] = np.zeros(hidden_dim,)
    
    self.params["W3"] =  weight_scale * np.random.randn(hidden_dim,num_classes)
    self.params["b3"] = np.zeros(num_classes,)

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of the three layer CNN.  Store the output
    #   scores as the variable "scores".
    # ================================================================ #
    scores, cache_l1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    output_shape = scores.shape
    scores, cache_l2 = affine_relu_forward(scores, W2, b2)
    scores, cache_l3 = affine_forward(scores, W3, b3)
    
    #scores = np.exp(scores-np.max(scores,1)[:,np.newaxis])/np.sum(np.exp(scores-np.max(scores,1)[:,np.newaxis]),1)[:,np.newaxis]

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    if y is None:
      return scores
    
    loss, grads = 0, {}
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backward pass of the three layer CNN.  Store the grads
    #   in the grads dictionary, exactly as before (i.e., the gradient of 
    #   self.params[k] will be grads[k]).  Store the loss as "loss", and
    #   don't forget to add regularization on ALL weight matrices.
    # ================================================================ #
    soft_loss, dx = softmax_loss(scores, y)
    reg_loss = 0.5 * (np.linalg.norm(W1)**2 + np.linalg.norm(W2)**2 + np.linalg.norm(W3)**2)
    loss = soft_loss + self.reg*reg_loss
    
    dx, dw, db = affine_backward(dx, cache_l3)
    grads['W3'] = dw + 0.5*self.reg*2*W3
    grads['b3'] = db
          
    dx, dw, db = affine_relu_backward(dx, cache_l2)
    grads['W2'] = dw + 0.5*self.reg*2*W2
    grads['b2']= db

    dx = dx.reshape(*output_shape)
    
    _, dw, db = conv_relu_pool_backward(dx, cache_l1)
    grads['W1'] = dw + 0.5*self.reg*2*W1
    grads['b1']= db
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return loss, grads
  