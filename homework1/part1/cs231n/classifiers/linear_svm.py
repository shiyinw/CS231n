import numpy as np
from random import shuffle
import sys
sys.path.append('/Users/sherilynw/Desktop/3_1/人工智能：原理与技术/CS231n/homework1/part1/')
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
  Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    M = np.zeros(shape=(num_train, num_classes))
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i].T
                dW[:, y[i]] -= X[i].T
                M[i, j] = margin
    # Y = WX  ->  L = \simga \max (w_j x - w_{y_j} x + 1)

                # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg*W


    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    scores = X.dot(W)

    # duplicate the right class score num_classes times to replace the other items in the score matrix
    correct_class_scores = np.zeros((num_train, num_classes))
    correct_class_scores[np.arange(num_train), ]=np.repeat(scores[np.arange(num_train), y], num_classes).reshape(num_train, num_classes)
    margin = scores - correct_class_scores + np.ones((num_train, num_classes))
    L = np.maximum(np.zeros((num_train, num_classes)), margin)

    # set the right class scores to 0
    L[np.arange(num_train), y] = 0
    loss = np.sum(L)/num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################

    I = np.zeros(shape = (num_train, num_classes))
    I[margin>0] = 1
    I[range(num_train), list(y)] = 0
    I[range(num_train), list(y)] = -np.sum(I, axis=1)

    dW = (X.T).dot(I)/num_train

    # Add regularization to the loss.
    dW += reg * W
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW


if __name__ == "__main__":
    import pickle
    (X_dev, y_dev) = pickle.load(open("svm_dev.pickle", 'rb'))
    W = np.random.randn(3073, 10) * 0.0001

    loss, grad1 = svm_loss_naive(W, X_dev, y_dev, 0.000005)
    print('loss: %f' % (loss))

    loss, grad2 = svm_loss_vectorized(W, X_dev, y_dev, 0.000005)
    print('loss: %f' % (loss))

    print(np.sum(grad2-grad1))