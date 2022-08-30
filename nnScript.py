import numpy as np
from numpy import exp
from scipy.optimize import minimize
import pickle
from scipy.io import loadmat
from math import sqrt
import time


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    sigZ=1/(1+exp(-z))
    return  sigZ# your code here


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.
    # splitting data as instructed
    # initiaizing arrays for pre process
    train_data_preprocess = np.zeros(shape=(50000, 784))
    validation_data_preprocess = np.zeros(shape=(10000, 784))
    test_data_preprocess = np.zeros(shape=(10000, 784))
    train_label_preprocess = np.zeros(shape=(50000,))
    validation_label_preprocess = np.zeros(shape=(10000,))
    test_label_preprocess = np.zeros(shape=(10000,))


    # Initializing flags
    train_length = 0
    validation_length = 0
    test_length = 0
    train_label_length = 0
    validation_label_length = 0
   

    for key in mat:
        if "train" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_permutation = np.random.permutation(sap)
            tup_length = len(tup)
            tag_length = tup_length - 1000

            # adding data to training set
            train_data_preprocess[train_length:train_length + tag_length] = tup[tup_permutation[1000:], :]
            train_length += tag_length

            train_label_preprocess[train_label_length:train_label_length + tag_length] = label
            train_label_length += tag_length

            # adding data to validation set
            validation_data_preprocess[validation_length:validation_length + 1000] = tup[tup_permutation[0:1000], :]
            validation_length += 1000

            validation_label_preprocess[validation_label_length:validation_label_length + 1000] = label
            validation_label_length += 1000

        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_permutation = np.random.permutation(sap)
            tup_length = len(tup)
            test_label_preprocess[test_length:test_length + tup_length] = label
            test_data_preprocess[test_length:test_length + tup_length] = tup[tup_permutation]
            test_length += tup_length
    
    # process of shuffling, doubling and normalization      
    train_size = range(train_data_preprocess.shape[0])
    train_permutation = np.random.permutation(train_size)
    train_data = train_data_preprocess[train_permutation]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = train_label_preprocess[train_permutation]

    validation_size = range(validation_data_preprocess.shape[0])
    valid_permutation = np.random.permutation(validation_size)
    validation_data = validation_data_preprocess[valid_permutation]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = validation_label_preprocess[valid_permutation]

    test_size = range(test_data_preprocess.shape[0])
    test_permutation = np.random.permutation(test_size)
    test_data = test_data_preprocess[test_permutation]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = test_label_preprocess[test_permutation]

    # Feature selection
    # Your code here.

    data_allfeatures = np.array(np.vstack((train_data,validation_data,test_data)))
    num_feature = data_allfeatures.shape[1]

    column_indexes = np.arange(num_feature)

    same_val = np.all(data_allfeatures == data_allfeatures[0,:], axis = 0)
    column_id = column_indexes[same_val]
    listtotal = np.arange(784)
    
    global indices
    indices = 0
    indices = np.setdiff1d(listtotal,column_id)
    data_redfeatures = np.delete(data_allfeatures,column_id,axis=1)
    num_features = data_redfeatures.shape[1]
    
    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    #
    #
    #
    #
    #
    n = training_data.shape[0]
    # Feed forward code
    w1_transpose = np.transpose(w1)
    w2_transpose = np.transpose(w2)

    input_bias = np.ones(shape=(n, 1), dtype = np.float64) 
    biased_training_data = np.append(training_data, input_bias, axis=1)
    
    aj = np.dot(biased_training_data, w1_transpose) 
    zj = sigmoid(aj) 
    
    hidden_bias = np.ones(shape=(zj.shape[0], 1), dtype = np.float64) 
    biased_zj = np.append(zj, hidden_bias, axis=1)
    
    bl = np.dot(biased_zj, w2_transpose)
    ol = sigmoid(bl)
    
    # Labelling output
    yl = np.zeros(shape=(n, 10), dtype = np.float64) 
        
    for i in range(yl.shape[0]):   
        for j in range(yl.shape[1]):
            if j==training_label[i]:
                yl[i][j] = 1.0             
        
    # Error function
    
    p = yl*np.log(ol) 
    q = (1-yl)*np.log(1-ol)
    sum1 = np.sum(p + q)
    constant = -1*n
    error = sum1/constant 
    
    # Regularised error function  
    
    w1_square_sum = np.sum(np.square(w1))
    w2_square_sum = np.sum(np.square(w2))
    sum2 = w1_square_sum + w1_square_sum                
    reg_factor = (sum2*lambdaval)/(2*n)
    reg_error = error + reg_factor
    
    obj_val = reg_error 
    
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.array([])
    delta_l = ol-yl
    delta_l_transpose = np.transpose(delta_l)
    
    grad_w2 = np.dot(delta_l_transpose, biased_zj)
    
    r = (1-biased_zj[:,0:n_hidden])*biased_zj[:,0:n_hidden]
    s = np.dot(delta_l, w2[:,0:n_hidden])
    rs = r*s
    rs_transpose = np.transpose(rs)
    grad_w1 = np.dot(rs_transpose, biased_training_data)
    
    # Regularised gradients
    
    reg_grad_w2 = (grad_w2 + (lambdaval*w2))/n
    reg_grad_w1 = (grad_w1 + lambdaval*w1)/n
    obj_grad = np.concatenate((reg_grad_w1.flatten(), reg_grad_w2.flatten()),0)
    
    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here

    d_bias = np.ones((data.shape[0], 1),dtype = np.uint8)
    data = np.column_stack([data, d_bias])

    #Creating o/p layer
    output = sigmoid(np.dot(data, w1.transpose()))

    #Making Bias (output size)
    o_bias = np.ones((output.shape[0], 1),dtype = np.uint8)

    # Adding, Dot product and Max value output
    output = np.column_stack([output, o_bias])
    labels = sigmoid(np.dot(output, w2.transpose()))
    labels = np.argmax(labels, axis=1)

    return labels


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
#n_hidden_array = [4,8,12,16,20]

#for x in n_hidden_array:
#    for y in range(0,70,10):
        
# set the number of nodes in hidden unit (not including bias unit)
#n_hidden = x # values - 4,8,12,16,20
n_hidden = 20 # Optimal hidden units

# set the number of nodes in output unit
n_class = 10

# Note current time
t1 = time.time()

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 10 # Optimal lambda value

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

t2 = time.time()

print('\n Time taken:'+str(t2-t1))

global indices
obj = [indices, n_hidden, w1, w2, lambdaval]
# selected_features is a list of feature indices that you use after removing unwanted features in feature selection step

pickle.dump(obj, open('params.pickle', 'wb'))