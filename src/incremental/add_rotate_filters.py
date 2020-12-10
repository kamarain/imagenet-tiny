#!/usr/bin/env python
"""
adding_rotate.py
It is used to the network with added-roteated filters.
-- liuyuan
"""

import numpy as np

caffe_root = '../'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

from functions import loadTiny, loadLarge, trans, rotateArray

# define files
file_tiny_net = 'path/to/tiny/network/deploy/file/deploy-net-128.prototxt'
file_tiny_model = 'path/to/tiny/network/model/file/caffe_ImageNet-tiny-128x128-sRGB-net-128_iter_120000.caffemodel'
file_large_net = 'path/to/large/network/deploy/file/deploy-net-256.prototxt'
file_large_model = 'path/to/large/network/model/file/trans_128_256.caffemodel'

# load tiny images' net, save parameters, and output the dimension of parameters
net_tiny = caffe.Net( caffe_root + file_tiny_net, caffe_root + file_tiny_model) # load files
layers_tiny, params_tiny = loadTiny(net_tiny) # load layers and parameters
for i in layers_tiny: # output the dimension of weights and bias in each layer
    print '{} weights are {} dimensional and biases are {} dimensional'.format(i, params_tiny[i][0].shape, params_tiny[i][1].shape)

# load large images' net, and output the dimension of parameters
net_large = caffe.Net( caffe_root + file_large_net) # load files
layers_large, params_large = loadLarge(net_large) # load layers and initialise parameters
net_large.save(caffe_root + file_large_model) # initialise new caffemodel file
net_large = caffe.Net( caffe_root + file_large_net, caffe_root + file_large_model) # reload files
params_large = {pr: (net_large.params[pr][0].data, net_large.params[pr][1].data) for pr in layers_large} # reload parameters

for i in layers_large: # output the dimension of weights and bias in each layer
    print '{} weights are {} dimensional and biases are {} dimensional'.format(i, params_large[i][0].shape, params_large[i][1].shape)


# transform bias
# conv1 - double copy from tiny net
# fc6 -  copy from tiny net
# fc8 -  copy from tiny net
for pr_tiny, pr_large in zip(layers_tiny, layers_large):
    if params_large[pr_large][1].shape[3] >= params_tiny[pr_tiny][1].shape[3]: # check the bias dimension
        temp = params_tiny[pr_tiny][1].shape[3]
        params_large[pr_large][1][:,:,:,0:temp] = params_tiny[pr_tiny][1] # copy bias from tiny to large
        if params_large[pr_large][1].shape[3] > params_tiny[pr_tiny][1].shape[3]: # convolutional layer 12-23 filters
            params_large[pr_large][1][:,:,:,temp:] = params_large[pr_large][1][:,:,:,0:temp] # copy from 0-11 filters   
    else:
        sys.exit('Error: The dimension of {} layer bias are not matched.'.format(pr_tiny))


# transform weights
# conv1 - copy from tiny net, set new filters' weights as flip(vertically, horizontally, centrally, diagonally) and copy from tiny net
# fc6 -  copy from tiny net, set new weights as flip and copy from tiny net
# fc8 -  copy from tiny net
for pr_tiny, pr_large in zip(layers_tiny, layers_large):
    # get dimension of weights
    dim1_tiny, dim2_tiny, dim3_tiny, dim4_tiny = params_tiny[pr_tiny][0].shape # the dimension of weights - tiny
    dim1_large, dim2_large, dim3_large, dim4_large = params_large[pr_large][0].shape # the dimension of weights - large
    W_tiny = params_tiny[pr_tiny][0] # get tiny weights
    W_large = np.zeros((dim1_large, dim2_large, dim3_large, dim4_large)) # initialise large weights
    if pr_large == 'conv1': # convolution layer's parameters transform
        if dim1_large >= dim1_tiny and dim3_large%dim3_tiny == 0 and dim4_large%dim4_tiny == 0: # check kernel size
            for i in range(dim1_large):
                for j in range(dim2_tiny):
                    if i < dim1_tiny: # 0 - 11
                        W_large[i,j,:,:] = np.repeat(np.repeat(W_tiny[i,j,:,:], dim3_large/dim3_tiny, axis=1), dim3_large/dim3_tiny, axis=0)/4 # transform and save weights
                    else: # 12 - 23
                        #W_large[i,j,:,:] = np.flipud(W_large[i-12,j,:,:]) # Flip an array vertically
                        #W_large[i,j,:,:] = np.fliplr(W_large[i-12,j,:,:]) # Flip an array horizontally
                        #W_large[i,j,:,:] = np.fliplr(np.flipud(W_large[i-12,j,:,:])) # Flip an array centrally
                        W_large[i,j,:,:] = np.transpose(W_large[i-12,j,:,:]) # Flip an array diagonally
        else:
            sys.exit('Error: The dimension of {} layer weight are not matched.'.format(pr_large))
    elif pr_large == 'fc6': # fc6 layer's parameters transform
        n_tiny, h_tiny, w_tiny = net_tiny.blobs['conv1'].data.shape[1:] # the dimension of convolution layer's output - tiny
        h_large, w_large = net_large.blobs['conv1'].data.shape[2:] # the dimension of convolution layer's output - large
        if dim3_large >= dim3_tiny and h_large == h_tiny*2-1 and w_large == w_tiny*2-1:
            for i in range(dim3_large): # 512
                temp = W_tiny[:,:,i,:].reshape(n_tiny, h_tiny, w_tiny) # reshape
                temp_new = trans(temp, n_tiny, h_large, w_large) # transform
                W_large[:,:,i,0:n_tiny*h_large*w_large] = temp_new.reshape(1, n_tiny*h_large*w_large) # reshape and save 0-11 filters
                temp_new = rotateArray(temp_new) # flip
                W_large[:,:,i,n_tiny*h_large*w_large:] = temp_new.reshape(1, n_tiny*h_large*w_large) # 12-23 filters flip and copy from 0-11 filters
        else:
            sys.exit('Error: The dimension of {} layer weights are not matched!'.format(pr_large))
    else: # fc8 layer's parameters transform, do not need to extend, just copy
        if dim3_large == dim3_tiny:
            W_large = W_tiny
        else:
            sys.exit('Error: The dimension of {} layer weights are not matched!'.format(pr_large))

    params_large[pr_large][0][...] = W_large # save new weights


# save new caffemodel file
net_large.save(caffe_root + file_large_model)
