#!/usr/bin/env python
"""
IL.py
It is used to the network with added layers.
-- liuyuan
"""

import numpy as np

caffe_root = '../'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

from functions import loadTiny, loadLarge

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

# add 'empty' to the tiny network
if len(layers_tiny) >= 3 and len(layers_large) > len(layers_tiny):
    layers_tiny.insert(len(layers_tiny)-3,'empty')
else:
    layers_tiny.insert(0,'empty')
"""
# check the list of layers
print layers_tiny
print '\n'
print layers_large
sys.exit(0)
"""
# transform bias - initialisation + copy
for pr_tiny, pr_large in zip(layers_tiny, layers_large):
    # initialisation
    if pr_large == 'conv1' or pr_large == 'conv3' or pr_large == 'fc8':
        params_large[pr_large][1][:,:,:,:] = 0
    else:
        params_large[pr_large][1][:,:,:,:] = 0.01
    # copy bias
    if pr_tiny == pr_large:
        if params_large[pr_large][1].shape[3] >= params_tiny[pr_tiny][1].shape[3]:
            temp = params_tiny[pr_tiny][1].shape[3]
            params_large[pr_large][1][:,:,:,0:temp] = params_tiny[pr_tiny][1]
        else:
            print 'The dimension of tiny network is {}.'.format(params_tiny[pr_tiny][1].shape)
            print 'The dimension of large network is {}.'.format(params_large[pr_large][1].shape)
            sys.exit('Error: The dimension of {} layer bias are not matched.'.format(pr_tiny))

# transform weights
# conv* -> conv*: extend/copy + initialise
# N/A -> conv* : initialise + 1
# fc* -> fc*: copy + initialise
for pr_tiny, pr_large in zip(layers_tiny, layers_large):
    dim1_large, dim2_large, dim3_large, dim4_large = params_large[pr_large][0].shape # the dimension of weights - large
    # initialisation
    if pr_large == 'fc6' or pr_large == 'fc7':
        W_large = np.random.normal(0, 0.05, (dim1_large, dim2_large, dim3_large, dim4_large)) / 100
    else:
        W_large = np.random.normal(0, 0.01, (dim1_large, dim2_large, dim3_large, dim4_large)) / 100
    # the same layers    
    if pr_tiny == pr_large:
        dim1_tiny, dim2_tiny, dim3_tiny, dim4_tiny = params_tiny[pr_tiny][0].shape # the dimension of weights - tiny
        W_tiny = params_tiny[pr_tiny][0] # get tiny weights
        # conv* parameters transformation
        if 'conv' in pr_large:
            # conv1 parameters transformation - extend
            if pr_large == 'conv1':
                if dim1_large >= dim1_tiny and dim3_large > dim3_tiny and dim4_large > dim4_tiny: # check kernel size
                    for i in range(dim1_tiny):
                        for j in range(dim2_tiny):
                            temp = np.repeat(np.repeat(W_tiny[i,j,:,:], 2, axis=1), 2, axis=0)/4
                            if dim3_large%dim3_tiny == 0 and dim4_large%dim4_tiny == 0:
                                W_large[i,j,:,:] = temp
                            else:
                                W_large[i,j,:,:] = temp[0:dim3_large,0:dim4_large]
                                W_large[i,j,:,-1] = W_large[i,j,:,-1] * 2
                                W_large[i,j,-1,:] = W_large[i,j,-1,:] * 2
                else:
                    print 'The dimension of tiny network is {}.'.format(params_tiny[pr_tiny][0].shape)
                    print 'The dimension of large network is {}.'.format(params_large[pr_large][0].shape)
                    sys.exit('Error: The dimension of {} layer weight are not matched.'.format(pr_large))      
            # conv2, conv3, conv4 parameters transformation - copy
            else:
                if dim1_large >= dim1_tiny and dim2_large >= dim2_tiny and dim3_large == dim3_tiny and dim4_large == dim4_tiny: # check kernel size
                    W_large[0:dim1_tiny,0:dim2_tiny,:,:] = W_tiny
                else:
                    print 'The dimension of tiny network is {}.'.format(params_tiny[pr_tiny][0].shape)
                    print 'The dimension of large network is {}.'.format(params_large[pr_large][0].shape)
                    sys.exit('Error: The dimension of {} layer weight are not matched.'.format(pr_large))  
        # fc parameters transformation - copy    
        elif 'fc' in pr_large:
            if dim4_large >= dim4_tiny:
                W_large[:,:,:,0:dim4_tiny] = W_tiny # copy weights
            else:
                print 'The dimension of tiny network is {}.'.format(params_tiny[pr_tiny][0].shape)
                print 'The dimension of large network is {}.'.format(params_large[pr_large][0].shape)
                sys.exit('Error: The dimension of {} layer weights are not matched!'.format(pr_large))
        else:
            sys.exit('Error: The {} layer is unknown!'.format(pr_large))
    # the new added layers
    else:
        if pr_large == 'fc7':
            for i in range(48):
                temp = W_large[:,:,i,:].reshape((3,8,8))
                if i/16 == 0:
                    rgb = 0
                elif i/16 == 1:
                    rgb = 1
                else:
                    rgb = 2
                temp[rgb,i/2:i/2+2,i%4*2:i%4*2+2] = 0.25
                W_large[:,:,i,:] = temp.reshape((1,3*8*8))
        elif pr_large == 'fc6':
            for i in range(192):
                temp = W_large[:,:,i,:].reshape((3,16,16))
                if i/64 == 0:
                    rgb = 0
                elif i/64 == 1:
                    rgb = 1
                else:
                    rgb = 2
                temp[rgb,i/4:i/4+2,i%8*2:i%8*2+2] = 0.25
                W_large[:,:,i,:] = temp.reshape((1,3*16*16))
        elif 'conv' in pr_large:
            ind = layers_large.index(pr_large)
            prev_layers = layers_large[ind-1]
            dim1_tiny, dim2_tiny, dim3_tiny, dim4_tiny = params_tiny[prev_layers][0].shape
            medium = (dim3_large-1)/2
            for i in range(dim1_tiny):
                W_large[i,i,medium,medium] = 1         
        else:
            sys.exit('Error: The {} layer is unknown!'.format(pr_large))
    # save new weights        
    params_large[pr_large][0][...] = W_large
    print 'The transformation of {} layer is done'.format(pr_large)

# save new caffemodel file
net_large.save(caffe_root + file_large_model)
