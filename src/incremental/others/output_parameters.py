#!/usr/bin/env python
"""
params.py
It is used to read parameters from caffe file, and save them (weights and bias) to a txt file or print them in screen.
-- liuyuan
"""

import numpy as np

caffe_root = '../'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# load files
net = caffe.Classifier( caffe_root + 'path/to/network/deploy/file/deploy-net-256.prototxt',
                        caffe_root + 'path/to/network/model/file/trans_128_256.caffemodel')

# create a file to save all parameters
f = open(caffe_root + 'path/to/save/parameters/file/paramsfile256.txt', 'w')


# write weights in different layers to the file
f.write('weights: \n')
for k, v in net.params.items():
    f.write(k) # write layer's name
    f.write(str(v[0].data.shape)) # write weight's dimension
    f.write('\n')
    dim1, dim2, dim3, dim4 = v[0].data.shape
    params = v[0].data
    for i in range(dim1): # write weights
        f.write(str(i))
        f.write('\n')
        for j in range(dim2):
            for m in range(dim3):
                for n in range(dim4):
                   f.write(str(params[i,j,m,n]))
                   f.write(', ')
                f.write('\n')
            f.write('\n')
        f.write('\n')
    f.write('\n')

# write bias in different layers to the file
f.write('bias: \n')
for k, v in net.params.items():
    f.write(k) # write layer's name
    f.write(str(v[1].data.shape)) # write weight's dimension
    dim1, dim2, dim3, dim4 = v[1].data.shape
    params = v[1].data
    for i in range(dim1): # write weights
        for j in range(dim2):
            for m in range(dim3):
                for n in range(dim4):
                    f.write(str(params[i,j,m,n]))
                    f.write(', ')
                f.write('\n')
            f.write('\n')
        f.write('\n')
    f.write('\n')
    
# close the file
f.close()

#print blobs' dimension
print [(k, v.data.shape) for k, v in net.blobs.items()]
# print parameters' dimension
print [(k, v[0].data.shape) for k, v in net.params.items()] # weights
print [(k, v[1].data.shape) for k, v in net.params.items()] # bias


