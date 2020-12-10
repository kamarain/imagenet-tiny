#!/usr/bin/env python
"""
compareData.py
It is used to print data in blobs.
-- liuyuan
"""

import numpy as np

caffe_root = '../'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

caffe.set_mode_cpu()
net = caffe.Classifier(caffe_root + 'path/to/network/deploy/file/deploy-net-256.prototxt',
                       caffe_root + 'path/to/network/model/file/trans_128_256.caffemodel')
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
net.set_mean('data', np.load(caffe_root + 'path/to/mean/file/ImageNet-tiny-256x256-sRGB_mean.npy'))  # mean
net.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
net.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

scores = net.predict([caffe.io.load_image(caffe_root + 'path/to/image/cat256.jpg')])


# create a file to save all parameters
f = open(caffe_root + 'path/to/save/data/file/datafile256.txt', 'w')

# write data in different layers to the file
f.write('data: \n')
for k, v in net.blobs.items():
    f.write(k) # write layer's name
    f.write(str(v.data.shape)) # write weight's dimension
    f.write('\n')
    dim1, dim2, dim3, dim4 = v.data.shape
    data = v.data
    for j in range(dim2): # write data
        for m in range(dim3):
            for n in range(dim4):
                f.write(str(data[0,j,m,n]))
                f.write(', ')
            f.write('\n')
        f.write('\n')
    f.write('\n')

# close the file
f.close()

# print parameters' dimension
#print [(k, v[0].data.shape) for k, v in net.params.items()] # weights
#print [(k, v[1].data.shape) for k, v in net.params.items()] # bias



