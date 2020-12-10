#!/usr/bin/env python
"""
npy.py
It is used to transform binartproto file to npy file.
--liuyuan
"""

import numpy as np

caffe_root = '../'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

blob = caffe.proto.caffe_pb2.BlobProto()
data = open('../../' + caffe_root + 'path/to/mean/file/ImageNet-tiny-256x256-sRGB_mean.binaryproto', 'rb').read()
blob.ParseFromString(data)
arr = np.array(caffe.io.blobproto_to_array(blob))
out = arr[0]
np.save(caffe_root + 'path/to/save/new/mean/file/ImageNet-tiny-256x256-sRGB_mean.npy', out)

