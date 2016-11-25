
# coding: utf-8

# In[11]:

import numpy as np
import matplotlib.pyplot as plt
import math
# get_ipython().magic(u'matplotlib inline')
import scipy.io
import h5py,numpy as np


# Make sure that caffe is on the python path:
caffe_root = '/users/gpu/agjayant1/caffe-PersonReID/'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import fileinput

cur_path = '/users/gpu/agjayant1/Image_Tagging/tagnet/'
cur2_path = '/users/gpu/agjayant1/IA12_example/'

train_test_path = cur2_path + 'l2_normalized_semantic_SVM_full_data_with_val_291labels_no_zero.mat'
train_test_data_package = h5py.File(train_test_path,'r')

testing_data_target=train_test_data_package.get('prepared_testing_label')
testing_data_target=np.array(testing_data_target).transpose().astype("float32")

normalized_test_data_source=train_test_data_package.get('prepared_testing_data')
normalized_test_data_source=np.array(normalized_test_data_source).transpose().astype("float32")

import os


# In[6]:

# print testing_data_target[1961]
# print normalized_test_data_source[1961]
# print len(normalized_test_data_source)


# In[3]:

caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net( cur_path + 'deploy_tagnet.prototxt',
                 cur_path + 'models3/_iter_50000.caffemodel',
                caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
#transformer.set_transpose('data', (2,0,1))
#transformer.set_mean('data', np.load(caffe_root + 'rank_scripts/query_128x128_market.npy').mean(1).mean(1)) # mean pixel
#transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
#transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB


# In[22]:

num_images = len(normalized_test_data_source)
BatchSize = 100
j=0
images_tags = {}
while j < num_images:
    net = caffe.Net( cur_path + 'deploy_tagnet.prototxt',
                 cur_path + 'models3/_iter_50000.caffemodel',
                caffe.TEST)


    # set net to batch size
    net.blobs['data'].reshape(BatchSize,4096)
    i = 0
    k = j
    while j < num_images and i < BatchSize:
        query_image = normalized_test_data_source[j]
        net.blobs['data'].data[i] = transformer.preprocess('data', query_image)
        i = i + 1
        j = j + 1

    out = net.forward()
    i=0
    while k < num_images and i < BatchSize:
        a=out['fc4'][i]
        images_tags[k]=sorted(range(len(a)), key=lambda i: a[i], reverse=True)[:10]
        i = i + 1
        k = k + 1


# In[8]:

# check = images_tags[150]
# for i in range(291):
#     if check[i] > -3.5:
#         print i


# In[16]:

# a = images_tags[150]
# print a


# In[10]:

# check2 = testing_data_target[150]
# for i in range(291):
#     if check2[i] >0:
#         print i


# In[23]:

i = 0
F1 = 0
while i < 291:
    j=0
    a = 0 # correct Predictions
    b = 0 # Number of Images Having Tag i
    c = 0 # Total Predictions
    while j < len(images_tags):
        if i in images_tags[j]:
            c += 1
            if testing_data_target[j][i] > 0:
                a += 1
        if testing_data_target[j][i] > 0:
            b += 1
        j+=1
    i+=1

    if a > 0:
        precision = a*1.0/c
    else:
        precision = 0

    #if b > 0:
    recall = a*1.0/b
    #else:
    #    recall = 0

    if precision + recall > 0 :
        f1 = 2*precision*recall/(precision+recall)
    else:
        f1 = 0
    F1 += f1
    print "Tag: ", i," ,precision: ",precision," ,recall" , recall," ,f1: ", f1

F1 = F1/291
print "Final F1: ", F1

