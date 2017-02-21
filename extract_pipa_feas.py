
# coding: utf-8

# In[1]:

import cv2
import numpy as np
from numpy import loadtxt
import sys

#from pylab import *
#%pylab inline
#pylab.rcParams['figure.figsize'] = (50, 10)
#np.set_printoptions(precision = 17)

#%matplotlib inline
#import matplotlib.pyplot as plt
import scipy


# In[2]:

import PIL.Image
from cStringIO import StringIO
import IPython.display

def showarray(a, fmt='png'):
    a = np.uint8(a)
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    IPython.display.display(IPython.display.Image(data=f.getvalue()))

# In[3]:

# caffe layers
caffe_root = '/users/vijay.kumar/caffe/'
sys.path.insert(0, caffe_root + 'python')

import caffe
from caffe import layers as L

# enable gpu
caffe.set_mode_gpu()


# In[4]:

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': (1,3,224,224)})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
#transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR # not required..


# In[5]:

proto_file = '/users/vijay.kumar/code/person_recognition/models/new/vgg_face_caffe/VGG_FACE_deploy.prototxt'
weights = '/users/vijay.kumar/code/person_recognition/models/new/vgg_face_caffe/VGG_FACE.caffemodel'
net = caffe.Net(proto_file, weights, caffe.TEST)
net.forward()
avgImg = [129.1863,104.7624,93.5940]


# In[24]:

album_gt_path = '/data6/vijay.kumar/person_recognition/annotations/index.txt'
image_dir = '/data6/vijay.kumar/person_recognition/data/'
count = 0
num_examples = sum(1 for line in open(album_gt_path))

split_dir = {1:'train',2:'val',3:'test'}
features = np.zeros((num_examples, 4096))
labels = np.zeros((num_examples, 1))
with open(album_gt_path) as f:
    for line in f:        
        line_split = line.strip().split(' ')        
        img_name = line_split[0] + '_' + line_split[1] + '.jpg'
        split_no = int(line_split[7])
        if  split_no == 0:
            continue
        
        img_name = image_dir + split_dir[split_no] + '/' + img_name                
        label = line_split[6]
        image = cv2.imread(img_name)     ## NOTE USING OPENCV TO READ.. # BGR Format
        #showarray(image)                
        x1 = max(0, int(line_split[2])-1)
        y1 = max(0, int(line_split[3])-1)
        w = int(line_split[4])
        h = int(line_split[5])
         
        if len(image.shape) > 2:
        	face = image[y1:min(y1+h,image.shape[0]), x1:min(x1+w, image.shape[1]),:]                                 
	else:
                temp = image[y1:min(y1+h,image.shape[0]), x1:min(x1+w, image.shape[1])]
                face = np.zeros((temp.shape[0], temp.shape[1],3))
                face[:,:,0] = temp
                face[:,:,1] = temp
                face[:,:,2] = temp
        
        if face.shape[0] == 0 or face.shape[1]==0:
             continue
                
        #face[:,:,0] = face[:,:,0] - avgImg[2]
        #face[:,:,1] = face[:,:,1] - avgImg[1]
        #face[:,:,2] = face[:,:,2] - avgImg[0]
                                              
        transformed_image = transformer.preprocess('data', face)                
        net.blobs['data'].data[0, ...] = transformed_image
        net.forward(start='conv1_1')
        features[count] = net.blobs['fc7'].data
        labels[count] = int(label)
        count = count + 1        
        
        if count%100 == 0:
            print count
        
np.savetxt('/home/vijay.kumar/semi-supervised/pipa_feas2',features)
np.savetxt('/home/vijay.kumar/semi-supervised/pipa_labels2',labels)

# In[ ]:



