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

REGION_TYPE = 'FACE'

# caffe layers
caffe_root = '/users/vijay.kumar/caffe/'
sys.path.insert(0, caffe_root + 'python')

import caffe
from caffe import layers as L

# enable gpu
caffe.set_mode_gpu()

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': (1,3,224,224)})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
#transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR # not required..

proto_file = '/users/vijay.kumar/code/person_recognition/models/new/vgg_face_caffe/VGG_FACE_deploy.prototxt'
weights = '/users/vijay.kumar/code/person_recognition/models/new/vgg_face_caffe/VGG_FACE.caffemodel'    
net = caffe.Net(proto_file, weights, caffe.TEST)
net.forward()


import os
labeled_actors = [1, 11, 12, 133, 134, 135, 139, 140, 141 , 
                   150,  18,  20,  21,  22,  23,  238,  25,  26,  27,  28,
                  3,  34,  35,  39,  4,  40]

imdb_imgs_dir = '/data6/vijay.kumar/DATA/HANNAH/imdb/images/'
imdb_annot_dir = '/data6/vijay.kumar/DATA/HANNAH/imdb/annot/'


# In[ ]:

def get_region_imdb(image, box, region_type):
    
    region = None
    box = box.astype(int)
    if region_type == 'FACE':
        box[1] = box[1] + 0.3*box[3]
        box[3] = 0.6*box[3]
        box[0] = box[0] + 0.1*box[2]
        box[2] = 0.9*box[2]
        
        region = image[max(1,box[1]):min(box[1]+box[3], image.shape[0]), 
                    max(1,box[0]):min(box[0]+box[2], image.shape[1]),:]                
         
    return region


def get_region_hannah(image, fbox, region_type):
    
    region = None
    fbox = fbox.astype(int)        
    if region_type == 'FACE':
        region = image[max(1,fbox[1]):min(fbox[1]+fbox[3], image.shape[0]), 
                    max(1,fbox[0]):min(fbox[0]+fbox[2], image.shape[1]),:]            
        
    return region


num_examples = 0
for la in labeled_actors:
    annotations = np.genfromtxt(imdb_annot_dir + str(la) + '.txt', dtype = float, delimiter=',')

    if len(annotations.shape)==1:
        annotations = annotations.reshape((1,5))    
    num_examples = num_examples + len(annotations)

'''
train_feats_initial = np.zeros((num_examples, 4096))
train_labels_initial = np.zeros((num_examples,1))

count = 0
for la in labeled_actors:
    print 'Extracting features for actor:', la
    annotations = loadtxt(imdb_annot_dir + str(la) + '.txt', dtype = float, delimiter=',')
    
    if len(annotations.shape) == 1:
        annotations = annotations.reshape((1,5))
        
    for annot_sample in annotations:        
        
        img_name = imdb_imgs_dir + str(la) + '/' + str(int(annot_sample[0])) + '.jpg'        
        image = cv2.imread(img_name)     ## NOTE USING OPENCV TO READ.. # BGR Format
        hbox = annot_sample[1:] 

        region = get_region_imdb(image, hbox, REGION_TYPE)
        
        #plt.figure(1);plt.imshow(region);plt.show();plt.close()        
        #image_save_path = str(int(count)) + '.png'
        #scipy.misc.imsave(image_save_path, region)

        if region.shape[0] < 10 or region.shape[1] < 10:            
            continue
            
        transformed_image = transformer.preprocess('data', region)                
        net.blobs['data'].data[0, ...] = transformed_image
        net.forward(start='conv1_1')
        feat = net.blobs['fc7'].data
        
        transformed_image = transformer.preprocess('data', cv2.flip(region,1))                                               
        net.blobs['data'].data[0, ...] = transformed_image
        net.forward(start='conv1_1')
        feat2 = net.blobs['fc7'].data
        
        train_feats_initial[count] = 0.5*(feat+feat2)
        train_labels_initial[count] = la
        count = count + 1  

train_feats = train_feats_initial[0:count]
train_labels = train_labels_initial[0:count]
'''

# In[ ]:

faces = loadtxt('/data6/vijay.kumar/DATA/HANNAH/annotations/hannah_video_faces_.txt')
track_char_map = loadtxt('/data6/vijay.kumar/DATA/HANNAH/annotations/hannah_video_tracks.txt', usecols=(0,1), skiprows=2)
track_char_dict_map = {}
for i in range(track_char_map.shape[0]):
    track_char_dict_map[track_char_map[i,0]] = track_char_map[i,1]   


# In[ ]:
#sys.path.insert(0, '/users/vijay.kumar/tools/liblinear-2.1/python')
#from liblinearutil import *
#model = train(train_labels[:,0].tolist(), train_feats.tolist(), '-s 1 -c 1')
#np.savetxt('imdb_feas.txt',train_feats)
#np.savetxt('imdb_labels.txt',train_labels)


# In[ ]:

video_names = ['/data6/vijay.kumar/DATA/HANNAH/hannah-001_1_37784.vob', 
	       '/data6/vijay.kumar/DATA/HANNAH/hannah-002_37785_75444.vob', 
               '/data6/vijay.kumar/DATA/HANNAH/hannah-003_75445_115473.vob', 
               '/data6/vijay.kumar/DATA/HANNAH/hannah-004_115474_153475.vob']

video_frame_split = np.zeros((5,))
for i in range(4):
    cap = cv2.VideoCapture(video_names[i])
    video_frame_split[i+1] = video_frame_split[i] + cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    cap.release()
print video_frame_split
print "total_frames:", np.sum(video_frame_split)


# In[ ]:

def get_frame_no(fno, vf_split, vid_objs):

    for i in range(4):
        if fno > vf_split[i] and fno <= vf_split[i+1]:
            fno = fno - vf_split[i]
            working_cap = vid_objs[i]
    return working_cap, fno

video_objs = {} 
for i in range(4):
    video_objs[i] = cv2.VideoCapture(video_names[i])    
    success,image = video_objs[i].read()    
    
frame_interval = 10
track_ids = faces[:,5]
num_tracks = track_ids.shape[0]
unique_track_ids = np.unique(track_ids)
st_tr_id = 0 #int(sys.argv[1])
en_tr_id = len(unique_track_ids) #int(sys.argv[2])

test_feats = []
test_labels = []
test_tracks = []
for i in range(st_tr_id, en_tr_id):       
        tid = unique_track_ids[i]
        print 'Extracting track:',tid
        track_data = faces[np.where(track_ids==tid)]
        track_data = track_data[0::frame_interval,:]
        frame_nos = track_data[:,0]                    
        num_frames = len(frame_nos)
        for j in range(num_frames):             
            cap, fno = get_frame_no(frame_nos[j], video_frame_split, video_objs)
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, fno-3)
            fbox = track_data[j,1:5]             
            success, image = cap.read()            
            if success:
                image = cv2.resize(image,(996, 560))
                #image_save_path = '/data6/vijay.kumar/DATA/HANNAH/frames/' + str(int(fno)) + '.png'
                #scipy.misc.imsave(image_save_path, image)
                                                
                region = get_region_hannah(image, fbox, REGION_TYPE)                                
                if region is None:
                    continue
                
                #image_save_path = str(int(tid)) + '.png'
                #scipy.misc.imsave(image_save_path, region)                              
                #plt.figure(1);imshow(image);plt.show();plt.close()
                #plt.figure(2);imshow(region);plt.show();plt.close()
               
                transformed_image = transformer.preprocess('data', region)                
                net.blobs['data'].data[0, ...] = transformed_image
                net.forward(start='conv1_1')
                feat = net.blobs['fc7'].data
                
                transformed_image = transformer.preprocess('data', cv2.flip(region,1))                
                net.blobs['data'].data[0, ...] = transformed_image
                net.forward(start='conv1_1')
                feat2 = net.blobs['fc7'].data
                
                feat = 0.5*(feat + feat2)                
                test_feats.append(feat)                
                test_labels.append(track_char_dict_map[tid])                
                test_tracks.append(tid)
        
for i in range(4):            
    video_objs[i].release()

test_feats_np = np.squeeze(np.array(test_feats))
test_labels_np = np.array(test_labels)
test_tracks_np = np.array(test_tracks)

print test_feats_np.shape
print test_tracks_np.shape
    
np.savetxt('hannah_labels.txt',test_labels_np)
np.savetxt('hannah_tracks.txt',test_tracks_np)
np.savetxt('hannah_feas.txt',test_feats_np)
