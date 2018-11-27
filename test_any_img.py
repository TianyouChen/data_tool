#encoding=utf8
'''
Detection with SSD
In this example, we will load a SSD model and use it to detect objects.
'''
import glob
import os
import sys
import argparse
import cv2
import numpy as np
import time
from PIL import Image, ImageDraw
# Make sure that caffe is on the python path:
caffe_root = '/home/ubuntu/caffe_ssd'
os.chdir(caffe_root) #切换到caffe_root目录
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe

from google.protobuf import text_format
from caffe.proto import caffe_pb2

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

class CaffeDetection:
    #def __init__(self, gpu_id, model_def, model_weights, image_resize, labelmap_file):
    def __init__(self, model_def, model_weights, image_resize, labelmap_file):
        #caffe.set_device(gpu_id)
        caffe.set_mode_cpu()

        self.image_resize = image_resize
        # Load the net in the test phase for inference, and configure input preprocessing.
        self.net = caffe.Net(model_def,      # defines the structure of the model
                             model_weights,  # contains the trained weights
                             caffe.TEST)     # use test mode (e.g., don't perform dropout)
         # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array([104, 117, 123])) # mean pixel
        # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_raw_scale('data', 255)
        # the reference model has channels in BGR order instead of RGB
        self.transformer.set_channel_swap('data', (2, 1, 0))

        # load PASCAL VOC labels
        file = open(labelmap_file, 'r')
        self.labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), self.labelmap)

    def detect(self, image_file, conf_thresh=0.5, topn=100):
        '''
        SSD detection
        '''
        # set net to batch size of 1
        # image_resize = 300
        self.net.blobs['data'].reshape(1, 3, self.image_resize, self.image_resize)
        image = caffe.io.load_image(image_file)

        #Run the net and examine the top_k results
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image

        # Forward pass.
        T1 =time.time()
        #T1 =time.clock()
        detections = self.net.forward()['detection_out']
        T2 = time.time()
        print(T2-T1)
        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = get_labelname(self.labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        print 'top coordinate float'
        print top_xmax
        print top_ymax

        result = []
        for i in xrange(min(topn, top_conf.shape[0])):
            xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = top_labels[i]
            result.append([xmin, ymin, xmax, ymax, label, score, label_name])
        return result

def main(args):
    detection = CaffeDetection(args.model_def, args.model_weights,
                               args.image_resize, args.labelmap_file)
    

    dir_list = os.listdir(args.test_image_dir)
    for i in range(0,len(dir_list)):
        path = os.path.join(args.test_image_dir,dir_list[i])
        file_list = os.listdir(path)
        
        
        for j in range(0,len(file_list)):
            img_dir = os.path.join(path,file_list[j])
            print(file_list[j])
    #img_list = os.listdir(args.test_image_dir)
    #for im_file1 in img_list:
        #print(im_file1) 
        #im_file = '/home/ubuntu/Save/'+im_file1  
        #print (im_file)
            result = detection.detect(img_dir, args.thresh)
            image =cv2.imread(img_dir)
            #images =cv2.imread(img_dir,0)#read huidu tu
            #re = result[0]
            #xmin = re[0] 
        
            #ymin = re[1]
            #xmax = re[2]
            #ymax = re[3]
            
            img = image[result[0][0]-30:result[0][2], result[0][1]+80:result[0][3]+10]
            #cv2.rectangle(image,(xmin+20, ymin),(xmax + 15,image.shape[0]-2),(0,0,255))

            cv2.imwrite('/home/ubuntu/Save1/'+file_list[j],img)
            #print 'saving result'
            #print (re)
            #print 'img size' 
            #print image.shape

            #cv2.rectangle(images,(1,1),(5,5),(0,0,255))
            #ccc = cv2.resize(img,(48,48),interpolation=cv2.INTER_CUBIC)
            #cv2.imwrite('/home/ubuntu/Save1/'+file_list[j],img)
                   

    print('Done!')



def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    #parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--labelmap_file',
                        default='/home/ubuntu/labelmap_face.prototxt')
    parser.add_argument('--model_def',
                        default='deploy-ResNet-v1-384.prototxt')
    parser.add_argument('--image_resize', default=384, type=int)
    parser.add_argument('--thresh', default=0.25, type=float)
    parser.add_argument('--model_weights',
                        default='resnet-384_iter_210000.caffemodel')
    parser.add_argument('--test_image_dir', default='/home/ubuntu/Save_img/')
    #parser.add_argument('--result_dir', default='/home/ubuntu/caffe_ssd/data/fddb/)
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())
