from pre_model_256 import *
from pre_model_128 import *
import os

def pre_net(fenlei,model):
	 #test_file = "static/Predict"

    ###alexnet
    MODEL_FILE = 'model_file/'+fenlei+'/alexnet/deploy.prototxt'
    PRETRAINED = 'model_file/'+fenlei+'/alexnet/caffe_alexnet_train_'+model+'_iter_2000.caffemodel'
    MEAN_FILE = 'model_file/'+fenlei+'/alexnet/mnist_train_lmdb_mean.binaryproto'
    net_alexnet=pre_model_256(MODEL_FILE,PRETRAINED,MEAN_FILE)

    ###googlenet
    MODEL_FILE = 'model_file/'+fenlei+'/googlenet/deploy.prototxt'
    PRETRAINED = 'model_file/'+fenlei+'/googlenet/caffe_googlenet_train_'+model+'_iter_2000.caffemodel'
    MEAN_FILE = 'model_file/'+fenlei+'/googlenet/mnist_train_lmdb_mean.binaryproto'
    net_googlenet=pre_model_256(MODEL_FILE,PRETRAINED,MEAN_FILE)

    ###resnet_50
    MODEL_FILE = 'model_file/'+fenlei+'/resnet_50/deploy.prototxt'
    PRETRAINED = 'model_file/'+fenlei+'/resnet_50/caffe_resnet50_train_'+model+'_iter_2000.caffemodel'
    MEAN_FILE = 'model_file/'+fenlei+'/resnet_50/mnist_train_lmdb_mean.binaryproto'
    net_resnet_50=pre_model_128(MODEL_FILE,PRETRAINED,MEAN_FILE)

    return net_alexnet,net_googlenet,net_resnet_50
