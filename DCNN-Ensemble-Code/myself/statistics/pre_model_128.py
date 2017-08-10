def pre_model_128(MODEL_FILE,PRETRAINED,MEAN_FILE):
	

	import numpy as np
	import matplotlib.pyplot as plt
	import os

	caffe_root = '/home/shiyan/caffe-five-cost-sensitive/'
	import sys
	sys.path.append('/home/shiyan/caffe-five-cost-sensitive/python')
	import caffe

        print MODEL_FILE
        print PRETRAINED
        print MEAN_FILE

	# Open mean.binaryproto file
	
	blob = caffe.proto.caffe_pb2.BlobProto()
	data = open(MEAN_FILE , 'rb').read()
	blob.ParseFromString(data)
	mean_arr = caffe.io.blobproto_to_array(blob)

	mean_test1 = mean_arr[0].mean(1).mean(1)
	mean_zero=[0,0,0]
	mean_tmp=np.array(mean_zero)
	mean_tmp[0]=int(round(mean_test1[0]))
	mean_tmp[1]=int(round(mean_test1[1]))
	mean_tmp[2]=int(round(mean_test1[2]))
	
	# Initialize NN
	# Initialize NN
	net = caffe.Classifier(MODEL_FILE, PRETRAINED,


						   image_dims=(128,128),

						   #mean=np.load(caffe_root + 'meanfile/mnist_train_lmdb_mean.npy').mean(1).mean(1),
						
						   #mean = np.array([93,85,79]),
						   #mean= mean_arr[0].mean(1).mean(1),
						   #mean=np.load(caffe_root + 'myself/python_okornotok/okornotok_result/mnist_train_lmdb_mean.npy').mean(1).mean(1),

						   #input_scale=255,


						   mean = mean_tmp,
						   raw_scale=255,
						   channel_swap=(2,1,0)
							)
	return net
