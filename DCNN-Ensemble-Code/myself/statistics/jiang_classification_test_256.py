# -*-  encoding:utf-8  -*-
def jiang_classification_test_256(net,recurrence_file,surgery_file,image,model,fenlei):
	import numpy as np
	import matplotlib.pyplot as plt
	import os

	caffe_root = '/home/shiyan/caffe-five-cost-sensitive/'
	import sys
	sys.path.append('/home/shiyan/caffe-five-cost-sensitive/python')
	import caffe


	test_other=caffe_root+'myself/external_test_data/web-256/'+recurrence_file
    	#test_normal=caffe_root+'myself/twoclass_test_data/'+normal_file
	test_surgery=caffe_root+'myself/external_test_data/web-256/'+surgery_file

	'''
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


						   #image_dims=(256,256),

						   #mean=np.load(caffe_root + 'meanfile/mnist_train_lmdb_mean.npy').mean(1).mean(1),
						
						   #mean = np.array([93,85,79]),
						   #mean= mean_arr[0].mean(1).mean(1),
						   #mean=np.load(caffe_root + 'myself/python_okornotok/okornotok_result/mnist_train_lmdb_mean.npy').mean(1).mean(1),

						   #input_scale=255,


						   mean = mean_tmp,
						   raw_scale=255,
						   channel_swap=(2,1,0)
							)
	'''
	net.blobs['data'].reshape(1,3,224,224)
	
#	outfile='static/display/test.txt'

	sum_surgery=0
	error_surgery_number=0
	list_surgery0=[]
	list_surgery1=[]
	list_surgery2=[]
	surgery = []
	for root, dirs, files in os.walk(test_surgery):
		#	output=open(outfile,'w')
		for file in files:
			#print file
			IMAGE_FILE = os.path.join(root,file)
			prediction = net.predict([caffe.io.load_image(IMAGE_FILE)],oversample=True) #prediction = net.predict([caffe.io.load_image(IMAGE_FILE)],oversample=False)
			#print 'predicted class:',prediction[0].argmax()
			sum_surgery=sum_surgery+1
			surgery.append(prediction[0])
			if prediction[0].argmax() == 0:
				error_surgery_number = error_surgery_number+1
				list_surgery0.append(file+'\t'+'predicted class:'+str(prediction[0].argmax())+'\t'+'predicted:'+str(prediction[0]))
			if prediction[0].argmax() == 1:
				list_surgery1.append(file+'\t'+'predicted class:'+str(prediction[0].argmax())+'\t'+'predicted:'+str(prediction[0]))
	list_surgery2.append('sum_surgery:'+str(sum_surgery)+'\t'+'error_surgery_number:'+str(error_surgery_number)+'\t'+'the surgery accuracy is:' + str(float((sum_surgery)-(error_surgery_number))/float(sum_surgery)))



	sum_other=0
	error_other_number=0
	list_other0=[]
	list_other1=[]
	list_other2=[]
	other = []
	for root,dirs,files in os.walk(test_other):
		for file in files:
			#print file
			IMAGE_FILE = os.path.join(root,file)
			prediction = net.predict([caffe.io.load_image(IMAGE_FILE)],oversample=True) #prediction = net.predict([caffe.io.load_image(IMAGE_FILE)],oversample=False)
			#print 'image: ',file
			#print 'predicted class:',prediction[0].argmax()
			sum_other=sum_other+1
			other.append(prediction[0])
			if prediction[0].argmax() == 0:
				list_other0.append(file+'\t'+'predicted class:'+str(prediction[0].argmax())+'\t'+'predicted:'+str(prediction[0]))
			if prediction[0].argmax() == 1:
				error_other_number = error_other_number+1
				list_other1.append(file+'\t'+'predicted class:'+str(prediction[0].argmax())+'\t'+'predicted:'+str(prediction[0]))
	list_other2.append('sum_recurrence:'+str(sum_other)+'\t'+'error_recurrence_number:'+str(error_other_number)+'\t'+'the other accuracy is:' + str(float((sum_other)-(error_other_number))/float(sum_other)))

	accuracy = float((sum_other+sum_surgery)-(error_other_number+error_surgery_number))/float(sum_other+sum_surgery)
	print 'the accuracy is:', accuracy*100
	list_surgery2.append('the total accuracy is:' + str(accuracy))

	outfile = 'web_data_result/'+fenlei+'/googlenet_train_'+model+'.txt'
	file_object = open(outfile, 'w')
	file_object.writelines('ERROR_RECURRENCE\n')
	for i in list_other0:
		file_object.writelines(i+'\n')
	for i in list_other1:
		file_object.writelines(i+'\n')
	for i in list_other2:
		file_object.writelines(i+'\n')
	file_object.writelines('-------------------------------------------------------\n')
	file_object.writelines('ERROR_SUGERY\n')
	for i in list_surgery0:
		file_object.writelines(i+'\n')
	for i in list_surgery1:
		file_object.writelines(i+'\n')
	for i in list_surgery2:
		file_object.writelines(i+'\n')
	file_object.close()

	print other,surgery
	print other[0],surgery[0]
	print other[1],surgery[1]
	return other,surgery

		










