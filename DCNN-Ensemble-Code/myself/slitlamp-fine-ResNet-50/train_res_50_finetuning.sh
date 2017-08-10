#!/usr/bin/env sh

TOOLS=./build/tools


for i in 1 2 4 6 8
	do
	sed -i 's/caffe_resnet50.*/caffe_resnet50_train_'$i'"/g' ./myself/slitlamp-fine-ResNet-50/resnet_50_solver.prototxt
	sed -i 's/pos_mult:.*/pos_mult: '$i'/g' ./myself/slitlamp-fine-ResNet-50/resnet_50.prototxt
	GLOG_logtostderr=1 $TOOLS/caffe train --solver=./myself/slitlamp-fine-ResNet-50/resnet_50_solver.prototxt --weights ./myself/slitlamp-fine-ResNet-50/ResNet-50-model.caffemodel -gpu 0,1,2,3
	echo "Done."
	done


