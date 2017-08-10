#!/usr/bin/env sh

TOOLS=./build/tools

for i in 1 2 4 6 8
	
	do
	sed -i 's/caffe_googlenet.*/caffe_googlenet_train_'$i'"/g' ./myself/slitlamp-fine-googlenet/solver.prototxt
	sed -i 's/pos_mult:.*/pos_mult: '$i'/g' ./myself/slitlamp-fine-googlenet/train_val.prototxt
	GLOG_logtostderr=1 $TOOLS/caffe train --solver=./myself/slitlamp-fine-googlenet/solver.prototxt --weights ./myself/slitlamp-fine-googlenet/bvlc_googlenet.caffemodel -gpu 0,1,2,3  
	echo "Done."
	done



