#!/usr/bin/env sh

TOOLS=./build/tools


for i in 1 2 4 6 8

	do
	sed -i 's/caffe_alexnet.*/caffe_alexnet_train_'$i'"/g' ./myself/slitlamp-fine-alex/solver.prototxt
	sed -i 's/pos_mult:.*/pos_mult: '$i'/g' ./myself/slitlamp-fine-alex/train_val.prototxt
	GLOG_logtostderr=1 $TOOLS/caffe train --solver=./myself/slitlamp-fine-alex/solver.prototxt --weights ./myself/slitlamp-fine-alex/bvlc_alexnet.caffemodel -gpu 0,1,2,3
	echo "Done."
	done


