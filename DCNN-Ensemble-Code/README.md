# Usage Instructions:
### Train
* Clone this repository somewhere, let's refer to it as $ROOT
```
git clone https://github.com/Ophthalmology-CAD/DCNN-Ensemble.git
```
* Compile the caffe and pycaffe.
```
cd $ROOT
make all 
make test 
make runtest 
make pycaffe
```
* Run the train_alexnet_finetuning.sh in in the terminal window to train the alexnet model
```
cd $ROOT
sh myself/slitlamp-fine-alex/train_alexnet_finetuning.sh
```
* Run the train_googlenet_finetuning.sh in the terminal window to train the googlenet model
```
cd $ROOT
sh myself/slitlamp-fine-googlenet/train_googlenet_finetuning.sh
```
* Run the train_res_50_finetuning.sh in the terminal window to train the ResNet-50 model
```
cd $ROOT
sh myself/slitlamp-fine-ResNet-50/train_res_50_finetuning.sh
```

### Test

The test code is in $ROOT/myself/statistics

* Run the img_process.py to test: in python terminal. 


