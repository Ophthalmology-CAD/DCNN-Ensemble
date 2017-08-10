## Usage Instructions for Auto-localization-lens:
* Clone this repository somewhere, let's refer to it as $ROOT
```
git clone https://github.com/Ophthalmology-CAD/DCNN-Ensemble.git
```
* For Auto-localization-lens, the "cut.m" is the startup file and could be executed in MATLAB. 
    <br /> 
    <br />
    <br />

## Usage Instructions for DCNN-Ensemble-Code:
### Train
* Compile the caffe and pycaffe.
```
cd $ROOT/DCNN-Ensemble-Code
make all 
make test 
make runtest 
make pycaffe
```
* Download the pre-trained. 
**alexnet model:（https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet）
**googlenet model:（https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet）
**ResNet-50 model:（https://github.com/KaimingHe/deep-residual-networks#models）
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


