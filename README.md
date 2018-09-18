# CE2P

This respository includes a PyTorch implementation of [CE2P]() that won the 1st places of single human parsing in the 2nd LIP Challenge.  

The code is based upon [https://github.com/speedinghzl/Pytorch-Deeplab](https://github.com/speedinghzl/Pytorch-Deeplab)

### Requirements

To install PyTorch, please refer to https://github.com/pytorch/pytorch#installation.

### Compiling

Some parts of InPlace-ABN have a native CUDA implementation, which must be compiled with the following commands:
```bash
cd modules
sh build.sh
python build.py
``` 
The `build.sh` script assumes that the `nvcc` compiler is available in the current system search path.
The CUDA kernels are compiled for `sm_50`, `sm_52` and `sm_61` by default.
To change this (_e.g._ if you are using a Kepler GPU), please edit the `CUDA_GENCODE` variable in `build.sh`.

### Dataset and pretrained model

Plesae download LIP dataset and modify the `DATA_DIRECTORY` in job_loacl.sh. We also provide the label files of [edges]() which are generated upon the parsing labels.
 
Please download imagenet pretrained [resent-101](https://pan.baidu.com/s/1YMiL0lFgpzhIfD_IjwSJjw), and put it into dataset folder.

### Training and Evaluation
```bash
./job_local.sh
./job_evaluation.sh
``` 

