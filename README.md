# CE2P

This respository includes a PyTorch implementation of [CE2P](https://arxiv.org/abs/1809.05996) that won the 1st places of single human parsing in the 2nd LIP Challenge.  

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
Note that the left and right label should be swapped when the label file is flipped. 

Plesae download LIP dataset and modify the `DATA_DIRECTORY` in job_train.sh. 
 
Please download imagenet pretrained resent-101, label files of edge and the trained models from [baidu drive](https://pan.baidu.com/s/15Fxrqe-kF4-tNuh3gka2DQ), and put it into dataset folder.

### Training and Evaluation
```bash
./job_train.sh
./job_evaluation.sh
``` 
If this code is helpful for your research, please cite the following paper:

    @article{LiuCE2P,
      title={Devil in the Details: Towards Accurate Single and Multiple Human Parsing},
      author={Ting Liu, Tao Ruan, Zilong Huang, Yunchao Wei, Shikui Wei, Yao Zhao, Thomas Huang},
      journal={arXiv:1809.05996},
      year={2018}
    }
