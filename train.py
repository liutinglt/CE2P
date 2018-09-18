import argparse 
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
import os.path as osp
import math 
import random
import timeit
import logging
from tensorboardX import SummaryWriter
from models import Res_CE2P
from dataset.datasets import LIPParsingEdgeDataSet
from utils.utils import decode_labels, inv_preprocess, decode_predictions
from utils.criterion import CriterionCrossEntropyEdgeParsing 
from utils.encoding import SelfDataParallel, ModelDataParallel, CriterionDataParallel
from numpy.distutils.tests.test_exec_command import emulate_nonposix
 
start = timeit.default_timer()

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

BATCH_SIZE = 40
DATA_DIRECTORY = '/home/amax/LIP/LIP_data_model/data/resize473/split'
DATA_LIST_PATH = './dataset/list/lip/trainList_split.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '473, 473'
LEARNING_RATE = 7e-4
MOMENTUM = 0.9
NUM_CLASSES = 20
NUM_STEPS = 300000
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = './dataset/MS_DeepLab_resnet_pretrained_init.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 1000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005
GPU_DEVICES = '0,1,2,3,4' 

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--start-iters", type=int, default=0,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=str, default=GPU_DEVICES,
                        help="choose gpu device.")   
    return parser.parse_args()

args = get_arguments()
 
          
def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))
 
def adjust_learning_rate(optimizer, i_iter): 
     
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)     
        
    optimizer.param_groups[0]['lr'] = lr
    return lr

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def set_bn_momentum(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1 or classname.find('InPlaceABN') != -1:
        m.momentum = 0.0003

def main():
    """Create the model and start the training."""
    writer = SummaryWriter(args.snapshot_dir)
    
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    cudnn.enabled = True
 
    deeplab = Res_CE2P(num_classes=args.num_classes) 

    saved_state_dict = torch.load(args.restore_from)
    new_params = deeplab.state_dict().copy()
  
    for i in saved_state_dict: 
        i_parts = i.split('.') 
        if not i_parts[1]=='layer5':
            new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
    if args.start_iters > 0:
        deeplab.load_state_dict(saved_state_dict)
    else:
        deeplab.load_state_dict(new_params)
    print(deeplab) 
    model = ModelDataParallel(deeplab) 
    model.train()
    model.float() 
    model.cuda()    
 
    criterion = CriterionCrossEntropyEdgeParsing()
    criterion = CriterionDataParallel(criterion)
    criterion.cuda()
    
    cudnn.benchmark = True

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)


    trainloader = data.DataLoader(LIPParsingEdgeDataSet(args.data_dir, args.data_list, max_iters=args.num_steps*args.batch_size, crop_size=input_size, 
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN), 
                    batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    optimizer = optim.SGD([{'params': filter(lambda p: p.requires_grad, deeplab.parameters()), 'lr': args.learning_rate }], 
                lr=args.learning_rate, momentum=args.momentum,weight_decay=args.weight_decay)
    optimizer.zero_grad()
  
    for i_iter, batch in enumerate(trainloader):
        i_iter += args.start_iters
        images, labels, edges, _, _ = batch
        images = Variable(images.cuda())
        labels = Variable(labels.long().cuda())
        edges =  Variable(edges.long().cuda())

        optimizer.zero_grad()
        lr = adjust_learning_rate(optimizer, i_iter)
        preds = model(images) 
        loss = criterion(preds, [labels,edges])
        loss.backward()
        optimizer.step()

        if i_iter % 100 == 0:
            writer.add_scalar('learning_rate', lr, i_iter)
            writer.add_scalar('loss', loss.data.cpu().numpy(), i_iter)

        if i_iter % args.save_pred_every == 0:
            images_inv = inv_preprocess(images, args.save_num_images, IMG_MEAN)
            labels_colors = decode_labels(labels, args.save_num_images, args.num_classes)
            edges_colors =  decode_labels(edges, args.save_num_images, 2) 
            if isinstance(preds, list):
                preds = preds[0] 
            preds_colors = decode_predictions(preds[1], args.save_num_images, args.num_classes)
            pred_edges_colors = decode_predictions(preds[2], args.save_num_images, args.num_classes)
            for index, (img, lab) in enumerate(zip(images_inv, labels_colors)):
                writer.add_image('Images/'+str(index), img, i_iter)
                writer.add_image('Labels/'+str(index), lab, i_iter)
                writer.add_image('preds/'+str(index), preds_colors[index], i_iter)
                writer.add_image('edges/'+str(index), edges_colors[index], i_iter)
                writer.add_image('pred_edges/'+str(index), pred_edges_colors[index], i_iter) 
              
        print('iter = {} of {} completed, loss = {}, learning_rate = {:e}'.format(i_iter, args.num_steps, loss.data.cpu().numpy(), lr))

        if i_iter >= args.num_steps-1:
            print('save model ...')
            torch.save(deeplab.state_dict(),osp.join(args.snapshot_dir, 'LIP_'+str(args.num_steps)+'.pth'))                                                                         
            break

        if i_iter % args.save_pred_every == 0:
            print('taking snapshot ...')
            torch.save(deeplab.state_dict(),osp.join(args.snapshot_dir, 'LIP_'+str(i_iter)+'.pth'))     

    end = timeit.default_timer()
    print(end-start,'seconds')

if __name__ == '__main__':
    main()
