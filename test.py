import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys
import time 

import torch
import cv2
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data
from models import Res_CE2P
from dataset.datasets import LIPDataTestSet
from tensorboardX import SummaryWriter
from collections import OrderedDict
import os
import scipy.ndimage as nd
from math import ceil
from PIL import Image as PILImage

import torch.nn as nn 
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
  

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data-dir", type=str, 
                        help="Path to the directory containing the PASCAL VOC dataset.") 
    parser.add_argument("--data-list", type=str, 
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=255,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=(473,473),
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--num-classes", type=int, default=20,
                        help="Number of classes to predict (including background).") 
    parser.add_argument("--restore-from", type=str, 
                        help="Where restore model parameters from.")
    parser.add_argument("--is-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--save-dir", type=str, 
                        help="Path to the output results.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    return parser.parse_args()

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """

    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def get_lip_palette():  
    palette = [ 0,0,0,
          128,0,0,
          255,0,0,
          0,85,0,
          170,0,51,
          255,85,0,
          0,0,85,
          0,119,221,
          85,85,0,
          0,85,85,
          85,51,0,
          52,86,128,
          0,128,0,
          0,0,255,
          51,170,221,
          0,255,255,
          85,255,170,
          170,255,85,
          255,255,0,
          255,170,0] 
    return palette 
 
def scale_image(image, scale):  
    image = image[0,:,:,:]
    image = image.transpose((1, 2, 0)) 
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation = cv2.INTER_LINEAR) 
    image = image.transpose((2, 0, 1)) 
    return image 
  
def predict(net, image, output_size, is_mirror=True, scales=[1]): 
    if is_mirror:
        image_rev = image[:,:,:,::-1]

    interp = nn.Upsample(size=output_size, mode='bilinear')

    outputs = []
    if is_mirror:
        for scale in scales:
            if scale != 1:
                image_scale = scale_image(image=image, scale=scale)
                image_rev_scale = scale_image(image=image_rev, scale=scale)
            else:
                image_scale = image[0,:,:,:]
                image_rev_scale = image_rev[0,:,:,:]

            image_scale = np.stack((image_scale,image_rev_scale))

            prediction = net(Variable(torch.from_numpy(image_scale), volatile=True).cuda())

            prediction = interp(prediction[1]).cpu().data.numpy()

            prediction_rev = prediction[1,:,:,:].copy()
            prediction_rev[14,:,:] = prediction[1,15,:,:]
            prediction_rev[15,:,:] = prediction[1,14,:,:]
            prediction_rev[16,:,:] = prediction[1,17,:,:]
            prediction_rev[17,:,:] = prediction[1,16,:,:]
            prediction_rev[18,:,:] = prediction[1,19,:,:]
            prediction_rev[19,:,:] = prediction[1,18,:,:]
            prediction_rev = prediction_rev[:,:,::-1]
            prediction = prediction[0,:,:,:]
            prediction = np.mean([prediction, prediction_rev], axis=0)

            outputs.append(prediction)

        outputs = np.mean(outputs, axis=0)
        outputs = outputs.transpose(1,2,0)  
    else:
        for scale in scales:
            if scale != 1:
                image_scale = scale_image(image=image, scale=scale)
            else:
                image_scale = image[0,:,:,:]

            prediction = net(Variable(torch.from_numpy(image_scale).unsqueeze(0), volatile=True).cuda())
            prediction = interp(prediction[1]).cpu().data.numpy()
            outputs.append(prediction[0,:,:,:])

        outputs = np.mean(outputs, axis=0)
        outputs = outputs.transpose(1,2,0)  

    return outputs
  
 
    

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w) 
    
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    model = Res_CE2P(num_classes=args.num_classes)
       
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir) 
        
    palette = get_lip_palette()  
    restore_from  = args.restore_from 
    saved_state_dict = torch.load(restore_from)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda()

    testloader = data.DataLoader(LIPDataTestSet(args.data_dir, args.data_list, crop_size=input_size, mean=IMG_MEAN), 
                                    batch_size=1, shuffle=False, pin_memory=True)
        
    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print('%d images have been proceeded'%(index))
        image,  ori_size, name = batch 
          
        ori_size = ori_size[0].numpy()
         
        output = predict(model, image.numpy(), (np.asscalar(ori_size[0]), np.asscalar(ori_size[1])), is_mirror=args.is_mirror, scales=[1])
        seg_pred = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        output_im = PILImage.fromarray(seg_pred) 
        output_im.putpalette(palette)
        output_im.save(args.save_dir+name[0]+'.png')
         
     
if __name__ == '__main__':
    main() 
