import argparse

import torch
torch.multiprocessing.set_start_method("spawn", force=True)
from torch.utils import data
import numpy as np
import torch.optim as optim
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
import os
import os.path as osp
from networks.CE2P import Res_Deeplab
from dataset.datasets import LIPDataSet
import torchvision.transforms as transforms
import timeit
from tensorboardX import SummaryWriter
from utils.utils import decode_parsing, inv_preprocess
from utils.criterion import CriterionAll
from utils.encoding import DataParallelModel, DataParallelCriterion 
from utils.miou import compute_mean_ioU
from evaluate import valid

start = timeit.default_timer()
  
BATCH_SIZE = 8
DATA_DIRECTORY = 'cityscapes'
DATA_LIST_PATH = './dataset/list/cityscapes/train.lst'
IGNORE_LABEL = 255
INPUT_SIZE = '769,769'
LEARNING_RATE = 1e-2
MOMENTUM = 0.9
NUM_CLASSES = 20
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = './dataset/MS_DeepLab_resnet_pretrained_init.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 10000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005
 
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="CE2P Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the dataset.")
    parser.add_argument("--dataset", type=str, default='train', choices=['train', 'val', 'trainval', 'test'],
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).") 
    parser.add_argument("--start-iters", type=int, default=0,
                        help="Number of classes to predict (including background).") 
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
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
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--gpu", type=str, default='None',
                        help="choose gpu device.")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="choose the number of recurrence.")
    parser.add_argument("--epochs", type=int, default=150,
                        help="choose the number of recurrence.")
    return parser.parse_args()


args = get_arguments()


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, total_iters):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(args.learning_rate, i_iter, total_iters, args.power)
    optimizer.param_groups[0]['lr'] = lr
    return lr


def adjust_learning_rate_pose(optimizer, epoch):
    decay = 0
    if epoch + 1 >= 230:
        decay = 0.05
    elif epoch + 1 >= 200:
        decay = 0.1
    elif epoch + 1 >= 120:
        decay = 0.25
    elif epoch + 1 >= 90:
        decay = 0.5
    else:
        decay = 1

    lr = args.learning_rate * decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
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

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    writer = SummaryWriter(args.snapshot_dir)
    gpus = [int(i) for i in args.gpu.split(',')]
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    h, w = map(int, args.input_size.split(','))
    input_size = [h, w]

    cudnn.enabled = True
    # cudnn related setting
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
 

    deeplab = Res_Deeplab(num_classes=args.num_classes)

    # dump_input = torch.rand((args.batch_size, 3, input_size[0], input_size[1]))
    # writer.add_graph(deeplab.cuda(), dump_input.cuda(), verbose=False)

    saved_state_dict = torch.load(args.restore_from)
    new_params = deeplab.state_dict().copy()
    for i in saved_state_dict:
        i_parts = i.split('.')
        # print(i_parts)
        if not i_parts[0] == 'fc':
            new_params['.'.join(i_parts[0:])] = saved_state_dict[i]

    deeplab.load_state_dict(new_params)
   
    model = DataParallelModel(deeplab)
    model.cuda()

    criterion = CriterionAll()
    criterion = DataParallelCriterion(criterion)
    criterion.cuda()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    trainloader = data.DataLoader(LIPDataSet(args.data_dir, args.dataset, crop_size=input_size, transform=transform),
                                  batch_size=args.batch_size * len(gpus), shuffle=True, num_workers=2,
                                  pin_memory=True)
    #lip_dataset = LIPDataSet(args.data_dir, 'val', crop_size=input_size, transform=transform)
    #num_samples = len(lip_dataset)
    
    #valloader = data.DataLoader(lip_dataset, batch_size=args.batch_size * len(gpus),
    #                             shuffle=False, pin_memory=True)

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    optimizer.zero_grad()

    total_iters = args.epochs * len(trainloader)
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        for i_iter, batch in enumerate(trainloader):
            i_iter += len(trainloader) * epoch
            lr = adjust_learning_rate(optimizer, i_iter, total_iters)

            images, labels, edges, _ = batch
            labels = labels.long().cuda(non_blocking=True)
            edges = edges.long().cuda(non_blocking=True)

            preds = model(images)

            loss = criterion(preds, [labels, edges])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i_iter % 100 == 0:
                writer.add_scalar('learning_rate', lr, i_iter)
                writer.add_scalar('loss', loss.data.cpu().numpy(), i_iter)

            if i_iter % 500 == 0:

                images_inv = inv_preprocess(images, args.save_num_images)
                labels_colors = decode_parsing(labels, args.save_num_images, args.num_classes, is_pred=False)
                edges_colors = decode_parsing(edges, args.save_num_images, 2, is_pred=False)

                if isinstance(preds, list):
                    preds = preds[0]
                preds_colors = decode_parsing(preds[0][-1], args.save_num_images, args.num_classes, is_pred=True)
                pred_edges = decode_parsing(preds[1][-1], args.save_num_images, 2, is_pred=True)

                img = vutils.make_grid(images_inv, normalize=False, scale_each=True)
                lab = vutils.make_grid(labels_colors, normalize=False, scale_each=True)
                pred = vutils.make_grid(preds_colors, normalize=False, scale_each=True)
                edge = vutils.make_grid(edges_colors, normalize=False, scale_each=True)
                pred_edge = vutils.make_grid(pred_edges, normalize=False, scale_each=True)

                writer.add_image('Images/', img, i_iter)
                writer.add_image('Labels/', lab, i_iter)
                writer.add_image('Preds/', pred, i_iter)
                writer.add_image('Edges/', edge, i_iter)
                writer.add_image('PredEdges/', pred_edge, i_iter)

            print('iter = {} of {} completed, loss = {}'.format(i_iter, total_iters, loss.data.cpu().numpy()))

        torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'LIP_epoch_' + str(epoch) + '.pth'))

        #parsing_preds, scales, centers = valid(model, valloader, input_size,  num_samples, len(gpus))

        #mIoU = compute_mean_ioU(parsing_preds, scales, centers, args.num_classes, args.data_dir, input_size)

        #print(mIoU)
        #writer.add_scalars('mIoU', mIoU, epoch)

    end = timeit.default_timer()
    print(end - start, 'seconds')
 

if __name__ == '__main__':
    main()
