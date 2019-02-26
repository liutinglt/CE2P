import os
import sys
import numpy as np
import random
import cv2

def generate_edge(label, edge_width=3):
    h, w = label.shape
    edge = np.zeros(label.shape)

    # right
    edge_right = edge[1:h, :]
    edge_right[(label[1:h, :] != label[:h - 1, :]) & (label[1:h, :] != 255)
               & (label[:h - 1, :] != 255)] = 1

    # up
    edge_up = edge[:, :w - 1]
    edge_up[(label[:, :w - 1] != label[:, 1:w])
            & (label[:, :w - 1] != 255)
            & (label[:, 1:w] != 255)] = 1

    # upright
    edge_upright = edge[:h - 1, :w - 1]
    edge_upright[(label[:h - 1, :w - 1] != label[1:h, 1:w])
                 & (label[:h - 1, :w - 1] != 255)
                 & (label[1:h, 1:w] != 255)] = 1

    # bottomright
    edge_bottomright = edge[:h - 1, 1:w]
    edge_bottomright[(label[:h - 1, 1:w] != label[1:h, :w - 1])
                     & (label[:h - 1, 1:w] != 255)
                     & (label[1:h, :w - 1] != 255)] = 1

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_width, edge_width))
    edge = cv2.dilate(edge, kernel)
    return edge


def generate_edge_old(label, edge_width=2):
    h, w = label.shape
    edge = np.zeros(label.shape)
    for bin in range(edge_width):
        bin = bin + 1
        # right
        edge_right = edge[bin:h, :]
        edge_right[(label[bin:h, :] != label[:h - bin, :])
                   & (label[bin:h, :] != 255)
                   & (label[:h - bin, :] != 255)] = 1

        # left
        edge_left = edge[:h - bin, :]
        edge_left[(label[:h - bin, :] != label[bin:h, :])
                  & (label[:h - bin, :] != 255)
                  & (label[bin:h, :] != 255)] = 1
        # up
        edge_up = edge[:, :w - bin]
        edge_up[(label[:, :w - bin] != label[:, bin:w])
                & (label[:, :w - bin] != 255)
                & (label[:, bin:w] != 255)] = 1
        # bottom
        edge_bottom = edge[:, bin:w]
        edge_bottom[(label[:, bin:w] != label[:, :w - bin])
                    & (label[:, bin:w] != 255)
                    & (label[:, :w - bin] != 255)] = 1
        # upright
        edge_upright = edge[:h - bin, :w - bin]
        edge_upright[(label[:h - bin, :w - bin] != label[bin:h, bin:w])
                     & (label[:h - bin, :w - bin] != 255)
                     & (label[bin:h, bin:w] != 255)] = 1
        # upleft
        edge_upleft = edge[bin:h, bin:w]
        edge_upleft[(label[bin:h, bin:w] != label[:h - bin, :w - bin])
                    & (label[bin:h, bin:w] != 255)
                    & (label[:h - bin, :w - bin] != 255)] = 1
        # bottomright
        edge_bottomright = edge[:h - bin, bin:w]
        edge_bottomright[(label[:h - bin, bin:w] != label[bin:h, :w - bin])
                         & (label[:h - bin, bin:w] != 255)
                         & (label[bin:h, :w - bin] != 255)] = 1
        # bottomleft
        edge_bottomleft = edge[bin:h, :w - bin]
        edge_bottomleft[(label[bin:h, :w - bin] != label[:h - bin, bin:w])
                        & (label[bin:h, :w - bin] != 255)
                        & (label[:h - bin, bin:w] != 255)] = 1
    return edge


def generate_target(joints, joints_vis, num_joints, crop_size, heatmap_size, sigma=3):
    '''
    :param joints:  [num_joints, 3]
    :param joints_vis: [num_joints, 3]
    :return: target, target_weight(1: visible, 0: invisible)
    '''
    target_weight = np.ones((num_joints, 1), dtype=np.float32)
    target_weight[:, 0] = joints_vis[:, 0]

    target = np.zeros((num_joints,
                       heatmap_size[0],
                       heatmap_size[1]),
                      dtype=np.float32)

    tmp_size = sigma * 3

    for joint_id in range(num_joints):
        feat_stride = crop_size / heatmap_size
        mu_x = int(joints[joint_id][0] / feat_stride[1] + 0.5)
        mu_y = int(joints[joint_id][1] / feat_stride[0] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= heatmap_size[1] or ul[1] >= heatmap_size[0] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            target_weight[joint_id] = 0
            continue

        # # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], heatmap_size[1]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size[0]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], heatmap_size[1])
        img_y = max(0, ul[1]), min(br[1], heatmap_size[0])

        v = target_weight[joint_id]
        if v > 0.5:
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return target, target_weight


def _box2cs(box, aspect_ratio, pixel_std):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h, aspect_ratio, pixel_std)


def _xywh2cs(x, y, w, h, aspect_ratio, pixel_std):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
    # if center[0] != -1:
    #    scale = scale * 1.25

    return center, scale
