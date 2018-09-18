#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-01

import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

MAX_ITER = 10
POS_W = 3
POS_XY_STD = 3
Bi_W = 4
Bi_XY_STD = 49
Bi_RGB_STD = 5


def dense_crf(img, probs):
    c = probs.shape[0]
    h = probs.shape[1]
    w = probs.shape[2]

    U = utils.unary_from_softmax(probs)
    U = np.ascontiguousarray(U)

    img = np.ascontiguousarray(img)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=img, compat=Bi_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))

    return Q
