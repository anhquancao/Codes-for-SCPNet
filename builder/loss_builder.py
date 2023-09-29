# -*- coding:utf-8 -*-
# author: Xinge
# @file: loss_builder.py 

import torch
from utils.lovasz_losses import lovasz_softmax
import numpy as np




def build(wce=True, lovasz=True, num_class=20, ignore_label=0):
    freq = np.array([5.4226e+09, 1.5640e+07, 1.1710e+05, 1.1879e+05, 6.0278e+05, 8.3570e+05,
        2.6682e+05, 2.6566e+05, 1.6459e+05, 6.1145e+07, 4.2558e+06, 4.4079e+07,
        2.5098e+06, 5.6889e+07, 1.5568e+07, 1.5888e+08, 2.0582e+06, 3.7056e+07,
        1.1631e+06, 3.3958e+05])
    complt_num_per_class = freq
    compl_labelweights = complt_num_per_class / np.sum(complt_num_per_class)
    compl_labelweights = np.power(np.amax(compl_labelweights) / compl_labelweights, 1 / 3.0)

    compl_labelweights = torch.from_numpy(compl_labelweights).cuda().float()

    loss_funs = torch.nn.CrossEntropyLoss(weight=compl_labelweights, ignore_index=ignore_label)
  

    if wce and lovasz:
        return loss_funs, lovasz_softmax
    elif wce and not lovasz:
        return wce
    elif not wce and lovasz:
        return lovasz_softmax
    else:
        raise NotImplementedError
