# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import config

class FullModel(nn.Module):

  def __init__(self, model, sem_loss, bd_loss):
    super(FullModel, self).__init__()
    self.model = model
    self.sem_loss = sem_loss
    self.bd_loss = bd_loss

  def pixel_acc(self, pred, label):
    _, preds = torch.max(pred, dim=1) # preds:[6, 1024, 1024]
    valid = (label >= 0).long()
    acc_sum = torch.sum(valid * (preds == label).long())
    pixel_sum = torch.sum(valid)
    acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
    return acc

  def forward(self, inputs, labels, bd_gt, *args, **kwargs):
    
    # output: [x_extra_p, x_, x_extra_d]，均为logits值
    outputs = self.model(inputs, *args, **kwargs)
    
    h, w = labels.size(1), labels.size(2)
    ph, pw = outputs[0].size(2), outputs[0].size(3)
    if ph != h or pw != w:
        for i in range(len(outputs)):
            outputs[i] = F.interpolate(outputs[i], size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

    # x_extra_p:[6,19,1024,1024], x_:[6,19,1024,1024], x_extra_d;[6,1,1024,1024]
    acc  = self.pixel_acc(outputs[-2], labels) # len(acc)=1
    # loss_s： extra P_branch CE loss L0(standard) + predicted CE loss L2(OHEM)
    # loss_b：weighted binary cross entropy loss L1
    # loss_sb：boundary-awareness CE loss L3
    loss_s = self.sem_loss(outputs[:-1], labels) # loss_s=[b,h,w]:[6,1024,1024]
    # loss_s = self.sem_loss(outputs[-2], labels)
    loss_b = self.bd_loss(outputs[-1], bd_gt) # loss_b:1

    filler = torch.ones_like(labels) * config.TRAIN.IGNORE_LABEL
    # torch.where(condition, x, y) 的语法如下：
    # condition：一个布尔张量，表示元素级的条件。
    #   如果条件为 True，则选择 x 中对应位置的元素；
    #   如果条件为 False，则选择 y 中对应位置的元素。
    # size: bd_label = labels = filler
    
    # boundary = torch.sigmoid(outputs[-1][:,0,:,:])
    # max_value = torch.max(boundary)
    # min_value = torch.min(boundary)
    # print("max_value:",max_value)
    # print("min_value:",min_value)
    
    bd_label = torch.where(torch.sigmoid(outputs[-1][:,0,:,:])>0.6, labels, filler)
    loss_sb = self.sem_loss(outputs[-2], bd_label) # loss_sb:1
    # loss = loss_s + loss_b + loss_sb # loss:[b,h,w]
    loss = loss_s + loss_sb # loss:[b,h,w]
    # return:loss=[1,b,h,w], outputs=[x_extra_p, x_], acc=1, [loss_s, loss_b]
    return torch.unsqueeze(loss,0), outputs[:-1], acc, [loss_s, loss_b]
    # return torch.unsqueeze(loss,0), outputs[:-1], acc, [loss_s]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
            (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int32)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix

def adjust_learning_rate(optimizer, base_lr, max_iters, 
        cur_iters, power=0.9, nbb_mult=10):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = lr * nbb_mult
    return lr

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}