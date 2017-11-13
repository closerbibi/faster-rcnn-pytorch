#!/usr/bin/env python

import os
import torch
import numpy as np
import pdb
from datetime import datetime

from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN, RPN
from faster_rcnn.utils.timer import Timer

import faster_rcnn.roi_data_layer.roidb as rdl_roidb
from faster_rcnn.roi_data_layer.layer import RoIDataLayer
from faster_rcnn.datasets.factory import get_imdb
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file

import subprocess
from logger import Logger

try:
    from termcolor import cprint
except ImportError:
    cprint = None

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)



# hyper-parameters
# ------------
#imdb_name = 'voc_2007_trainval'
logger = Logger('./tb-logs')
imdb_name = 'inria_train'
cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
pretrained_model = 'data/pretrain_model/VGG_imagenet.npy'
#pretrained_model = 'models/gupta_19classes/faster_rcnn_100000.h5'
output_dir = 'models/resnet101-bn-block1-fix'

start_step = 0
end_step = 160000
lr_decay_steps = {60000, 100000, 130000}
lr_decay = 1./10
print 'iter : %d, step_size : %s, lr_decay : %f'%(end_step, lr_decay_steps, lr_decay)
rand_seed = 1024
_DEBUG = True
use_tensorboard = True
remove_all_log = False   # remove all historical experiments in TensorBoard
exp_name = None # the previous experiment name in TensorBoard

# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config
cfg_from_file(cfg_file)
lr = cfg.TRAIN.LEARNING_RATE
#lr = 0.00000001
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
#disp_interval = cfg.TRAIN.DISPLAY
disp_interval = 100
log_interval = cfg.TRAIN.LOG_IMAGE_ITERS

# load data
imdb = get_imdb(imdb_name)
rdl_roidb.prepare_roidb(imdb)
roidb = imdb.roidb
#pdb.set_trace()
data_layer = RoIDataLayer(roidb, imdb.num_classes)
#pdb.set_trace()
# load net
net = FasterRCNN(classes=imdb.classes, debug=_DEBUG)
network.weights_normal_init(net, dev=0.01)

# VGG
#network.load_pretrained_npy(net, pretrained_model)

# resnet
weight_path = {
    'resnet50coco': '/home/closerbibi/.torch/models/res50_faster_rcnn_iter_1190000.pth',
    'myres101': '/home/closerbibi/workspace/pytorch-repo/frcnn/frcnn-resnet/output/default/voc_2007_trainval/default/res101_faster_rcnn_iter_70000.pth',
    'ruores101': '/home/closerbibi/.torch/models/res101_faster_rcnn_iter_110000.pth'
}
network.load_resnet_weight(net, weight_path['ruores101'])

# model_file = '/media/longc/Data/models/VGGnet_fast_rcnn_iter_70000.h5'
# model_file = 'models/saved_model3/faster_rcnn_60000.h5'
# network.load_net(model_file, net)
# exp_name = 'vgg16_02-19_13-24'
# start_step = 60001
# lr /= 10.
# network.weights_normal_init(1[net.bbox_fc, net.score_fc, net.fc6, net.fc7], dev=0.01)

net.cuda()
net.train()

params = list(net.parameters())
# adam doesn't need to update lr manually
require_update = False
#optimizer = torch.optim.Adam(params[-8:], lr=lr)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params), lr=lr)
#optimizer = torch.optim.SGD(params[8:], lr=lr, momentum=momentum, weight_decay=weight_decay)
#optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, params[8:]), \
#                        lr=lr, momentum=momentum, weight_decay=weight_decay)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# training
train_loss = 0
tp, tf, fg, bg = 0., 0., 0, 0
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()
for step in range(start_step, end_step+1):
    #print step
    # get one batch
    blobs = data_layer.forward()
    im_data = blobs['data']
    im_info = blobs['im_info']
    gt_boxes = blobs['gt_boxes']
    gt_ishard = blobs['gt_ishard']
    dontcare_areas = blobs['dontcare_areas']
    #print 'now at %s'%blobs['im_name']
    #pdb.set_trace()
    # forward
    net(im_data, im_info, gt_boxes, gt_ishard, dontcare_areas)
    loss = net.loss + net.rpn.loss

    if _DEBUG:
        tp += float(net.tp)
        tf += float(net.tf)
        fg += net.fg_cnt
        bg += net.bg_cnt
    #print 'tp:%f, tf:%f, fg:%f, bg:%f'%(tp, tf, fg, bg)
    train_loss += loss.data[0]
    step_cnt += 1
    #pdb.set_trace()
    # backward
    optimizer.zero_grad()
    loss.backward()
    network.clip_gradient(net, 10.)
    optimizer.step()
    #pdb.set_trace()
    total_loss = net.rpn.cross_entropy.data.cpu().numpy()[0]+ net.rpn.loss_box.data.cpu().numpy()[0]+net.cross_entropy.data.cpu().numpy()[0]+net.loss_box.data.cpu().numpy()[0]
    if step % disp_interval == 0:
        duration = t.toc(average=False)
        fps = step_cnt / duration

        log_text = 'step %d, image: %s, loss: %.4f, fps: %.2f (%.2fs per batch)' % (
            step, blobs['im_name'], train_loss / step_cnt, fps, 1./fps)
        log_print(log_text, color='green', attrs=['bold'])

        if _DEBUG:
            log_print('\tTP: %.2f%%, TF: %.2f%%, fg/bg=(%d/%d)' % (tp/fg*100., tf/bg*100., fg/step_cnt, bg/step_cnt))
            log_print('\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box: %.4f, \ntotal loss: %.4f' % (
                net.rpn.cross_entropy.data.cpu().numpy()[0], net.rpn.loss_box.data.cpu().numpy()[0],
                net.cross_entropy.data.cpu().numpy()[0], net.loss_box.data.cpu().numpy()[0],
                total_loss))
            #print 'run'
        re_cnt = True

    if use_tensorboard and step % log_interval == 0:
        info = {
            'train_loss': loss.data[0],
            'current_total_loss': total_loss,
            'learning_rate': lr,
            'rpn_cls': float(net.rpn.cross_entropy.data.cpu().numpy()[0]),
            'rpn_box': float(net.rpn.loss_box.data.cpu().numpy()[0]),
            'rcnn_cls': float(net.cross_entropy.data.cpu().numpy()[0]),
            'rcnn_box': float(net.loss_box.data.cpu().numpy()[0]),
        }
        for tag, value in info.items():
            logger.scalar_summary(tag, value, step)

    if (step % 40000 == 0) and step > 0:
        save_name = os.path.join(output_dir, 'faster_rcnn_{}.h5'.format(step))
        network.save_net(save_name, net)
        print('save model: {}'.format(save_name))
        # evaluation
        print('Now evaluating...')
        cut_name = save_name.split('models/')[1].split('/')[0]
        bashCommand = "./test.py --name {} --it {:d} --rset {}".format(cut_name, step, 'test')
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
    # plot loss to jpg file
    #if (step % 500 == 0) and step > 0:
    #    os.system('python /home/closerbibi/workspace/tools/plot_pytorch_loss.py --inputfile=/home/closerbibi/workspace/pytorch/faster-rcnn_HaoEvaluation/log/os.system("python /home/closerbibi/workspace/tools/plot_pytorch_loss.py --inputfile=/home/closerbibi/workspace/pytorch/faster-rcnn_HaoEvaluation/log/log_trying.txt')
    if step in lr_decay_steps and require_update:
        lr *= lr_decay
        #optimizer = torch.optim.SGD(params[8:], lr=lr, momentum=momentum, weight_decay=weight_decay)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, params[8:]), \
                                lr=lr, momentum=momentum, weight_decay=weight_decay)

    if re_cnt:
        tp, tf, fg, bg = 0., 0., 0, 0
        train_loss = 0
        step_cnt = 0
        t.tic()
        re_cnt = False

