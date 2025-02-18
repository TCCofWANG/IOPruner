import torch
import torch.nn as nn
import argparse
import utils.common as utils
from importlib import import_module
from thop import profile
from utils.builder import get_builder
from collections import OrderedDict
import numpy as np
from data import cifar10, cifar100, imagenet
from utils.options import args
import random

device = 'cpu'
cfg = []
cfg2 = []
layer_cfg = []
cfg_mask = []
print('==> Building model..')

if args.arch == 'vgg_cifar':
    orimodel_flops = import_module(f'model.{args.arch}').VGG_flops(args.cfg).to(device)
    ckpt = torch.load(args.sparse_model, map_location = device)
    model_flops = import_module(f'model.{args.arch}').VGG_sparse_flops(args.cfg,layer_cfg=ckpt['layer_cfg']).to(device)
elif args.arch == 'googlenet_cifar':
    orimodel_flops = import_module(f'model.{args.arch}').googlenet_flops().to(device)
    ckpt = torch.load(args.sparse_model, map_location=device)
    print(ckpt['layer_cfg'])
    model_flops = import_module(f'model.{args.arch}').googlenet_flops(layer_cfg=ckpt['layer_cfg']).to(device)
    model_flops.load_state_dict(ckpt['state_dict'][0])
elif args.arch == 'resnet':
    if args.cfg == 'resnet50':
        orimodel_flops = import_module(f'model.{args.arch}').resnet50_flops().to(device)
    elif args.cfg == 'resnet101':
        orimodel_flops = import_module(f'model.{args.arch}').resnet101_flops().to(device)
    elif args.cfg == 'resnet152':
        orimodel_flops = import_module(f'model.{args.arch}').resnet152_flops().to(device)
    ckpt = torch.load(args.sparse_model, map_location = device)
    print(ckpt['layer_cfg'])
    if args.cfg == 'resnet50':
        model_flops = import_module(f'model.{args.arch}').resnet50_flops(layer_cfg = ckpt['layer_cfg']).to(device)
    elif args.cfg == 'resnet101':
        model_flops = import_module(f'model.{args.arch}').resnet101_flops(layer_cfg = ckpt['layer_cfg']).to(device)
    elif args.cfg == 'resnet152':
        model_flops = import_module(f'model.{args.arch}').resnet152_flops(layer_cfg = ckpt['layer_cfg']).to(device)
    model_flops.load_state_dict(ckpt['state_dict'][0])

if args.data_set == 'cifar10':
    input_image_size = 32
    loader = cifar10.Data(args)
elif args.data_set == 'imagenet':
    input_image_size = 224
    loader = imagenet.Data(args)

input = torch.randn(1, 3, input_image_size, input_image_size).to(device)

oriflops, oriparams = profile(orimodel_flops, inputs=(input, ))
flops, params = profile(model_flops, inputs=(input, ))

print('--------------Original Model--------------')
print('Params: %.2f M '%(oriparams/1000000))
print('FLOPS: %.2f M '%(oriflops/1000000))

print('--------------Prune Model--------------')
print('Params: %.2f M'%(params/1000000))
print('FLOPS: %.2f M'%(flops/1000000))

print('--------------Prune Rate--------------')
print('Params Compress Rate: %.2f M/%.2f M(%.2f%%)' % (params/1000000, oriparams/1000000, 100. * (oriparams - params) / oriparams))
print('FLOPS Compress Rate: %.2f M/%.2f M(%.2f%%)' % (flops/1000000, oriflops/1000000, 100. * (oriflops - flops) / oriflops))