import torch
import torch.nn as nn
import torch.optim as optim
from utils.builder import get_builder
from utils.options import args
import utils.common as utils
import os
import time
import math
import copy
import sys
import random
import numpy as np
import heapq
from data import imagenet
from importlib import import_module
from torchvision import datasets, transforms


checkpoint = utils.checkpoint(args)
device = torch.device(f'cuda:{args.gpus[0]}')if torch.cuda.is_available() else 'cpu'
logger = utils.get_logger(os.path.join(args.job_dir + 'logger.log'))
criterion = torch.nn.CrossEntropyLoss()

print('==> Loading Data..')
loader = imagenet.Data(args)

def get_model(args):
    if args.cfg == 'resnet50':
        model = import_module(f'model.{args.arch}').resnet50().to(device)
    elif args.cfg == 'resnet101':
        model = import_module(f'model.{args.arch}').resnet101().to(device)
    elif args.cfg == 'resnet152':
        model = import_module(f'model.{args.arch}').resnet152().to(device)
    ckpt = torch.load(args.pretrain_model)
    model.load_state_dict(ckpt,strict = False)

    print('==> Testing Baseline Model..')
    test(model, testLoader=loader.testLoader)

    return model

def get_optimizer(args, model):
    parameters = list(model.named_parameters())
    bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
    rest_params = [v for n, v in parameters if ("bn" not in n) and v.requires_grad]
    optimizer = torch.optim.SGD(
        [
            {
                "params": bn_params,
                "weight_decay": args.weight_decay,
            },
            {"params": rest_params, "weight_decay": args.weight_decay},
        ],
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    return optimizer

def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = args.lr*(0.1**factor)

    """Warmup"""
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(model, optimizer, trainLoader, args, epoch):
    model.train()
    losses = utils.AverageMeter()
    accurary = utils.AverageMeter()
    accurary5 = utils.AverageMeter()
    print_freq = len(trainLoader.dataset) // args.train_batch_size // 10
    start_time = time.time()
    for batch, (inputs, targets) in enumerate(trainLoader):
        inputs, targets = inputs.to(device), targets.to(device)
        train_loader_len = int(math.ceil(len(trainLoader.dataset) / args.train_batch_size))

        adjust_learning_rate(optimizer, epoch, batch, train_loader_len)

        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)

        loss.backward()
        losses.update(loss.item(), inputs.size(0))
        optimizer.step()

        prec1,prec5 = utils.accuracy(output, targets, topk = (1, 5))
        accurary.update(prec1[0], inputs.size(0))
        accurary5.update(prec5[0], inputs.size(0))

        if batch % print_freq == 0 and batch != 0:
            current_time = time.time()
            cost_time = current_time - start_time
            logger.info(
                'Epoch[{}] ({}/{}):\t'
                'Loss {:.4f}\t'
                'Top1 {:.2f}%\t'
                'Top5 {:.2f}%\t\t'
                'Time {:.2f}s'.format(
                    epoch, batch * args.train_batch_size, len(trainLoader.dataset),
                    float(losses.avg), float(accurary.avg), float(accurary5.avg), cost_time
                )
            )
            start_time = current_time

def test(model, testLoader):
    model.eval()

    losses = utils.AverageMeter()
    accurary = utils.AverageMeter()
    accurary5 = utils.AverageMeter()
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            prec1,prec5 = utils.accuracy(outputs, targets, topk = (1, 5))
            accurary.update(prec1[0], inputs.size(0))
            accurary5.update(prec5[0], inputs.size(0))

        current_time = time.time()
        logger.info(
            'Test Loss {:.4f}\tTop1 {:.2f}%\tTop5 {:.2f}%\t\tTime {:.2f}s\n'
            .format(float(losses.avg), float(accurary.avg), float(accurary5.avg), (current_time - start_time))
        )
    return (accurary.avg,accurary5.avg)

def main():
    start_epoch = 0
    best_acc = 0.0
    best_acc5 = 0.0
    first_conv = True
    print('==> Building Baseline Model...')
    model = get_model(args)


    if args.resume == None:
        last_module = []
        layer_cfg = []
        print('==> Generating Masked Model...')
        for name, module in model.named_modules():
            if hasattr(module, "mask"):
                if first_conv == True:
                    first_conv = False
                    continue
                else:
                    if 'conv1' in name:
                        last_module = module
                    if 'conv2' in name:
                        indice_to_keep_out,indice_to_keep_in = module.get_mask_inout_resnet_imagenet(args)
                        last_module.get_mask_out_resnet_imagenet(indice_to_keep_in)
                    if 'conv3' in name:
                        module.get_mask_in_resnet_imagenet(indice_to_keep_out)
        print('==> Generating Pruned Model...')
        first_conv = True
        cfg_mask = []
        for name, module in model.named_modules():
            if hasattr(module, "mask"):
                c_out, c_in, k_1, k_2 = module.mask.shape
                ww = module.mask.contiguous().view(c_out, c_in * k_1 * k_2)
                ww = torch.sum(torch.abs(ww), 1)
                cfg_mask_indice = torch.zeros(c_out)
                cfg_mask_indice[torch.nonzero(ww)] = 1
                cfg_mask.append(cfg_mask_indice)
                if 'conv1' in name or 'conv2' in name:
                    if first_conv:
                        first_conv = False
                        continue
                    else:
                        layer_cfg.append(torch.nonzero(ww).size(0))
        print(len(layer_cfg))
        logger.info('layer_cfg: {}'.format(layer_cfg))
        if args.cfg == 'resnet50':
            model_flops = import_module(f'model.{args.arch}').resnet50_flops(layer_cfg = layer_cfg).to(device)
        elif args.cfg == 'resnet101':
            model_flops = import_module(f'model.{args.arch}').resnet101_flops(layer_cfg = layer_cfg).to(device)
        elif args.cfg == 'resnet152':
            model_flops = import_module(f'model.{args.arch}').resnet152_flops(layer_cfg = layer_cfg).to(device)
        old_modules = list(model.modules())
        new_modules = list(model_flops.modules())
        layer_id_in_cfg = 0
        start_mask = torch.ones(3)
        end_mask = cfg_mask[layer_id_in_cfg]
        conv_count = 0
        for layer_id in range(len(old_modules)):
            m0 = old_modules[layer_id]
            m1 = new_modules[layer_id]
            if isinstance(m0, nn.BatchNorm2d):
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                m1.running_var = m0.running_var[idx1.tolist()].clone()
                if args.cfg == 'resnet50':
                    if conv_count == 4 or conv_count == 14 or conv_count == 27 or conv_count == 46:
                        layer_id_in_cfg += 1
                    else:
                        layer_id_in_cfg += 1
                        start_mask = end_mask.clone()
                        if layer_id_in_cfg < len(cfg_mask):
                            end_mask = cfg_mask[layer_id_in_cfg]
                elif args.cfg == 'resnet101':
                    if conv_count == 4 or conv_count == 14 or conv_count == 27 or conv_count == 97:
                        layer_id_in_cfg += 1
                    else:
                        layer_id_in_cfg += 1
                        start_mask = end_mask.clone()
                        if layer_id_in_cfg < len(cfg_mask):
                            end_mask = cfg_mask[layer_id_in_cfg]
                elif args.cfg == 'resnet152':
                    if conv_count == 4 or conv_count == 14 or conv_count == 39 or conv_count == 148:
                        layer_id_in_cfg += 1
                    else:
                        layer_id_in_cfg += 1
                        start_mask = end_mask.clone()
                        if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                            end_mask = cfg_mask[layer_id_in_cfg]
            elif isinstance(m0, nn.Conv2d):
                if args.cfg == 'resnet50':
                    if conv_count == 0 or conv_count == 4 or conv_count == 14 or conv_count == 27 or conv_count == 46:
                        m1.weight.data = m0.weight.data.clone()
                        conv_count += 1
                        continue
                elif args.cfg == 'resnet101':
                    if conv_count == 0 or conv_count == 4 or conv_count == 14 or conv_count == 27 or conv_count == 97:
                        m1.weight.data = m0.weight.data.clone()
                        conv_count += 1
                        continue
                elif args.cfg == 'resnet152':
                    if conv_count == 0 or conv_count == 4 or conv_count == 14 or conv_count == 39 or conv_count == 148:
                        m1.weight.data = m0.weight.data.clone()
                        conv_count += 1
                        continue
                if isinstance(old_modules[layer_id + 1], nn.BatchNorm2d):
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                    w1 = w1[idx1.tolist(), :, :, :].clone()
                    m1.weight.data = w1.clone()
                    conv_count += 1
            elif isinstance(m0, nn.Linear):
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                m1.weight.data = m0.weight.data[:, idx0].clone()
                m1.bias.data = m0.bias.data.clone()

        optimizer = get_optimizer(args, model_flops)
    else:
        checkpoint_r = torch.load(args.resume, map_location = device)
        start_epoch = checkpoint_r['epoch']
        best_acc = checkpoint_r['best_acc']
        best_acc5 = checkpoint_r['best_acc5']
        layer_cfg = checkpoint_r['layer_cfg']
        if args.cfg == 'resnet50':
            model_flops = import_module(f'model.{args.arch}').resnet50_flops(layer_cfg = layer_cfg).to(device)
        elif args.cfg == 'resnet101':
            model_flops = import_module(f'model.{args.arch}').resnet101_flops(layer_cfg = layer_cfg).to(device)
        elif args.cfg == 'resnet152':
            model_flops = import_module(f'model.{args.arch}').resnet152_flops(layer_cfg = layer_cfg).to(device)
        model_flops.load_state_dict(checkpoint_r['state_dict'][0])
        optimizer = get_optimizer(args, model_flops)
        optimizer.load_state_dict(checkpoint_r['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {}) Top1: {:.2f} Top5: {:.2f}".format(args.resume, checkpoint_r['epoch'],
                                                                                      float(best_acc), float(best_acc5)))

    if len(args.gpus) != 1:
        model_flops = nn.DataParallel(model_flops, device_ids = args.gpus)

    if args.test_only:
        test(model_flops, loader.testLoader)
    else:
        print('==> Start Train...')
        for epoch in range(start_epoch, args.num_epochs):
            train(model_flops, optimizer, loader.trainLoader, args, epoch)
            test_acc = test(model_flops, loader.testLoader)

            top1 = test_acc[0]
            top5 = test_acc[1]

            is_best = best_acc < top1
            best_acc = max(best_acc, top1)
            best_acc5 = max(best_acc5, top5)

            model_state_dict = model_flops.module.state_dict() if len(args.gpus) > 1 else model.state_dict(),

            state = {
                'state_dict': model_state_dict,
                'best_acc': best_acc,
                'best_acc5': best_acc5,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'layer_cfg': layer_cfg
            }
            checkpoint.save_model(state, epoch + 1, is_best)

        logger.info('Best accurary(Top1): {:.2f} (Top5): {:.2f}'.format(float(best_acc), float(best_acc5)))
        logger.info('--------------------------------------------------------------------------------------------------------------')

if __name__ == '__main__':
    main()
