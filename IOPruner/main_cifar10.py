import torch
import torch.nn as nn
import torch.optim as optim

from model.googlenet_cifar import Inception_flops, GoogLeNet_flops
from utils.builder import get_builder
from utils.options import args
import utils.common as utils
import os
import time
import copy
import sys
import random
import numpy as np
import heapq
from data import cifar10, cifar100
from importlib import import_module
from torchvision import datasets, transforms

checkpoint = utils.checkpoint(args)
device = torch.device(f'cuda:{args.gpus[0]}')if torch.cuda.is_available() else 'cpu'
logger = utils.get_logger(os.path.join(args.job_dir + 'logger.log'))
criterion = torch.nn.CrossEntropyLoss()

print('==> Loading Data..')
if args.data_set == 'cifar10':
    loader = cifar10.Data(args)
elif args.data_set == 'cifar100':
    loader = cifar100.Data(args)

def get_model(args):
    if args.arch == 'vgg_cifar':
        model = import_module(f'model.{args.arch}').VGG(get_builder(),args.cfg).to(device)
        ckpt = torch.load(args.pretrain_model, map_location = device)
        model.load_state_dict(ckpt['state_dict'],strict = False)
    elif args.arch == 'resnet_cifar':
        model = import_module(f'model.{args.arch}').resnet(args.cfg).to(device)
        ckpt = torch.load(args.pretrain_model, map_location = device)
        model.load_state_dict(ckpt['state_dict'], strict = False)
    elif args.arch == 'googlenet_cifar':
        model = import_module(f'model.{args.arch}').googlenet().to(device)
        ckpt = torch.load(args.pretrain_model, map_location=device)
        model.load_state_dict(ckpt['state_dict'], strict=False)

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

def train(model, optimizer, trainLoader, args, epoch):
    model.train()
    losses = utils.AverageMeter()
    accurary = utils.AverageMeter()
    print_freq = len(trainLoader.dataset) // args.train_batch_size // 10
    start_time = time.time()
    for batch, (inputs, targets) in enumerate(trainLoader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)

        loss.backward()
        losses.update(loss.item(), inputs.size(0))
        optimizer.step()

        prec1 = utils.accuracy(output, targets)
        accurary.update(prec1[0], inputs.size(0))

        if batch % print_freq == 0 and batch != 0:
            current_time = time.time()
            cost_time = current_time - start_time
            logger.info(
                'Epoch[{}] ({}/{}):\t'
                'Loss {:.4f}\t'
                'Accurary {:.2f}%\t\t'
                'Time {:.2f}s'.format(
                    epoch, batch * args.train_batch_size, len(trainLoader.dataset),
                    float(losses.avg), float(accurary.avg), cost_time
                )
            )
            start_time = current_time

def test(model, testLoader):
    model.eval()

    losses = utils.AverageMeter()
    accurary = utils.AverageMeter()
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            predicted = utils.accuracy(outputs, targets)
            accurary.update(predicted[0], inputs.size(0))

        current_time = time.time()
        logger.info(
            'Test Loss {:.4f}\tAccurary {:.2f}%\t\tTime {:.2f}s\n'
            .format(float(losses.avg), float(accurary.avg), (current_time - start_time))
        )
    return accurary.avg

def main():
    start_epoch = 0
    best_acc = 0.0
    print('==> Building Baseline Model...')
    model = get_model(args)

    if args.resume == None:
        i = 0
        j = 0
        indice_to_keep_out_set = []
        last_module = []
        next_module = []
        cfg = []
        cfg2 = []
        cfg_mask = []
        cfg_mask_1_1 = []
        cfg_mask_3_3 = []
        cfg_mask_5_5 = []
        cfg_mask_pool = []
        mask1 = []
        mask2 = []
        mask3 = []
        mask4 = []
        mask_all = []

        reversed_modules = list(reversed(list(model.named_modules())))

        layer_cfg = []
        print('==> Generating Mask Model...')
        if args.arch == 'vgg_cifar':
            for name, module in model.named_modules():
                if hasattr(module, "mask"):
                    if i % 2 == 0:
                        i += 1
                        last_module = module
                        continue
                    else:
                        indice_to_keep_out,indice_to_keep_in = module.get_mask_inout_vgg(args)
                        indice_to_keep_out_set.append(indice_to_keep_out)
                        last_module.get_mask_out_vgg(indice_to_keep_in)
                        i += 1
            for name,module in reversed_modules:
                if hasattr(module,"mask"):
                    if j % 2 == 0:
                        j += 1
                        next_module = module
                        continue
                    else:
                        indice_to_keep_out = indice_to_keep_out_set[len(indice_to_keep_out_set) - int((j + 1) / 2)]
                        next_module.get_mask_in_vgg(indice_to_keep_out)
                        j += 1
            print('==> Generating Pruned Model...')
            for name, module in model.named_modules():
                if hasattr(module, "mask"):
                    c_out, c_in, k_1, k_2 = module.mask.shape
                    w = module.mask.permute(1, 0, 2, 3)
                    w = w.contiguous().view(c_in, c_out * k_1 * k_2)
                    w = torch.sum(torch.abs(w), 1)
                    non_zero = torch.nonzero(w).size(0)
                    cfg.append(non_zero)
                    ww = module.mask.contiguous().view(c_out, c_in * k_1 * k_2)
                    ww = torch.sum(torch.abs(ww), 1)
                    non_zero2 = torch.nonzero(ww).size(0)
                    cfg2.append(non_zero2)
                    cfg_mask_indice = torch.zeros(c_out)
                    cfg_mask_indice[torch.nonzero(ww)] = 1
                    cfg_mask.append(cfg_mask_indice)

            logger.info('layer_cfg: {}'.format(cfg2))
            model_flops = import_module(f'model.{args.arch}').VGG_sparse_flops(args.cfg, layer_cfg = cfg2).to(device)

            layer_id_in_cfg = 0
            start_mask = torch.ones(3)
            end_mask = cfg_mask[layer_id_in_cfg]
            for [m0, m1] in zip(model.modules(), model_flops.modules()):
                if isinstance(m1, nn.BatchNorm2d):
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                    m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                    m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                    m1.running_var = m0.running_var[idx1.tolist()].clone()
                    layer_id_in_cfg += 1
                    start_mask = end_mask.clone()
                    if layer_id_in_cfg < len(cfg_mask):
                        end_mask = cfg_mask[layer_id_in_cfg]
                elif isinstance(m1, nn.Conv2d):
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                    w1 = w1[idx1.tolist(), :, :, :].clone()
                    w2 = m0.bias.data[idx1.tolist()].clone()
                    m1.weight.data = w1.clone()
                    m1.bias.data = w2.clone()
                elif isinstance(m1, nn.Linear):
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    m1.weight.data = m0.weight.data[:, idx0].clone()
                    m1.bias.data = m0.bias.data.clone()
        elif args.arch == 'googlenet_cifar':
            for name, module in model.named_modules():
                if hasattr(module, "mask"):
                    if "branch3x3.0" in name:
                        last_module = module
                    elif "branch3x3.3" in name:
                        print("branch3x3.3")
                        indice_to_keep_out_3x3, indice_to_keep_in_3x3 = module.get_mask_inout_googlenet(args)
                        last_module.get_mask_out_googlenet(indice_to_keep_in_3x3)
                    elif "branch5x5.0" in name:
                        last_module = module
                    elif "branch5x5.3" in name:
                        print("branch5x5.3")
                        indice_to_keep_out_5x5, indice_to_keep_in_5x5 = module.get_mask_inout_googlenet(args)
                        last_module.get_mask_out_googlenet(indice_to_keep_in_5x5)
                    elif "branch5x5.6" in name:
                        module.get_mask_in_googlenet(indice_to_keep_out_5x5)
            print('==> Generating Pruned Model...')
            for name, module in model.named_modules():
                if hasattr(module, "mask"):
                    if "branch3x3.3" in name or "branch5x5.3" in name:
                        c_out, c_in, k_1, k_2 = module.mask.shape
                        w = module.mask.permute(1, 0, 2, 3)
                        w = w.contiguous().view(c_in, c_out * k_1 * k_2)
                        w = torch.sum(torch.abs(w), 1)

                        ww = module.mask.contiguous().view(c_out, c_in * k_1 * k_2)
                        ww = torch.sum(torch.abs(ww), 1)

                        cfg_mask_indice = torch.zeros(c_out)
                        cfg_mask_indice_in = torch.zeros(c_in)
                        cfg_mask_indice[torch.nonzero(ww)] = 1
                        cfg_mask_indice_in[torch.nonzero(w)] = 1
                        cfg_mask.append(cfg_mask_indice_in)
                        cfg_mask.append(cfg_mask_indice)

                        layer_cfg.append(torch.nonzero(w).size(0))
                        layer_cfg.append(torch.nonzero(ww).size(0))

                    if "branch1x1.0" in name:
                        c_out, c_in, k_1, k_2 = module.mask.shape
                        ww = module.mask.contiguous().view(c_out, c_in * k_1 * k_2)
                        ww = torch.sum(torch.abs(ww), 1)
                        cfg_mask_indice = torch.zeros(c_out)
                        cfg_mask_indice[torch.nonzero(ww)] = 1
                        cfg_mask_1_1.append(cfg_mask_indice)

                    if"branch3x3.3" in name:
                        c_out, c_in, k_1, k_2 = module.mask.shape
                        ww = module.mask.contiguous().view(c_out, c_in * k_1 * k_2)
                        ww = torch.sum(torch.abs(ww), 1)
                        cfg_mask_indice = torch.zeros(c_out)
                        cfg_mask_indice[torch.nonzero(ww)] = 1
                        cfg_mask_3_3.append(cfg_mask_indice)

                    if "branch5x5.6" in name:
                        c_out, c_in, k_1, k_2 = module.mask.shape
                        ww = module.mask.contiguous().view(c_out, c_in * k_1 * k_2)
                        ww = torch.sum(torch.abs(ww), 1)
                        cfg_mask_indice = torch.zeros(c_out)
                        cfg_mask_indice[torch.nonzero(ww)] = 1
                        cfg_mask_5_5.append(cfg_mask_indice)

                    if "branch_pool.1" in name:
                        c_out, c_in, k_1, k_2 = module.mask.shape
                        ww = module.mask.contiguous().view(c_out, c_in * k_1 * k_2)
                        ww = torch.sum(torch.abs(ww), 1)
                        cfg_mask_indice = torch.zeros(c_out)
                        cfg_mask_indice[torch.nonzero(ww)] = 1
                        cfg_mask_pool.append(cfg_mask_indice)

            logger.info('layer_cfg: {}'.format(layer_cfg))
            model_flops = import_module(f'model.{args.arch}').googlenet_flops(layer_cfg=layer_cfg).to(device)

            Inception_id_in_cfg = 0

            for a, b, c, d in zip(cfg_mask_1_1, cfg_mask_3_3, cfg_mask_5_5, cfg_mask_pool):
                concatenated = torch.cat((a, b, c, d), dim = 0)
                mask_all.append(concatenated)

            for [m0, m1] in zip(model.modules(), model_flops.modules()):
                pre_id = 0
                if isinstance(m1, GoogLeNet_flops):
                    if (Inception_id_in_cfg + 1) * 4 < len(cfg_mask):
                        mask1 = cfg_mask[0]
                        mask2 = cfg_mask[1]
                        mask3 = cfg_mask[2]
                        mask4 = cfg_mask[3]
                    if pre_id == 0:
                        w1 = m0.pre_layers[pre_id].weight.data[:, :, :, :].clone()
                        w1 = w1[:, :, :, :].clone()
                        w2 = m0.pre_layers[pre_id].bias.data.clone()
                        m1.pre_layers[pre_id].weight.data = w1.clone()
                        m1.pre_layers[pre_id].bias.data = w2.clone()
                        pre_id += 1
                    elif pre_id == 1:
                        m1.pre_layers[pre_id].weight.data = m0.pre_layers[pre_id].weight.data.clone()
                        m1.pre_layers[pre_id].bias.data = m0.pre_layers[pre_id].bias.data.clone()
                        m1.pre_layers[pre_id].running_mean = m0.pre_layers[pre_id].running_mean.clone()
                        m1.pre_layers[pre_id].running_var = m0.pre_layers[pre_id].running_var.clone()
                        pre_id += 1
                if isinstance(m1, Inception_flops):
                    for sub_name, sub_module in m1.named_modules():
                        if sub_name == 'branch1x1':
                            conv_id = 0
                            for sub_sub_module in sub_module.named_modules():
                                if isinstance(sub_sub_module[-1], nn.BatchNorm2d):
                                    branch1x1 = m0.branch1x1
                                    branch1x1_new = m1.branch1x1
                                    branch1x1_new[conv_id].weight.data = branch1x1[conv_id].weight.data.clone()
                                    branch1x1_new[conv_id].bias.data = branch1x1[conv_id].bias.data.clone()
                                    branch1x1_new[conv_id].running_mean = branch1x1[conv_id].running_mean.clone()
                                    branch1x1_new[conv_id].running_var = branch1x1[conv_id].running_var.clone()
                                elif isinstance(sub_sub_module[-1], nn.Conv2d):
                                    if Inception_id_in_cfg == 0:
                                        branch1x1 = m0.branch1x1
                                        branch1x1_new = m1.branch1x1
                                        w1 = branch1x1[conv_id].weight.data[:, :, :, :].clone()
                                        w1 = w1[:, :, :, :].clone()
                                        branch1x1_new[conv_id].weight.data = w1.clone()
                                        conv_id += 1
                                    else:
                                        idx1 = np.squeeze(np.argwhere(np.asarray(mask_all[Inception_id_in_cfg-1].cpu().numpy())))
                                        if idx1.size == 1:
                                            idx1 = np.resize(idx1, (1,))
                                        branch1x1 = m0.branch1x1
                                        branch1x1_new = m1.branch1x1
                                        w1 = branch1x1[conv_id].weight.data[:, :, :, :].clone()
                                        w1 = w1[:, idx1.tolist(), :, :].clone()
                                        branch1x1_new[conv_id].weight.data = w1.clone()
                                        conv_id += 1

                        elif sub_name == 'branch3x3':
                            conv_id = 0
                            for sub_sub_module in sub_module.named_modules():
                                if isinstance(sub_sub_module[-1], nn.BatchNorm2d):
                                    if conv_id != 4:
                                        idx1 = np.squeeze(np.argwhere(np.asarray(mask1.cpu().numpy())))
                                        if idx1.size == 1:
                                            idx1 = np.resize(idx1, (1,))
                                        branch3x3 = m0.branch3x3
                                        branch3x3_new = m1.branch3x3
                                        branch3x3_new[conv_id].weight.data = branch3x3[conv_id].weight.data[idx1.tolist()].clone()
                                        branch3x3_new[conv_id].bias.data = branch3x3[conv_id].bias.data[idx1.tolist()].clone()
                                        branch3x3_new[conv_id].running_mean = branch3x3[conv_id].running_mean[idx1.tolist()].clone()
                                        branch3x3_new[conv_id].running_var = branch3x3[conv_id].running_var[idx1.tolist()].clone()
                                        conv_id += 2

                                    else:
                                        idx1 = np.squeeze(np.argwhere(np.asarray(mask2.cpu().numpy())))
                                        if idx1.size == 1:
                                            idx1 = np.resize(idx1, (1,))
                                        branch3x3 = m0.branch3x3
                                        branch3x3_new = m1.branch3x3
                                        branch3x3_new[conv_id].weight.data = branch3x3[conv_id].weight.data[idx1.tolist()].clone()
                                        branch3x3_new[conv_id].bias.data = branch3x3[conv_id].bias.data[idx1.tolist()].clone()
                                        branch3x3_new[conv_id].running_mean = branch3x3[conv_id].running_mean[idx1.tolist()].clone()
                                        branch3x3_new[conv_id].running_var = branch3x3[conv_id].running_var[idx1.tolist()].clone()
                                elif isinstance(sub_sub_module[-1], nn.Conv2d):
                                    if conv_id == 0:
                                        if Inception_id_in_cfg == 0:
                                            idx1 = np.squeeze(np.argwhere(np.asarray(mask1.cpu().numpy())))
                                            if idx1.size == 1:
                                                idx1 = np.resize(idx1, (1,))
                                            branch3x3 = m0.branch3x3
                                            branch3x3_new = m1.branch3x3
                                            w1 = branch3x3[conv_id].weight.data[:, :, :, :].clone()
                                            w1 = w1[idx1.tolist(), :, :, :].clone()
                                            w2 = branch3x3[conv_id].bias.data[idx1.tolist()].clone()
                                            branch3x3_new[conv_id].weight.data = w1.clone()
                                            branch3x3_new[conv_id].bias.data = w2.clone()
                                            conv_id += 1
                                        else:
                                            idx0 = np.squeeze(np.argwhere(np.asarray(mask_all[Inception_id_in_cfg-1].cpu().numpy())))
                                            idx1 = np.squeeze(np.argwhere(np.asarray(mask1.cpu().numpy())))
                                            if idx0.size == 1:
                                                idx0 = np.resize(idx0, (1,))
                                            if idx1.size == 1:
                                                idx1 = np.resize(idx1, (1,))
                                            branch3x3 = m0.branch3x3
                                            branch3x3_new = m1.branch3x3
                                            w1 = branch3x3[conv_id].weight.data[:, idx0.tolist(), :, :].clone()
                                            w1 = w1[idx1.tolist(), :, :, :].clone()
                                            w2 = branch3x3[conv_id].bias.data[idx1.tolist()].clone()
                                            branch3x3_new[conv_id].weight.data = w1.clone()
                                            branch3x3_new[conv_id].bias.data = w2.clone()
                                            conv_id += 1

                                    if conv_id == 3:
                                        idx0 = np.squeeze(np.argwhere(np.asarray(mask1.cpu().numpy())))
                                        idx1 = np.squeeze(np.argwhere(np.asarray(mask2.cpu().numpy())))
                                        if idx0.size == 1:
                                            idx0 = np.resize(idx0, (1,))
                                        if idx1.size == 1:
                                            idx1 = np.resize(idx1, (1,))
                                        branch3x3 = m0.branch3x3
                                        branch3x3_new = m1.branch3x3
                                        w1 = branch3x3[conv_id].weight.data[:, idx0.tolist(), :, :].clone()
                                        w1 = w1[idx1.tolist(), :, :, :].clone()
                                        w2 = branch3x3[conv_id].bias.data[idx1.tolist()].clone()
                                        branch3x3_new[conv_id].weight.data = w1.clone()
                                        branch3x3_new[conv_id].bias.data = w2.clone()
                                        conv_id += 1

                        elif sub_name == 'branch5x5':
                            conv_id = 0
                            for sub_sub_module in sub_module.named_modules():
                                if isinstance(sub_sub_module[-1], nn.BatchNorm2d):
                                    if conv_id != 7:
                                        if conv_id == 1:
                                            idx1 = np.squeeze(np.argwhere(np.asarray(mask3.cpu().numpy())))
                                        elif conv_id == 4:
                                            idx1 = np.squeeze(np.argwhere(np.asarray(mask4.cpu().numpy())))

                                        if idx1.size == 1:
                                            idx1 = np.resize(idx1, (1,))
                                        branch5x5 = m0.branch5x5
                                        branch5x5_new = m1.branch5x5
                                        branch5x5_new[conv_id].weight.data = branch5x5[conv_id].weight.data[idx1.tolist()].clone()
                                        branch5x5_new[conv_id].bias.data = branch5x5[conv_id].bias.data[idx1.tolist()].clone()
                                        branch5x5_new[conv_id].running_mean = branch5x5[conv_id].running_mean[idx1.tolist()].clone()
                                        branch5x5_new[conv_id].running_var = branch5x5[conv_id].running_var[idx1.tolist()].clone()
                                        conv_id += 2
                                    else:
                                        branch5x5 = m0.branch5x5
                                        branch5x5_new = m1.branch5x5
                                        branch5x5_new[conv_id].weight.data = branch5x5[conv_id].weight.data.clone()
                                        branch5x5_new[conv_id].bias.data = branch5x5[conv_id].bias.data.clone()
                                        branch5x5_new[conv_id].running_mean = branch5x5[conv_id].running_mean.clone()
                                        branch5x5_new[conv_id].running_var = branch5x5[conv_id].running_var.clone()
                                elif isinstance(sub_sub_module[-1], nn.Conv2d):
                                    if Inception_id_in_cfg == 0:
                                        if conv_id == 0:
                                            idx1 = np.squeeze(np.argwhere(np.asarray(mask3.cpu().numpy())))
                                            if idx1.size == 1:
                                                idx1 = np.resize(idx1, (1,))
                                            branch5x5 = m0.branch5x5
                                            branch5x5_new = m1.branch5x5
                                            w1 = branch5x5[conv_id].weight.data[:, :, :, :].clone()
                                            w1 = w1[idx1.tolist(), :, :, :].clone()
                                            w2 = branch5x5[conv_id].bias.data[idx1.tolist()].clone()
                                            branch5x5_new[conv_id].weight.data = w1.clone()
                                            branch5x5_new[conv_id].bias.data = w2.clone()
                                            conv_id += 1

                                        elif conv_id == 3:
                                            idx0 = np.squeeze(np.argwhere(np.asarray(mask3.cpu().numpy())))
                                            idx1 = np.squeeze(np.argwhere(np.asarray(mask4.cpu().numpy())))
                                            if idx0.size == 1:
                                                idx0 = np.resize(idx0, (1,))
                                            if idx1.size == 1:
                                                idx1 = np.resize(idx1, (1,))
                                            branch5x5 = m0.branch5x5
                                            branch5x5_new = m1.branch5x5
                                            w1 = branch5x5[conv_id].weight.data[:, idx0.tolist(), :, :].clone()
                                            w1 = w1[idx1.tolist(), :, :, :].clone()
                                            w2 = branch5x5[conv_id].bias.data[idx1.tolist()].clone()
                                            branch5x5_new[conv_id].weight.data = w1.clone()
                                            branch5x5_new[conv_id].bias.data = w2.clone()
                                            conv_id += 1

                                        elif conv_id == 6:
                                            idx1 = np.squeeze(np.argwhere(np.asarray(mask4.cpu().numpy())))
                                            if idx1.size == 1:
                                                idx1 = np.resize(idx1, (1,))
                                            branch5x5 = m0.branch5x5
                                            branch5x5_new = m1.branch5x5
                                            w1 = branch5x5[conv_id].weight.data[:, :, :, :].clone()
                                            w1 = w1[:, idx1.tolist(), :, :].clone()
                                            branch5x5_new[conv_id].weight.data = w1.clone()
                                            conv_id += 1
                                    else:
                                        if conv_id == 0:
                                            idx0 = np.squeeze(np.argwhere(np.asarray(mask_all[Inception_id_in_cfg-1].cpu().numpy())))
                                            idx1 = np.squeeze(np.argwhere(np.asarray(mask3.cpu().numpy())))
                                            if idx1.size == 1:
                                                idx1 = np.resize(idx1, (1,))
                                            branch5x5 = m0.branch5x5
                                            branch5x5_new = m1.branch5x5
                                            w1 = branch5x5[conv_id].weight.data[:, idx0.tolist(), :, :].clone()
                                            w1 = w1[idx1.tolist(), :, :, :].clone()
                                            w2 = branch5x5[conv_id].bias.data[idx1.tolist()].clone()
                                            branch5x5_new[conv_id].weight.data = w1.clone()
                                            branch5x5_new[conv_id].bias.data = w2.clone()
                                            conv_id += 1

                                        elif conv_id == 3:
                                            idx0 = np.squeeze(np.argwhere(np.asarray(mask3.cpu().numpy())))
                                            idx1 = np.squeeze(np.argwhere(np.asarray(mask4.cpu().numpy())))
                                            if idx0.size == 1:
                                                idx0 = np.resize(idx0, (1,))
                                            if idx1.size == 1:
                                                idx1 = np.resize(idx1, (1,))
                                            branch5x5 = m0.branch5x5
                                            branch5x5_new = m1.branch5x5
                                            w1 = branch5x5[conv_id].weight.data[:, idx0.tolist(), :, :].clone()
                                            w1 = w1[idx1.tolist(), :, :, :].clone()
                                            w2 = branch5x5[conv_id].bias.data[idx1.tolist()].clone()
                                            branch5x5_new[conv_id].weight.data = w1.clone()
                                            branch5x5_new[conv_id].bias.data = w2.clone()
                                            conv_id += 1

                                        elif conv_id == 6:
                                            idx1 = np.squeeze(np.argwhere(np.asarray(mask4.cpu().numpy())))
                                            if idx1.size == 1:
                                                idx1 = np.resize(idx1, (1,))
                                            branch5x5 = m0.branch5x5
                                            branch5x5_new = m1.branch5x5
                                            w1 = branch5x5[conv_id].weight.data[:, :, :, :].clone()
                                            w1 = w1[:, idx1.tolist(), :, :].clone()
                                            branch5x5_new[conv_id].weight.data = w1.clone()
                                            conv_id += 1

                        elif sub_name == 'branch_pool':
                            conv_id = 1
                            for sub_sub_module in sub_module.named_modules():
                                if isinstance(sub_sub_module[-1], nn.BatchNorm2d):
                                    branch_pool = m0.branch_pool
                                    branch_pool_new = m1.branch_pool
                                    branch_pool_new[conv_id].weight.data = branch_pool[conv_id].weight.data.clone()
                                    branch_pool_new[conv_id].bias.data = branch_pool[conv_id].bias.data.clone()
                                    branch_pool_new[conv_id].running_mean = branch_pool[conv_id].running_mean.clone()
                                    branch_pool_new[conv_id].running_var = branch_pool[conv_id].running_var.clone()

                                    Inception_id_in_cfg += 1
                                    print(Inception_id_in_cfg)
                                    if ((Inception_id_in_cfg * 4) < len(cfg_mask)):
                                        mask1 = cfg_mask[0 + Inception_id_in_cfg * 4]
                                        mask2 = cfg_mask[1 + Inception_id_in_cfg * 4]
                                        mask3 = cfg_mask[2 + Inception_id_in_cfg * 4]
                                        mask4 = cfg_mask[3 + Inception_id_in_cfg * 4]
                                elif isinstance(sub_sub_module[-1], nn.Conv2d):
                                    if Inception_id_in_cfg == 0:
                                        branch_pool = m0.branch_pool
                                        branch_pool_new = m1.branch_pool
                                        w1 = branch_pool[conv_id].weight.data[:, :, :, :].clone()
                                        w1 = w1[:, :, :, :].clone()
                                        branch_pool_new[conv_id].weight.data = w1.clone()
                                        conv_id += 1
                                    else:
                                        idx1 = np.squeeze(np.argwhere(np.asarray(mask_all[Inception_id_in_cfg-1].cpu().numpy())))
                                        if idx1.size == 1:
                                            idx1 = np.resize(idx1, (1,))
                                        branch_pool = m0.branch_pool
                                        branch_pool_new = m1.branch_pool
                                        w1 = branch_pool[conv_id].weight.data[:, :, :, :].clone()
                                        w1 = w1[:, idx1.tolist(), :, :].clone()
                                        branch_pool_new[conv_id].weight.data = w1.clone()
                                        conv_id += 1
                if isinstance(m0, nn.Linear):
                    idx0 = np.squeeze(np.argwhere(np.asarray(mask_all[-1].cpu().numpy())))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    m1.weight.data = m0.weight.data[:, idx0].clone()
                    m1.bias.data = m0.bias.data.clone()
        print('Done!')
        optimizer = get_optimizer(args, model_flops)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = args.lr_decay_step, gamma = 0.1)

    else:
        checkpoint_r = torch.load(args.resume)
        if args.arch == 'vgg_cifar':
            model_flops = import_module(f'model.{args.arch}').VGG_sparse_flops(args.cfg, layer_cfg = checkpoint_r['layer_cfg']).to(device)
        elif args.arch == 'googlenet_cifar':
            model_flops = import_module(f'model.{args.arch}').googlenet_flops(layer_cfg = checkpoint_r['layer_cfg']).to(device)
        start_epoch = checkpoint_r['epoch']
        best_acc = checkpoint_r['best_acc']
        model_flops.load_state_dict(checkpoint_r['state_dict'][0])
        optimizer = get_optimizer(args, model_flops)
        optimizer.load_state_dict(checkpoint_r['optimizer'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = args.lr_decay_step, gamma = 0.1)
        scheduler.load_state_dict(checkpoint_r['scheduler'])
        logger.info('==> loaded checkpoint {} (epoch {}) Prec1: {:f}'.format(args.resume, checkpoint_r['epoch'], float(best_acc)))

    model_flops = model_flops.to(device)
    if len(args.gpus) != 1:
        model_flops = nn.DataParallel(model_flops, device_ids = args.gpus)

    if args.test_only:
        test(model_flops, loader.testLoader)
    else:
        print('==> Start Train...')
        for epoch in range(start_epoch, args.num_epochs):
            train(model_flops, optimizer, loader.trainLoader, args, epoch)
            scheduler.step()
            test_acc = test(model_flops, loader.testLoader)

            is_best = best_acc < test_acc
            best_acc = max(best_acc, test_acc)

            model_state_dict = model_flops.module.state_dict() if len(args.gpus) > 1 else model_flops.state_dict(),

            state = {
                'state_dict': model_state_dict,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'layer_cfg': cfg2 if args.arch == 'vgg_cifar' else layer_cfg
            }
            checkpoint.save_model(state, epoch + 1, is_best)

        logger.info('Best accurary: {:.2f}'.format(float(best_acc)))
        logger.info('--------------------------------------------------------------------------------------------------------------')

if __name__ == '__main__':
    main()
