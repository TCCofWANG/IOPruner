from torch.nn import init
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import math
from utils.options import args as parser_args

import numpy as np
import pdb
import pandas as pd
import random

def coordinate_descent(X, lambda_val, max_iterations):
    n, p = X.shape
    W = torch.zeros(p,p)
    for i in range(p):
        norm = torch.norm(X[:, i], p = 2)
        X[:, i] = X[:, i] / norm
    for iteration in range(max_iterations):
        for i in range(p):
            Xi = X[:, i]
            sum_XW = torch.zeros(n, p)
            for j in range(p):
                if j != i:
                    sum_XW += X[:, j].unsqueeze(1) @ W[j, :].unsqueeze(0)
            Zi = Xi.unsqueeze(1).T @ (X - sum_XW)
            Zi_norm = torch.norm(Zi, p=2)
            Wi_update_value = 1 - lambda_val / Zi_norm

            if Wi_update_value > 0:
                W[i, :] = Wi_update_value * Zi
            else:
                W[i, :] = torch.zeros(1,p)
    return W


def LSH_Random_Projection(x, random_matrix):
    projected_x = torch.matmul(x, random_matrix)
    rotated_vecs = torch.cat([projected_x, -projected_x], dim = -1)
    bucket_index = torch.argmax(rotated_vecs, dim = -1)
    return bucket_index


# CD
class CDBlockConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = nn.Parameter(torch.ones(self.weight.shape), requires_grad = False)

    def forward(self, x):
        sparseWeight = self.mask * self.weight
        x = F.conv2d(
                x, sparseWeight, self.bias, self.stride, self.padding, self.dilation, self.groups
                )
        return x

    def get_mask_inout_vgg(self,args):
        w = self.weight.detach().cpu()
        c_out, c_in, k_1, k_2 = w.shape

        w_out = torch.zeros(int(c_out/args.block_size),int(c_in/args.block_size))
        for i in range(w_out.size(0)):
            for j in range(w_out.size(1)):
                sub_tensor = w[i * args.block_size:(i + 1) * args.block_size, j * args.block_size:(j + 1) * args.block_size, :, :]
                l1_norm = sub_tensor.abs().sum()
                w_out[i, j] = l1_norm

        W_out = coordinate_descent(w_out.T,lambda_val = args.lamda, max_iterations = args.max_iterations)
        W_in = coordinate_descent(w_out,lambda_val = args.lamda, max_iterations = args.max_iterations)

        W_out_l1 = torch.sum(torch.abs(W_out),1)
        W_in_l1 = torch.sum(torch.abs(W_in),1)

        indice_out_pres = torch.nonzero(W_out_l1)
        indice_in_pres = torch.nonzero(W_in_l1)

        indice_out_pres_re = [range(args.block_size*i,args.block_size*i+args.block_size) for i in indice_out_pres]
        flattened_list_out = [item for sublist in indice_out_pres_re for item in sublist]

        indice_in_pres_re = [range(args.block_size*j,args.block_size*j+args.block_size) for j in indice_in_pres]
        flattened_list_in = [item for sublist in indice_in_pres_re for item in sublist]

        indice_to_keep_out = [i for i in range(c_out) if i in flattened_list_out]
        print(indice_to_keep_out)
        indice_to_keep_in = [j for j in range(c_in) if j in flattened_list_in]
        print(indice_to_keep_in)

        m_all = torch.zeros(c_out, c_in, k_1, k_2)
        for i in indice_to_keep_out:
            for j in indice_to_keep_in:
                m_all[i, j, :, :] = 1
        self.mask = nn.Parameter(m_all, requires_grad = False)

        return indice_to_keep_out, indice_to_keep_in

    def get_mask_out_vgg(self, indice_to_keep_in):
        w = self.weight.detach().cpu()
        c_out, c_in, k_1, k_2 = w.shape
        m_all = torch.zeros(c_out, c_in, k_1, k_2)
        for i in indice_to_keep_in:
            m_all[i, :, :, :] = 1
        self.mask = nn.Parameter(m_all, requires_grad = False)

    def get_mask_in_vgg(self, indice_to_keep_out):
        w = self.weight.detach().cpu()
        c_out, c_in, k_1, k_2 = w.shape
        if c_in != 3:  # skip first conv
            m_out = self.mask.detach()
            m_reshape = m_out.contiguous().view(-1, c_in * k_1 * k_2)
            u = torch.sum(torch.abs(m_reshape), 1)
            out_indices_keep = torch.nonzero(u)
            prune_indices = [i for i in range(m_out.size(0)) if i in out_indices_keep]

            m_all = torch.zeros(c_out, c_in, k_1, k_2)
            for i in prune_indices:
                for j in indice_to_keep_out:
                    m_all[i, j, :, :] = 1
            self.mask = nn.Parameter(m_all, requires_grad = False)

    def get_mask_inout_resnet_imagenet(self,args):
        w = self.weight.detach().cpu()
        c_out, c_in, k_1, k_2 = w.shape

        w_out = torch.zeros(int(c_out/args.block_size),int(c_in/args.block_size))
        for i in range(w_out.size(0)):
            for j in range(w_out.size(1)):
                sub_tensor = w[i * args.block_size:(i + 1) * args.block_size, j * args.block_size:(j + 1) * args.block_size, :, :]
                l1_norm = sub_tensor.abs().sum()
                w_out[i, j] = l1_norm
        W_out = coordinate_descent(w_out.T,lambda_val = args.lamda, max_iterations = args.max_iterations)
        W_in = coordinate_descent(w_out,lambda_val = args.lamda, max_iterations = args.max_iterations)

        W_out_l1 = torch.sum(torch.abs(W_out),1)
        W_in_l1 = torch.sum(torch.abs(W_in),1)

        indice_out_pres = torch.nonzero(W_out_l1)
        indice_in_pres = torch.nonzero(W_in_l1)

        indice_out_pres_re = [range(args.block_size*i,args.block_size*i+args.block_size) for i in indice_out_pres]
        flattened_list_out = [item for sublist in indice_out_pres_re for item in sublist]

        indice_in_pres_re = [range(args.block_size*j,args.block_size*j+args.block_size) for j in indice_in_pres]
        flattened_list_in = [item for sublist in indice_in_pres_re for item in sublist]

        indice_to_keep_out = [i for i in range(c_out) if i in flattened_list_out]
        print(indice_to_keep_out)
        indice_to_keep_in = [j for j in range(c_in) if j in flattened_list_in]
        print(indice_to_keep_in)

        m_all = torch.zeros(c_out, c_in, k_1, k_2)
        for i in indice_to_keep_out:
            for j in indice_to_keep_in:
                m_all[i, j, :, :] = 1
        self.mask = nn.Parameter(m_all, requires_grad = False)

        return indice_to_keep_out, indice_to_keep_in

    def get_mask_out_resnet_imagenet(self, indice_to_keep_in):
        w = self.weight.detach().cpu()
        c_out, c_in, k_1, k_2 = w.shape
        m_all = torch.zeros(c_out, c_in, k_1, k_2)
        for i in indice_to_keep_in:
            m_all[i, :, :, :] = 1
        self.mask = nn.Parameter(m_all, requires_grad = False)

    def get_mask_in_resnet_imagenet(self, indice_to_keep_out):
        w = self.weight.detach().cpu()
        c_out, c_in, k_1, k_2 = w.shape
        m_all = torch.zeros(c_out, c_in, k_1, k_2)
        for j in indice_to_keep_out:
            m_all[:, j, :, :] = 1
        self.mask = nn.Parameter(m_all, requires_grad = False)

    def get_mask_inout_googlenet(self, args):
        w = self.weight.detach().cpu()
        c_out, c_in, k_1, k_2 = w.shape

        w_out = torch.zeros(int(c_out / args.block_size), int(c_in / args.block_size))
        for i in range(w_out.size(0)):
            for j in range(w_out.size(1)):
                sub_tensor = w[i * args.block_size:(i + 1) * args.block_size,
                             j * args.block_size:(j + 1) * args.block_size, :, :]
                l1_norm = sub_tensor.abs().sum()
                w_out[i, j] = l1_norm
        W_out = coordinate_descent(w_out.T, lambda_val = args.lamda, max_iterations = args.max_iterations)
        W_in = coordinate_descent(w_out, lambda_val = args.lamda, max_iterations = args.max_iterations)

        W_out_l1 = torch.sum(torch.abs(W_out), 1)
        W_in_l1 = torch.sum(torch.abs(W_in), 1)

        indice_out_pres = torch.nonzero(W_out_l1)
        indice_in_pres = torch.nonzero(W_in_l1)

        indice_out_pres_re = [range(args.block_size * i, args.block_size * i + args.block_size) for i in
                              indice_out_pres]
        flattened_list_out = [item for sublist in indice_out_pres_re for item in sublist]

        indice_in_pres_re = [range(args.block_size * j, args.block_size * j + args.block_size) for j in indice_in_pres]
        flattened_list_in = [item for sublist in indice_in_pres_re for item in sublist]

        indice_to_keep_out = [i for i in range(c_out) if i in flattened_list_out]
        print(indice_to_keep_out)
        indice_to_keep_in = [j for j in range(c_in) if j in flattened_list_in]
        print(indice_to_keep_in)

        m_all = torch.zeros(c_out, c_in, k_1, k_2)
        for i in indice_to_keep_out:
            for j in indice_to_keep_in:
                m_all[i, j, :, :] = 1
        self.mask = nn.Parameter(m_all, requires_grad = False)

        return indice_to_keep_out, indice_to_keep_in

    def get_mask_out_googlenet(self, indice_to_keep_in):
        w = self.weight.detach().cpu()
        c_out, c_in, k_1, k_2 = w.shape
        m_all = torch.zeros(c_out, c_in, k_1, k_2)
        for i in indice_to_keep_in:
            m_all[i, :, :, :] = 1
        self.mask = nn.Parameter(m_all, requires_grad = False)

    def get_mask_in_googlenet(self, indice_to_keep_out):
        w = self.weight.detach().cpu()
        c_out, c_in, k_1, k_2 = w.shape
        m_all = torch.zeros(c_out, c_in, k_1, k_2)
        for j in indice_to_keep_out:
            m_all[:, j, :, :] = 1
        self.mask = nn.Parameter(m_all, requires_grad = False)

# LSH
class LSHBlockConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = nn.Parameter(torch.ones(self.weight.shape), requires_grad = False)

    def forward(self, x):
        sparseWeight = self.mask * self.weight
        x = F.conv2d(
                x, sparseWeight, self.bias, self.stride, self.padding, self.dilation, self.groups
                )
        return x

    def get_mask_inout_vgg(self, args):
        w = self.weight.detach().cpu()
        c_out, c_in, k_1, k_2 = w.shape

        w_out = torch.zeros(int(c_out / args.block_size), int(c_in / args.block_size))
        random_matrix1 = torch.randn(w[0 * args.block_size:(0 + 1) * args.block_size, :, :, :].reshape(-1).shape[0],
                                     int(args.bucket_num / 2))
        LSH_out_index = []
        LSH_in_index = []
        for i in range(w_out.size(0)):
            sub_tensor = []
            for j in range(w_out.size(1)):
                sub_tensor.append(
                    w[i * args.block_size:(i + 1) * args.block_size, j * args.block_size:(j + 1) * args.block_size, :,
                    :].reshape(-1))
            b_index_out = LSH_Random_Projection(torch.cat(sub_tensor), random_matrix1)
            LSH_out_index.append(b_index_out)

        random_matrix2 = torch.randn(w[:, 0 * args.block_size:(0 + 1) * args.block_size, :, :].reshape(-1).shape[0],
                                     int(args.bucket_num / 2))
        for j in range(w_out.size(1)):
            sub_tensor_in = []
            for i in range(w_out.size(0)):
                sub_tensor_in.append(
                    w[i * args.block_size:(i + 1) * args.block_size, j * args.block_size:(j + 1) * args.block_size, :,
                    :].reshape(-1))
            b_index_in = LSH_Random_Projection(torch.cat(sub_tensor_in), random_matrix2)
            LSH_in_index.append(b_index_in)

        LSH_out_index_t = torch.tensor([t.item() for t in LSH_out_index])
        LSH_in_index_t = torch.tensor([u.item() for u in LSH_in_index])
        out_random_indices = np.random.permutation(len(LSH_out_index))
        in_random_indices = np.random.permutation(len(LSH_in_index))
        LSH_out_index_random = LSH_out_index_t[out_random_indices].numpy()
        LSH_in_index_random = LSH_in_index_t[in_random_indices].numpy()
        s_out = pd.Series(LSH_out_index_random)
        s_in = pd.Series(LSH_in_index_random)
        unique_series_out = s_out.drop_duplicates(keep = 'last')
        unique_series_in = s_in.drop_duplicates(keep = 'last')
        unique_series_out_index = unique_series_out.index.tolist()
        unique_series_in_index = unique_series_in.index.tolist()
        indice_out_pres_re = [range(args.block_size * i, args.block_size * i + args.block_size) for i in
                              out_random_indices[unique_series_out_index]]
        flattened_list_out = [item for sublist in indice_out_pres_re for item in sublist]
        indice_in_pres_re = [range(args.block_size * j, args.block_size * j + args.block_size) for j in
                             in_random_indices[unique_series_in_index]]
        flattened_list_in = [item for sublist in indice_in_pres_re for item in sublist]
        indice_to_keep_out = [i for i in range(c_out) if i in flattened_list_out]
        indice_to_keep_in = [j for j in range(c_in) if j in flattened_list_in]

        m_all = torch.zeros(c_out, c_in, k_1, k_2)
        for i in indice_to_keep_out:
            for j in indice_to_keep_in:
                m_all[i, j, :, :] = 1
        self.mask = nn.Parameter(m_all, requires_grad = False)

        return indice_to_keep_out, indice_to_keep_in

    def get_mask_out_vgg(self, indice_to_keep_in):
        w = self.weight.detach().cpu()
        c_out, c_in, k_1, k_2 = w.shape
        m_all = torch.zeros(c_out, c_in, k_1, k_2)
        for i in indice_to_keep_in:
            m_all[i, :, :, :] = 1
        self.mask = nn.Parameter(m_all, requires_grad = False)

    def get_mask_in_vgg(self, indice_to_keep_out):
        w = self.weight.detach().cpu()
        c_out, c_in, k_1, k_2 = w.shape
        if c_in != 3:  # skip first conv
            m_out = self.mask.detach()
            m_reshape = m_out.contiguous().view(-1, c_in * k_1 * k_2)
            u = torch.sum(torch.abs(m_reshape), 1)
            out_indices_keep = torch.nonzero(u)
            prune_indices = [i for i in range(m_out.size(0)) if i in out_indices_keep]

            m_all = torch.zeros(c_out, c_in, k_1, k_2)
            for i in prune_indices:
                for j in indice_to_keep_out:
                    m_all[i, j, :, :] = 1
            self.mask = nn.Parameter(m_all, requires_grad = False)

    def get_mask_inout_googlenet(self, args):
        w = self.weight.detach().cpu()
        c_out, c_in, k_1, k_2 = w.shape

        w_out = torch.zeros(int(c_out / args.block_size), int(c_in / args.block_size))
        random_matrix1 = torch.randn(w[0 * args.block_size:(0 + 1) * args.block_size, :, :, :].reshape(-1).shape[0],
                                     int(args.bucket_num / 2))
        LSH_out_index = []
        LSH_in_index = []
        for i in range(w_out.size(0)):
            sub_tensor = []
            for j in range(w_out.size(1)):
                sub_tensor.append(w[i * args.block_size:(i + 1) * args.block_size, j * args.block_size:(j + 1) * args.block_size, :, :].reshape(-1))
            b_index_out = LSH_Random_Projection(torch.cat(sub_tensor), random_matrix1)
            LSH_out_index.append(b_index_out)


        random_matrix2 = torch.randn(w[:, 0 * args.block_size:(0 + 1) * args.block_size, :, :].reshape(-1).shape[0],int(args.bucket_num / 2))
        for j in range(w_out.size(1)):
            sub_tensor_in = []
            for i in range(w_out.size(0)):
                sub_tensor_in.append(w[i * args.block_size:(i + 1) * args.block_size, j * args.block_size:(j + 1) * args.block_size, :,:].reshape(-1))
            b_index_in = LSH_Random_Projection(torch.cat(sub_tensor_in), random_matrix2)
            LSH_in_index.append(b_index_in)

        LSH_out_index_t = torch.tensor([t.item() for t in LSH_out_index])
        LSH_in_index_t = torch.tensor([u.item() for u in LSH_in_index])
        out_random_indices = np.random.permutation(len(LSH_out_index))
        in_random_indices = np.random.permutation(len(LSH_in_index))
        LSH_out_index_random = LSH_out_index_t[out_random_indices].numpy()
        LSH_in_index_random = LSH_in_index_t[in_random_indices].numpy()
        s_out = pd.Series(LSH_out_index_random)
        s_in = pd.Series(LSH_in_index_random)
        unique_series_out = s_out.drop_duplicates(keep = 'last')
        unique_series_in = s_in.drop_duplicates(keep = 'last')
        unique_series_out_index = unique_series_out.index.tolist()
        unique_series_in_index = unique_series_in.index.tolist()
        indice_out_pres_re = [range(args.block_size * i, args.block_size * i + args.block_size) for i in
                              out_random_indices[unique_series_out_index]]
        flattened_list_out = [item for sublist in indice_out_pres_re for item in sublist]
        indice_in_pres_re = [range(args.block_size * j, args.block_size * j + args.block_size) for j in
                             in_random_indices[unique_series_in_index]]
        flattened_list_in = [item for sublist in indice_in_pres_re for item in sublist]
        indice_to_keep_out = [i for i in range(c_out) if i in flattened_list_out]
        indice_to_keep_in = [j for j in range(c_in) if j in flattened_list_in]

        m_all = torch.zeros(c_out, c_in, k_1, k_2)
        for i in indice_to_keep_out:
            for j in indice_to_keep_in:
                m_all[i, j, :, :] = 1
        self.mask = nn.Parameter(m_all, requires_grad = False)

        return indice_to_keep_out, indice_to_keep_in

    def get_mask_out_googlenet(self, indice_to_keep_in):
        w = self.weight.detach().cpu()
        c_out, c_in, k_1, k_2 = w.shape
        m_all = torch.zeros(c_out, c_in, k_1, k_2)
        for i in indice_to_keep_in:
            m_all[i, :, :, :] = 1
        self.mask = nn.Parameter(m_all, requires_grad = False)

    def get_mask_in_googlenet(self, indice_to_keep_out):
        w = self.weight.detach().cpu()
        c_out, c_in, k_1, k_2 = w.shape
        if c_in != 3:  # skip first conv
            m_out = self.mask.detach()
            m_reshape = m_out.contiguous().view(-1, c_in * k_1 * k_2)
            u = torch.sum(torch.abs(m_reshape), 1)
            out_indices_keep = torch.nonzero(u)
            prune_indices = [i for i in range(m_out.size(0)) if i in out_indices_keep]

            m_all = torch.zeros(c_out, c_in, k_1, k_2)
            for i in prune_indices:
                for j in indice_to_keep_out:
                    m_all[i, j, :, :] = 1
            self.mask = nn.Parameter(m_all, requires_grad = False)

    def get_mask_inout_resnet_imagenet(self, args):
        w = self.weight.detach().cpu()
        c_out, c_in, k_1, k_2 = w.shape

        w_out = torch.zeros(int(c_out / args.block_size), int(c_in / args.block_size))
        random_matrix1 = torch.randn(w[0 * args.block_size:(0 + 1) * args.block_size, :, :, :].reshape(-1).shape[0],
                                     int(args.bucket_num / 2))
        LSH_out_index = []
        LSH_in_index = []
        for i in range(w_out.size(0)):
            sub_tensor = []
            for j in range(w_out.size(1)):
                sub_tensor.append(w[i * args.block_size:(i + 1) * args.block_size, j * args.block_size:(j + 1) * args.block_size, :, :].reshape(-1))
            b_index_out = LSH_Random_Projection(torch.cat(sub_tensor), random_matrix1)
            LSH_out_index.append(b_index_out)


        random_matrix2 = torch.randn(w[:, 0 * args.block_size:(0 + 1) * args.block_size, :, :].reshape(-1).shape[0],int(args.bucket_num / 2))
        for j in range(w_out.size(1)):
            sub_tensor_in = []
            for i in range(w_out.size(0)):
                sub_tensor_in.append(w[i * args.block_size:(i + 1) * args.block_size, j * args.block_size:(j + 1) * args.block_size, :,:].reshape(-1))
            b_index_in = LSH_Random_Projection(torch.cat(sub_tensor_in), random_matrix2)
            LSH_in_index.append(b_index_in)

        LSH_out_index_t = torch.tensor([t.item() for t in LSH_out_index])
        LSH_in_index_t = torch.tensor([u.item() for u in LSH_in_index])
        out_random_indices = np.random.permutation(len(LSH_out_index))
        in_random_indices = np.random.permutation(len(LSH_in_index))
        LSH_out_index_random = LSH_out_index_t[out_random_indices].numpy()
        LSH_in_index_random = LSH_in_index_t[in_random_indices].numpy()
        s_out = pd.Series(LSH_out_index_random)
        s_in = pd.Series(LSH_in_index_random)
        unique_series_out = s_out.drop_duplicates(keep = 'last')
        unique_series_in = s_in.drop_duplicates(keep = 'last')
        unique_series_out_index = unique_series_out.index.tolist()
        unique_series_in_index = unique_series_in.index.tolist()
        indice_out_pres_re = [range(args.block_size * i, args.block_size * i + args.block_size) for i in
                              out_random_indices[unique_series_out_index]]
        flattened_list_out = [item for sublist in indice_out_pres_re for item in sublist]
        indice_in_pres_re = [range(args.block_size * j, args.block_size * j + args.block_size) for j in
                             in_random_indices[unique_series_in_index]]
        flattened_list_in = [item for sublist in indice_in_pres_re for item in sublist]
        indice_to_keep_out = [i for i in range(c_out) if i in flattened_list_out]
        indice_to_keep_in = [j for j in range(c_in) if j in flattened_list_in]

        m_all = torch.zeros(c_out, c_in, k_1, k_2)
        for i in indice_to_keep_out:
            for j in indice_to_keep_in:
                m_all[i, j, :, :] = 1
        self.mask = nn.Parameter(m_all, requires_grad = False)

        return indice_to_keep_out, indice_to_keep_in

    def get_mask_out_resnet_imagenet(self, indice_to_keep_in):
        w = self.weight.detach().cpu()
        c_out, c_in, k_1, k_2 = w.shape
        m_all = torch.zeros(c_out, c_in, k_1, k_2)
        for i in indice_to_keep_in:
            m_all[i, :, :, :] = 1
        self.mask = nn.Parameter(m_all, requires_grad = False)

    def get_mask_in_resnet_imagenet(self, indice_to_keep_out):
        w = self.weight.detach().cpu()
        c_out, c_in, k_1, k_2 = w.shape
        if c_in != 3:  # skip first conv
            m_out = self.mask.detach()
            m_reshape = m_out.contiguous().view(-1, c_in * k_1 * k_2)
            u = torch.sum(torch.abs(m_reshape), 1)
            out_indices_keep = torch.nonzero(u)
            prune_indices = [i for i in range(m_out.size(0)) if i in out_indices_keep]

            m_all = torch.zeros(c_out, c_in, k_1, k_2)
            for i in prune_indices:
                for j in indice_to_keep_out:
                    m_all[i, j, :, :] = 1
            self.mask = nn.Parameter(m_all, requires_grad = False)