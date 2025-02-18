import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch

class Data:
    def __init__(self, args):
        traindir = os.path.join(args.data_path, 'train')
        valdir = os.path.join(args.data_path, 'val')

        trainset = datasets.ImageFolder(
            traindir,
                transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                         std = [0.229, 0.224, 0.225]),
                                    ]))

        self.trainLoader = DataLoader(
            trainset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True)

        testset = datasets.ImageFolder(
            valdir,
            transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                     std = [0.229, 0.224, 0.225])
                                ]))

        self.testLoader = DataLoader(
            testset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True)
