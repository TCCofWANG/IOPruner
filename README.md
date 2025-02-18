# IOPruner

Channel pruning has become one of the most effective tools widely used for model compression. 
Most off-the-shelf methods depend on manually crafted importance criteria for filters, channels, and other pruning structures, leading to heavy computational burdens and significant reliance on expert experience.
In this paper, we try to overcome this problem by proposing two novel channel pruning methods termed CUR-Pruner and LSH-Pruner.
Specifically, for both of them, we introduce the concept of block partitioning into structured pruning, collaboratively and flexibly pruning input and output channels. 
This allow us to further expand the parameter selection space in the pruning process, helping to mitigate the impact on model performance.
For CUR-Pruner, guided by CUR decomposition, we establish sparse optimization problems so as to achieve the selection of blocks in the original weight matrix, thereby achieving structural pruning.
For LSH-Pruner, we achieve block selection by applying locality-sensitive hashing, a fast algorithm for solving approximate nearest-neighbor search problems in high-dimensional spaces.
We demonstrate the superior performance of the proposed methods on multiple benchmark networks such as VGGNet, GoogleNet, and ResNet in the image classification task. 

## Code Structure
```
├───data  
│       ├───cifar10.py                          (CIFAR-10 dataset)   
│       ├───imagenet.py                         (ImageNet dataset)  
├───model  
│       ├───googlenet_cifar.py                  (googlenet)
│       ├───resnet.py                           (resnet)
│       ├───vgg_cifar.py                        (vggnet)
├───utils  
│       ├───builder.py                          (Build conv layers)
│       ├───common.py                           (Logger functions, etc)
│       ├───conv_matrix.py                      (Prune methods)
│       ├───get_params_flops.py                 (Get parameters Flops of the original model and pruned model)
│       ├───options.py                          (Parameter settings)
main_cifar10.py                                 (Prune and fine-tune on CIFAR-10)
main_imagenet.py                                (Prune and fine-tune on ImageNet)
```

## Dataset
### CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
### ImageNet Dataset: https://image-net.org/challenges/LSVRC/2012
