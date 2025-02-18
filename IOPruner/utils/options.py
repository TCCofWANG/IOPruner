import argparse
import ast
import os

parser = argparse.ArgumentParser(description='Prune via Input and Output channel')

'''Normal Settings'''
parser.add_argument('--gpus',type=int,nargs='+',default=[0,1],help='Select gpu_id to use. default:[0]',)
parser.add_argument('--data_set',type=str,default='cifar10',help='Select dataset to train. default:cifar10',)
parser.add_argument('--data_path',type=str,default='',help='The dictionary where the input is stored. default:',)
parser.add_argument('--job_dir',type=str,default='',help='The directory where the summaries will be stored. default:./experiments')
parser.add_argument('--reset',action='store_true',default=False,help='reset the job-dir directory?')
parser.add_argument('--resume',type=str,default=None,help='Load the model from the specified checkpoint')
parser.add_argument('--refine',type=str,default=None,help='Path to the model to be fine-tuned')

'''Training Settings'''
parser.add_argument('--arch',type=str,default='vgg_cifar',help='Architecture of model. default:vgg_cifar')
parser.add_argument('--cfg',type=str,default='vgg16',help='Detail architecuture of model. default:vgg16')
parser.add_argument('--num_epochs',type=int,default=150,help='The num of epochs to train. default:150')
parser.add_argument('--train_batch_size',type=int,default=256,help='Batch size for training. default:256')
parser.add_argument('--eval_batch_size',type=int,default=256,help='Batch size for validation. default:256')
parser.add_argument('--momentum',type=float,default=0.9,help='Momentum for MomentumOptimizer. default:0.9')
parser.add_argument('--lr',type=float,default=0.01,help='Learning rate for train. default:0.01')
parser.add_argument('--lr_decay_step',type=int,nargs='+',default=[50,100],help='the iterval of learn rate decay. default:[50,100]')
parser.add_argument('--weight_decay',type=float,default=5e-3,help='The weight decay of loss. default:5e-3')
parser.add_argument('--test_only',default = False,action='store_true',help='Test only?')
parser.add_argument('--pretrain_model',type=str,default='',help='Path to the pretrain model . default:None')
parser.add_argument('--sparse_model',type=str,default='',help='Load the sparse model after training')

parser.add_argument('--lamda', type=float, default = 0.02, help = 'CD hyper-parameters')
parser.add_argument('--max_iterations', type=int, default = 200, help = 'CD hyper-parameters')
parser.add_argument('--bucket_num', type=int, default = 20, help = 'LSH hyper-parameters')
parser.add_argument('--block_size', type=int, default =4, help = 'The size of the cut along the input and output channels')
parser.add_argument("--conv_type",type=str,default='LSHBlockConv',help="Conv type of conv layer. "
                     "Option: CDBlockConv,LSHBlockConv")
parser.add_argument("--first-layer-type",type =str,default = None,help = "Conv type of first layer")

args = parser.parse_args()

if args.resume is not None and not os.path.isfile(args.resume):
    raise ValueError('No checkpoint found at {} to resume'.format(args.resume))
if args.refine is not None and not os.path.isfile(args.refine):
    raise ValueError('No checkpoint found at {} to refine'.format(args.refine))

