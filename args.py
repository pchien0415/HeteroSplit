import argparse
import datetime
import os

def modify_args(args):
    if args.use_gpu and args.gpu_idx:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx

    if args.use_valid:
        args.splits = ['train', 'val', 'test']
    # else:
    #     args.splits = ['train', 'test']

    if args.data == 'cifar10':
        args.num_classes = 10
        args.image_size = (32, 32)
    elif args.data == 'svhn':
        args.num_classes = 10
        args.image_size = (32, 32)
    elif args.data == 'cifar100':
        args.num_classes = 100
        args.image_size = (32, 32)
    elif args.data == 'fmnist':
        args.num_classes = 100
        args.image_size = (28, 28)
    else:
        raise NotImplementedError

    if not hasattr(args, "save_path") or args.save_path is None:
        args.save_path = f"outputs/{args.arch}_{args.data}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.num_clients}_{args.num_rounds}_{args.sample_rate}_{args.alpha}"

    return args



arg_parser = argparse.ArgumentParser()

# experiment related
exp_group = arg_parser.add_argument_group('exp', 'experiment setting')
exp_group.add_argument('--seed', default=793, type=int, help='random seed')
exp_group.add_argument('--use_gpu', default=1, type=int, help='Use CPU if zero')
exp_group.add_argument('--gpu_idx', default=None, type=str, help='Index of available GPU')

# dataset related
data_group = arg_parser.add_argument_group('data', 'dataset setting')
data_group.add_argument('--data', metavar='D', default='cifar10',
                        choices=['svhn', 'cifar10', 'cifar100'], help='data to work on')
data_group.add_argument('--data-root', metavar='DIR', default='data',
                        help='path to dataset (default: data)')
data_group.add_argument('--use_valid', default=1, 
                        help='use validation set or not')
data_group.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 0)')

# model arch related
arch_group = arg_parser.add_argument_group('arch', 'model architecture setting')
arch_group.add_argument('--arch', '-a', metavar='ARCH', default='resnet20_3',
                        type=str, help='model architecture')
arch_group.add_argument('--model_BB', type=int, nargs='*', default=[3, 3, 3])
arch_group.add_argument('--ee_locs', type=int, nargs='*', default=[3, 6], help='ee locations')
arch_group.add_argument('--num_exits', type=int, default=3, help='ee locations')

# training related
optim_group = arg_parser.add_argument_group('optimization', 'optimization setting')
optim_group.add_argument('-b', '--batch_size', type=int, default=128, help='mini-batch size')
optim_group.add_argument('-n_epoch', '--num_epoch', type=int, default=5, help='local epoch')
optim_group.add_argument('-lr', '--lr', type=float, default=0.1, help='learning rate')
optim_group.add_argument('-lr_type', '--lr_type', type=str, default='multistep', help='learning rate')
optim_group.add_argument('-decay_rate', '--decay_rate', type=float, default=0.1, help='learning rate decay')
optim_group.add_argument('--decay_rounds', type=float, nargs='*', default=[250, 325])
optim_group.add_argument('-weight_decay', '--weight_decay', type=float, default=5e-4, help='learning rate decay')
optim_group.add_argument('-momentum', '--momentum', type=float, default=0, help='learning rate decay')
optim_group.add_argument('-algo', '--algo', default='propose2', help='algo')

optim_group.add_argument('--KD', type=float, default=0, help='KD')
optim_group.add_argument('--KD_gamma', type=float, default=1, help='KD gamma')
optim_group.add_argument('--KD_T', type=int, default=1, help='KD T')



# FL related
fl_group = arg_parser.add_argument_group('fl', 'FL setting')
fl_group.add_argument('--vertical_scale_ratios', type=float, nargs='*', default=[1, 1, 1],
                      help='model split ratio vertically for each complexity level')
fl_group.add_argument('--horizontal_scale_ratios', type=int, nargs='*', default=[1, 2, 3],
                      help='model horizontal split indices for each complexity level')
fl_group.add_argument('--client_split_ratios', type=float, nargs='*', default=[0.4, 0.3, 0.3],
                      help='client ratio at each complexity level')
fl_group.add_argument('--num_rounds', type=int, default=400,
                      help='number of rounds')
fl_group.add_argument('--num_clients', type=int, default=100,
                      help='number of clients')
fl_group.add_argument('--sample_rate', type=float, default=0.2,
                      help='client sample rate')
fl_group.add_argument('--alpha', type=float, default=100,
                      help='data nonIID alpha')
fl_group.add_argument('-trs', '--track_running_stats', action='store_true',
                      help='trs')

# nargs='*': Can accept 0 to many values.