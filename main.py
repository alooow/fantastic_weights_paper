from __future__ import print_function
import argparse
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import sparselearning
from models import cifar_resnet, initializers, vgg
from models.mlp_cifar10 import MLP_CIFAR10, MLP_CIFAR10_DROPOUT
from data_handling.utils import get_mnist_dataloaders, get_cifar10_dataloaders, get_cifar100_dataloaders, str2bool, get_ImageNet_loaders, get_higgs_dataloaders, get_tinyimagenet_dataloaders, get_kmnist_dataloader, get_fashionmnist_dataloader
from models.conv_cifar10 import SmallConvNet_CIFAR10
from models.mlp_higgs import MLP_Higgs
from models.lenet import LeNet_300_100, LeNet_5_Caffe
from data_handling.logger import *
from trainer import run_testing, run_training, resume, run_eval
from setup_utils import get_mask
from models.imagenet_resnet import build_resnet




def get_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # training_and_eval
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test_batch_size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--multiplier', type=int, default=1, metavar='N',
                        help='extend training time by multiplier times')
    parser.add_argument('--epochs', type=int, default=160, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=str2bool, default="false",
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 0)')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=["adam", "sgd"],
                        help='The optimizer to use. Default: sgd. Options: sgd, adam.')
    parser.add_argument('--data', type=str, default='mnist', choices=["mnist", "cifar10", "higgs"])
    parser.add_argument('--fp16', type=str2bool, default="false", help='Run in fp16 mode.')
    parser.add_argument('--valid_split', type=float, default=0.1)
    parser.add_argument('--resume', type=str)
    # parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--model', type=str, default='', help='model to use. Options: mlp_cifar10, conv_cifar10, mlp_higgs, ...')
    parser.add_argument('--l2', type=float, default=5.0e-4)
    # other
    parser.add_argument('--bench', type=str2bool, default="true",
                        help='Enables the benchmarking of layers and estimates sparse speedups')
    parser.add_argument('--max_threads', type=int, default=10, help='How many threads to use for data loading.')
    # saving and logging
    parser.add_argument('--log_dir', type=str, default='./logs', help='where to store the logs')
    parser.add_argument('--save_dir', type=str, default='./save', help='where to store other results')
    parser.add_argument('--verbose', type=str2bool, default="true", help="toggle the verbose mode")
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_features', type=str2bool, default="false",
                        help='Resumes a saved model and saves its feature data to disk for plotting.')
    # sparse
    parser.add_argument('--scaled', type=str2bool, default="false", help='scale the initialization by 1/density')
    parser.add_argument('--use_wandb', type=str2bool, default="true")
    parser.add_argument('--save_locally', type=str2bool, default="true")
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--tag', type=str, help="experiment type. Used only for wandb")
    parser.add_argument('--opt_order', type=str, choices=["before", "after"], default="before")
    parser.add_argument('--manual_stop', type=str2bool, default="false", help="if true, will automatically stop the "
                                                                              "training after first pruning")
    parser.add_argument("--distributed", type=str2bool, default="false")
    parser.add_argument("--nesterov", type=str2bool, default="true")
    parser.add_argument("--reinit_workers", type=str2bool, default="false")
    parser.add_argument("--workers", type=int, default=4)
    sparselearning.core.add_sparse_args(parser)
    return parser


def main(args):
    logger = setup_logger(args)
    if args.verbose:
        print_and_log(args)

    if args.fp16:
        try:
            from apex.fp16_utils import FP16_Optimizer
        except:
            print('WARNING: apex not installed, ignoring --fp16 option')
            args.fp16 = False


    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if not args.no_cuda:
        assert torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    if args.verbose:
        print_and_log('\n\n')
        print_and_log('=' * 80)

    if args.death not in ["magnitude", "Random", "SET", "SETFixed", "RunningMagnitude"]:
        args.opt_order = "after"
    if args.manual_stop:
        args.opt_order = "after"

    fix_seeds(args)

    output, test_loader, train_loader, valid_loader = get_data(args)
    model = get_model(args, device, output)

    if args.verbose:
        info_beginning(args, model)
        print_and_log(f"Total number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizer = get_optimizer(args, model)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[int(args.epochs / 2) * args.multiplier,
                                                                    int(args.epochs * 3 / 4) * args.multiplier],
                                                        last_epoch=-1)
    if args.resume:
        resume(args, device, model, optimizer, test_loader)
    if args.fp16:
        model, optimizer = setup_fp16(args, model, optimizer)

    mask = get_mask(args, model, optimizer, train_loader, device)
    best_acc = 0.0

    # create output file
    save_subfolder = get_output_file(args)

    run_training(args, best_acc, device, lr_scheduler, mask, model, optimizer, save_subfolder, train_loader,
                 valid_loader, logger)
    run_eval(args, device, model, save_subfolder, test_loader, logger)
    run_testing(args, device, model, save_subfolder, test_loader, logger)


def setup_fp16(args, model, optimizer):
    if args.verbose:
        print('FP16')
    optimizer = FP16_Optimizer(optimizer,
                               static_loss_scale=None,
                               dynamic_loss_scale=True,
                               dynamic_loss_args={'init_scale': 2 ** 16})
    model = model.half()
    return model, optimizer


def get_output_file(args):
    save_path = os.path.join(args.save_dir,
                             os.path.join(str(args.model),
                                          os.path.join(str(args.data),
                                                           os.path.join(str(args.sparse_init), str(args.seed)))))
    if args.sparse:
        save_subfolder = os.path.join(save_path, 'sparsity' + str(1 - args.density))
    else:
        save_subfolder = os.path.join(save_path, 'dense')
    if not os.path.exists(save_subfolder): os.makedirs(save_subfolder)
    return save_subfolder




def get_optimizer(args, model):
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2,
                              nesterov=args.nesterov)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    else:
        print('Unknown optimizer: {0}'.format(args.optimizer))
        raise Exception('Unknown optimizer')
    return optimizer


def get_model(args, device, output):

    last = "logits" if "GraSP" in args.death else "logsoftmax"
    if args.model == 'resnet50' or args.model == 'resnet18':
        last = 'logits'
    if args.data == "higgs":
        last = 'logits'
    print("Last layer output type (data=higgs always uses logits):", last)
    if args.scaled:
        init_type = 'scaled_kaiming_normal'
    else:
        init_type = 'kaiming_normal'
    if 'vgg' in args.model:
        model = vgg.VGG(depth=int(args.model[-2:]), dataset=args.data, batchnorm=True, last=last).to(device)
    elif 'mlp_cifar10' == args.model:
        model = MLP_CIFAR10(last=last).to(device)
    elif 'mlp_fmnist' == args.model:
        model = LeNet_5_Caffe(last=last).to(device)
    elif 'mlp_cifar10_dropout' == args.model:
        model = MLP_CIFAR10_DROPOUT(last=last, density=args.density).to(device)
    elif 'resnet50' == args.model:
        model = build_resnet('resnet50', 'classic').to(device)
        model.last = "logits"
    elif 'resnet18plain' == args.model:
        model = build_resnet('resnet18plain', 'classic').to(device)
        model.last = "logits"
    elif 'resnet18' == args.model:
        import torchvision.models as models
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, output)
        model = model.to(device)
        model.last = "logits"
    elif 'efficientnet-b0-plain' in args.model:
        import torchvision.models as models
        model = models.efficientnet_b0(num_classes=output)
        model = model.to(device)
        model.last = "logits"
    elif 'efficientnet-b3-plain' in args.model:
        import torchvision.models as models
        model = models.efficientnet_b3(num_classes=output)
        model = model.to(device)
        model.last = "logits"
    elif 'efficientnet' in args.model:
        import torchvision.models as models
        if args.model == "efficientnet":
            model = models.efficientnet_b1(models.EfficientNet_B1_Weights, num_classes=1000)
            n_features = model.classifier[1].in_features
            model.classifier = torch.nn.Sequential(model.classifier[0], torch.nn.Linear(n_features, output))
        else:
            model = models.efficientnet_b1(num_classes=output)
        model = model.to(device)
        model.last = "logits"
    elif 'conv_cifar10' in args.model:
        model = SmallConvNet_CIFAR10(last=last).to(device)
    elif args.model == 'mlp_higgs':
        model = MLP_Higgs().to(device)
    elif 'cifar_resnet' in args.model:
        model = cifar_resnet.Model.get_model_from_name(args.model,
                                                       initializers.initializations(init_type, args.density),
                                                       outputs=output, last=last).to(device)
    else:
        raise ValueError("Unknown model {}".format(args.model))
    return model


def get_data(args):
    if args.reinit_workers:
        def _worker_init_fn(id):
            np.random.seed(seed=args.seed + id)
            random.seed(args.seed + id)
        worker_init_fn = _worker_init_fn
    else:
        worker_init_fn = None


    if args.data == 'mnist':
        train_loader, valid_loader, test_loader = get_mnist_dataloaders(args, validation_split=args.valid_split)
        output = 10
    elif args.data == 'cifar10':
        train_loader, valid_loader, test_loader = get_cifar10_dataloaders(args, args.valid_split,
                                                                          max_threads=args.max_threads,
                                                                          worker_init_fn=worker_init_fn)
        output = 10
    elif args.data == 'cifar100':
        train_loader, valid_loader, test_loader = get_cifar100_dataloaders(args, args.valid_split,
                                                                           max_threads=args.max_threads,
                                                                           worker_init_fn=worker_init_fn)
        output = 100
    elif args.data == 'kmnist':
        train_loader, valid_loader, test_loader = get_kmnist_dataloader(args, validation_split=args.valid_split,
                                                                        worker_init_fn=worker_init_fn)
        output = 10
    elif args.data == 'fashion_mnist':
        train_loader, valid_loader, test_loader = get_fashionmnist_dataloader(args, validation_split=args.valid_split,
                                                                              worker_init_fn=worker_init_fn)
        output = 10
    elif args.data == "tiny_imagenet":
        train_loader, valid_loader, test_loader = get_tinyimagenet_dataloaders(args, validation_split=args.valid_split,
                                                                               worker_init_fn=worker_init_fn)
        output = 200
    elif args.data == 'imagenet':
        print ("WARNING: valid and test are the same dataset in this implementation")
        train_loader, valid_loader, test_loader = get_ImageNet_loaders(args, distributed=False)
        output = 1000
    elif args.data == 'higgs':
        train_loader, valid_loader, test_loader = get_higgs_dataloaders(args, worker_init_fn=worker_init_fn)
        output = 1  # binary classification, just one output is needed
    else:
        raise ValueError("Unknown dataset")
    return output, test_loader, train_loader, valid_loader


def fix_seeds(args):
    # fix random seed for Reproducibility
    torch.backends.cudnn.benchmark = args.bench
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def parse_args_default(args=None):
    parser = get_parser()
    return parser.parse_args(args=args)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
