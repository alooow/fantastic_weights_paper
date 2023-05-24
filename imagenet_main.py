import argparse
import os
import shutil
import time
import random
import math

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
#from colorama import Fore
import sys
import sparselearning.core as core
from data_handling.utils import str2bool
from data_handling.logger import get_wandb_logger
from setup_utils import get_mask


from torch.nn.parallel import DistributedDataParallel as DDP
#try:
#    from apex.parallel import DistributedDataParallel as DDP
#    from apex.fp16_utils import *
#except ImportError:
#    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

import models.imagenet_resnet as models
from ImageNet.smoothing import LabelSmoothing

def add_parser_arguments(parser):
    model_names = models.resnet_versions.keys()
    model_configs = models.resnet_configs.keys()

    parser.add_argument('--data', metavar='DIR', default='./data',
                        help='path to dataset')
    parser.add_argument('--save_dir', default='save/',
                           type=str, metavar='SAVE',
                           help='path to the experiment logging directory'
                                '(default: save/debug)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
    parser.add_argument('--model-config', '-c', metavar='CONF', default='classic',
                        choices=model_configs,
                        help='model configs: ' +
                        ' | '.join(model_configs) + '(default: classic)')
    parser.add_argument('-j', '--workers', default=5, type=int, metavar='N',
                        help='number of data loading workers (default: 5)')
    parser.add_argument('--spp', type=str2bool, default='false',
                        help='flag to switch another grow initialization')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=int,
                        metavar='E', help='number of warmup epochs')
    parser.add_argument('--label-smoothing', default=0.0, type=float,
                        metavar='S', help='label smoothing')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--bn-weight-decay',
                        type=str2bool,
                        default='false',
                        help='use weight_decay on batch normalization learnable parameters, default: false)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', type=str2bool, default='false',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained-weights', default='', type=str, metavar='PATH',
                        help='file with weights')

    parser.add_argument('--fp16',
                        help='Run model fp16 mode.',
                        type=str2bool,
                        default='false'
                        )
    parser.add_argument('--static-loss-scale', type=float, default=1,
                        help='Static loss scale, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--dynamic-loss-scale', type=str2bool, default='false',
                        help='Use dynamic loss scaling.  If supplied, this argument supersedes ' +
                        '--static-loss-scale.')
    parser.add_argument('--prof', dest='prof', type=str2bool, default='false',
                        help='Only run 10 iterations for profiling.')
    parser.add_argument('--scaled', dest='scaled', type=str2bool, default='false',
                        help='scaled initialization used in whats hidden.')
    parser.add_argument('--explo-start-epoch', type=int, default=0, metavar='N',
                        help='Number of epochs for explo')
    parser.add_argument('--benchmark-training', dest='trainbench', type=str2bool, default='false',
                        help='Run benchmarking of training')
    parser.add_argument('--benchmark-inference', dest='inferbench',  type=str2bool, default='false',
                        help='Run benchmarking of training')
    parser.add_argument('--bench-iterations', type=int, default=20, metavar='N',
                        help='Run N iterations while benchmarking (ignored when training and validation)')
    parser.add_argument('--bench-warmup', type=int, default=20, metavar='N',
                        help='Number of warmup iterations for benchmarking')
    parser.add_argument('--master_port', '-master_port', default=10000, type=int,
              help="Port of master for torch.distributed training.")
    parser.add_argument("--local_rank", default=0, type=int)

    parser.add_argument('--seed', default=17, type=int,
                        help='random seed used for np and pytorch')

    parser.add_argument('--gather-checkpoints', action='store_true',
                        help='Gather checkpoints throughout the training')
    parser.add_argument('--grow-switch', default='', type=str,
                           help='flag to switch another grow initialization')
    parser.add_argument('--use_wandb', type=str2bool, default="true")
    parser.add_argument('--log_interval', type=int, default=1, metavar='N')
    parser.add_argument('--save_locally', type=str2bool, default="true")
    parser.add_argument('--distributed', type=str2bool, default="true")
    parser.add_argument('--tag', type=str, default='ImageNet')
    parser.add_argument('--multiplier', type=int, default=1)
    parser.add_argument('--opt_order', type=str, default='after', choices = ['before','after'])
    parser.add_argument('--manual_stop', type=str2bool, default="false", help="if true, will automatically stop the "
                                                                              "training after first pruning")

    parser.add_argument('--no_cuda', type=str2bool, default="false",
                        help='disables CUDA training')
    parser.add_argument('--gamma', type=float, default=1.0)

    core.add_sparse_args(parser)

def main():
    args.data = os.environ['IMAGENET_PYTORCH']
    if args.trainbench or args.inferbench:
        logger = BenchLogger
    else:
        logger = PrintLogger

    wandb_logger = get_wandb_logger(args)

    train_net(args, logger, wandb_logger)

def train_net(args, logger_cls, wandb_logger):
    exp_start_time = time.time()
    global best_prec1
    best_prec1 = 0

    if args.death not in ["magnitude", "Random", "SET", "SETFixed", "RunningMagnitude"]:
        args.opt_order = "after"
    if args.manual_stop:
        args.opt_order = "after"

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        os.environ['MASTER_PORT'] = str(args.master_port)
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    if args.seed is not None:
        print("Using seed = {}".format(args.seed))
        torch.manual_seed(args.seed + args.local_rank)
        torch.cuda.manual_seed(args.seed + args.local_rank)
        np.random.seed(seed=args.seed + args.local_rank)
        random.seed(args.seed + args.local_rank)

        def _worker_init_fn(id):
            np.random.seed(seed=args.seed + args.local_rank + id)
            random.seed(args.seed + args.local_rank + id)
    else:
        def _worker_init_fn(id):
            pass

    if args.fp16:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."

    if args.static_loss_scale != 1.0:
        if not args.fp16:
            print("Warning:  if --fp16 is not used, static_loss_scale will be ignored.")

    pretrained_weights = None
    if args.pretrained_weights:
        if os.path.isfile(args.pretrained_weights):
            print("=> loading pretrained weights from '{}'".format(args.pretrained_weights))
            pretrained_weights = torch.load(args.pretrained_weights)
        else:
            print("=> no pretrained weights found at '{}'".format(args.resume))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model_state = checkpoint['state_dict']
            optimizer_state = checkpoint['optimizer']
            print("=> loaded checkpoint '{}' (epoch {}) with acc {}"
                  .format(args.resume, checkpoint['epoch'], best_prec1))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            model_state = None
            optimizer_state = None
    else:
        model_state = None
        optimizer_state = None

    model_and_loss = ModelAndLoss(args,
            (args.arch, args.model_config),
            nn.CrossEntropyLoss if args.label_smoothing == 0.0 else (lambda: LabelSmoothing(args.label_smoothing)),
            pretrained_weights=pretrained_weights,
            state=model_state,
            cuda = True, fp16 = args.fp16, distributed = args.distributed)

    # Create data loaders and optimizers as needed

    if not (args.evaluate or args.inferbench):
        optimizer = get_optimizer(list(model_and_loss.model.named_parameters()),
                args.fp16,
                args.lr, args.momentum, args.weight_decay,
                bn_weight_decay = args.bn_weight_decay,
                state=optimizer_state,
                static_loss_scale = args.static_loss_scale,
                dynamic_loss_scale = args.dynamic_loss_scale)


        train_loader = get_train_loader(args.data, args.batch_size, workers=args.workers, _worker_init_fn=_worker_init_fn)
        train_loader_len = len(train_loader)
    else:
        train_loader = None
        optimizer = None
        train_loader_len = 0

    if not args.trainbench:
        val_loader = get_val_loader(args.data, args.batch_size, workers=args.workers, _worker_init_fn=_worker_init_fn)
        val_loader_len = len(val_loader)
    else:
        val_loader_len = 0

    device_type = torch.device("cuda")
    mask = get_mask(args, model_and_loss, optimizer, train_loader, device_type, set_mask=True, multiple_gpu=args.distributed)
    

    if args.scaled:
        print('scale the initialization')
        # scaled kaiming kormal reinitialization
        for name, weight in model_and_loss.model.named_parameters():
            if name not in mask.masks: continue
            if len(weight.shape) == 4:
                    print(f'scale layer {name}')
                    fan = nn.init._calculate_correct_fan(weight, 'fan_in')
                    layer_density = float((weight != 0).sum().int().item()/weight.numel())
                    fan = fan * layer_density
                    gain = nn.init.calculate_gain('relu')
                    std = gain / math.sqrt(fan)
                    with torch.no_grad():
                        weight.data.normal_(0, std)

    if args.evaluate:
        logger = logger_cls(train_loader_len, val_loader_len, args)
        validate(val_loader, model_and_loss, args.fp16, logger, 0)
        return

    if args.trainbench:
        model_and_loss.model.train()
        logger = logger_cls("Train", args.world_size * args.batch_size, args.bench_warmup)
        bench(get_train_step(model_and_loss, optimizer, args.fp16), train_loader,
              args.bench_warmup, args.bench_iterations, args.fp16, logger, epoch_warmup = True)
        return

    if args.inferbench:
        model_and_loss.model.eval()
        logger = logger_cls("Inference", args.world_size * args.batch_size, args.bench_warmup)
        bench(get_val_step(model_and_loss), val_loader,
              args.bench_warmup, args.bench_iterations, args.fp16, logger, epoch_warmup = False)
        return

    logger = logger_cls(train_loader_len, val_loader_len, args)
    train_loop(args, model_and_loss, optimizer, adjust_learning_rate(args), train_loader, val_loader, args.epochs*args.multiplier,
            args.fp16, logger, should_backup_checkpoint(args),
            start_epoch = args.start_epoch, best_prec1 = best_prec1, prof=args.prof, wandb_logger=wandb_logger)

    exp_duration = time.time() - exp_start_time
    logger.experiment_timer(exp_duration)
    logger.end_callback()
    wandb_logger.finish()
    print("Experiment ended")


# get optimizer {{{
def get_optimizer(parameters, fp16, lr, momentum, weight_decay,
                  true_wd=False,
                  nesterov=False,
                  state=None,
                  static_loss_scale=1., dynamic_loss_scale=False,
                  bn_weight_decay = False):

    if bn_weight_decay:
        print(" ! Weight decay applied to BN parameters ")
        optimizer = torch.optim.SGD([v for n, v in parameters], lr,
                           momentum=momentum,
                           weight_decay=weight_decay,
                           nesterov = nesterov)
    else:
        print(" ! Weight decay NOT applied to BN parameters ")
        bn_params = [v for n, v in parameters if 'bn' in n]
        rest_params = [v for n, v in parameters if not 'bn' in n]
        print(len(bn_params))
        print(len(rest_params))
        optimizer = torch.optim.SGD([{'params': bn_params, 'weight_decay' : 0},
                            {'params': rest_params, 'weight_decay' : weight_decay}],
                           lr,
                           momentum=momentum,
                           weight_decay=weight_decay,
                           nesterov = nesterov)
    if fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=static_loss_scale,
                                   dynamic_loss_scale=dynamic_loss_scale)

    if not state is None:
        optimizer.load_state_dict(state)

    return optimizer
# }}}

# ModelAndLoss {{{
class ModelAndLoss(nn.Module):
    def __init__(self, args, arch, loss, pretrained_weights=None, state=None, cuda=True, fp16=False, distributed=False):
        super(ModelAndLoss, self).__init__()
        self.arch = arch
        self.mask = None


        print("=> creating model '{}'".format(arch))
        model = models.build_resnet(arch[0], arch[1])
        if pretrained_weights is not None:
            print("=> using pre-trained model from a file '{}'".format(arch))
            model.load_state_dict(pretrained_weights)

        if cuda:
            model = model.cuda()
        if fp16:
            model = network_to_half(model)
        if distributed:
            # model = DDP(model, device_ids=[args.local_rank]) # for Pytorch DDP
            model = DDP(model)

        if not state is None:
            new_state = {}
            for key1, key2 in zip(state, model.state_dict()):
                new_state[key2] = state[key1]
            model.load_state_dict(new_state)

        # define loss function (criterion) and optimizer
        criterion = loss()

        if cuda:
            criterion = criterion.cuda()

        self.model = model
        self.loss = criterion

    def forward(self, data, target):
        output = self.model(data)
        loss = self.loss(output, target)

        return loss, output
# }}}

# Train loop {{{
def train_loop(args, model_and_loss, optimizer, lr_scheduler, train_loader, val_loader, epochs, fp16, logger,
               should_backup_checkpoint,
               best_prec1 = 0, start_epoch = 0, prof = False, wandb_logger=None):
    update_iter = 0
    for epoch in range(start_epoch, epochs):
        if torch.distributed.is_initialized():
            train_loader.sampler.set_epoch(epoch)

        lr_scheduler(optimizer, epoch)
        prec1, prec5, loss = train(train_loader, model_and_loss, optimizer, fp16, logger, epoch, prof = prof)
        #val_prec1, val_prec5, val_loss = validate(val_loader, model_and_loss, fp16, logger, epoch, prof = prof)

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:

            metrics = {'epoch': epoch,  
                       'train_acc':prec1, "train_top5":prec5, 'train_loss':loss}
            wandb_logger.log(metrics)
            if (epoch+1) % 10 == 0 :
                save = True
            else:
                save = False
            #best_prec1 = max(val_prec1, best_prec1)
            #print(Fore.GREEN + 'Best var_acc1 {}'.format(best_prec1) +
            #      Fore.RESET)
            if should_backup_checkpoint(epoch):
                backup_filename = args.save_dir + 'checkpoint-{}.pth.tar'.format(epoch + 1)
            else:
                backup_filename = None
            if model_and_loss.mask is not None:
                save_mask = model_and_loss.mask.masks
            else:
                save_mask = {}
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': model_and_loss.arch,
                'state_dict': model_and_loss.model.state_dict(),
                'train_prec1': prec1,
                'optimizer' : optimizer.state_dict(),
                'mask':save_mask
            }, save, epoch)
    val_prec1, val_prec5, val_loss = validate(val_loader, model_and_loss, fp16, logger, epochs, prof = prof)
    metrics = {'epoch':epochs, "val_acc":val_prec1, "val_top5":val_prec5, "val_loss":val_loss}
    wandb_logger.log(metrics)

# }}}

# Data Loading functions {{{
def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8 )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        # tens = torch.from_numpy(nump_array)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        nump_array_copy = np.copy(nump_array)
        tensor[i] += torch.from_numpy(nump_array_copy)

    return tensor, targets


def prefetched_loader(loader, fp16):
    mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
    std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
    if fp16:
        mean = mean.half()
        std = std.half()

    stream = torch.cuda.Stream()
    first = True

    for next_input, next_target in loader:
        with torch.cuda.stream(stream):
            next_input = next_input.cuda(non_blocking=True)
            next_target = next_target.cuda(non_blocking=True)
            if fp16:
                next_input = next_input.half()
            else:
                next_input = next_input.float()
            next_input = next_input.sub_(mean).div_(std)

        if not first:
            yield input, target
        else:
            first = False

        torch.cuda.current_stream().wait_stream(stream)
        input = next_input
        target = next_target

    yield input, target


def get_train_loader(data_path, batch_size, workers=5, _worker_init_fn=None):
    traindir = os.path.join(data_path, 'train')
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            #transforms.ToTensor(), Too slow
            #normalize,
        ]))

    if torch.distributed.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=True, sampler=train_sampler, collate_fn=fast_collate, drop_last=True)

    return train_loader

def get_val_loader(data_path, batch_size, workers=5, _worker_init_fn=None):
    valdir = os.path.join(data_path, 'val')

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=True,
        collate_fn=fast_collate)

    return val_loader
# }}}

# Train val bench {{{
def get_train_step(model_and_loss, optimizer, fp16):
    def _step(input, target):
        input_var = Variable(input)
        target_var = Variable(target)
        loss, output = model_and_loss(input_var, target_var)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if torch.distributed.is_initialized():
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        optimizer.zero_grad()
        if fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        if model_and_loss.mask is None:
            optimizer.step()
        else:
            model_and_loss.mask.step()

        torch.cuda.synchronize()

        return reduced_loss, prec1, prec5

    return _step


def bench(step, train_loader, warmup, iterations, fp16, logger, epoch_warmup = False):
    step = timed_function(step)

    if epoch_warmup:
        print("Running first epoch for warmup, please wait")

        for (input, target), dt in timed_generator(prefetched_loader(train_loader, fp16)):
            _, bt = step(input, target)

    print("Running benchmarked epoch")

    for (input, target), dt in timed_generator(prefetched_loader(train_loader, fp16)):
        _, bt = step(input, target)
        logger.iter_callback({'data_time': dt, 'batch_time': bt})

        if logger.i >= warmup + iterations:
            break

    logger.end_callback()

def train(train_loader, model_and_loss, optimizer, fp16, logger, epoch, prof=False):
    global update_iter
    step = get_train_step(model_and_loss, optimizer, fp16)

    model_and_loss.model.train()
    end = time.time()

    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_m = AverageMeter()

    for i, (input, target) in enumerate(prefetched_loader(train_loader, fp16)):
        data_time = time.time() - end
        if prof:
            if i > 10:
                break

        loss, prec1, prec5 = step(input, target)
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))
        loss_m.update(to_python_float(loss), input.size(0))

        logger.train_iter_callback(epoch, i,
                        {'size' : input.size(0),
                         'top1' : to_python_float(prec1),
                         'top5' : to_python_float(prec5),
                         'loss' : to_python_float(loss),
                         'time' : time.time() - end,
                         'data' : data_time})

        end = time.time()
        sys.stdout.flush()
        update_iter += 1
        if update_iter % args.update_frequency == 0 and not args.fix and args.sparse:
            model_and_loss.mask.at_end_of_epoch()
    logger.train_epoch_callback(epoch)
    return top1.avg, top5.avg, loss_m.avg


def get_val_step(model_and_loss):
    def _step(input, target):
        input_var = Variable(input)
        target_var = Variable(target)

        with torch.no_grad():
            loss, output = model_and_loss(input_var, target_var)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if torch.distributed.is_initialized():
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        torch.cuda.synchronize()

        return reduced_loss, prec1, prec5

    return _step


def validate(val_loader, model_and_loss, fp16, logger, epoch, prof=False):

    step = get_val_step(model_and_loss)

    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_m = AverageMeter()
    # switch to evaluate mode
    model_and_loss.model.eval()

    end = time.time()

    for i, (input, target) in enumerate(prefetched_loader(val_loader, fp16)):
        data_time = time.time() - end
        if prof:
            if i > 10:
                break

        loss, prec1, prec5 = step(input, target)

        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))
        loss_m.update(to_python_float(loss), input.size(0))

        logger.val_iter_callback(epoch, i,
                {'size' : input.size(0),
                 'top1' : to_python_float(prec1),
                 'top5' : to_python_float(prec5),
                 'loss' : to_python_float(loss),
                 'time' : time.time() - end,
                 'data' : data_time})

        end = time.time()

    logger.val_epoch_callback(epoch)

    return top1.avg, top5.avg, loss_m.avg

# }}}

# Logging {{{
class BenchLogger(object):
    def __init__(self, name, total_bs, warmup_iter):
        self.name = name
        self.data_time = AverageMeter()
        self.batch_time = AverageMeter()
        self.warmup_iter = warmup_iter
        self.total_bs = total_bs
        self.i = 0

    def reset(self):
        self.data_time.reset()
        self.batch_time.reset()
        self.i = 0

    def iter_callback(self, d):
        bt = d['batch_time']
        dt = d['data_time']
        if self.i >= self.warmup_iter:
            self.data_time.update(dt)
            self.batch_time.update(bt)
        self.i += 1

        print("Iter: [{}]\tbatch: {:.3f}\tdata: {:.3f}\timg/s (compute): {:.3f}\timg/s (total): {:.3f}".format(
            self.i, dt + bt, dt,
            self.total_bs / bt, self.total_bs / (bt + dt)))

    def end_callback(self):
        print("{} summary\tBatch Time: {:.3f}\tData Time: {:.3f}\timg/s (compute): {:.1f}\timg/s (total): {:.1f}".format(
              self.name,
              self.batch_time.avg, self.data_time.avg,
              self.total_bs / self.batch_time.avg,
              self.total_bs / (self.batch_time.avg + self.data_time.avg)))


class EpochLogger(object):
    def __init__(self, name, total_iterations, args):
        self.name = name
        self.args = args
        self.print_freq = args.print_freq
        self.total_iterations = total_iterations
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()
        self.loss = AverageMeter()
        self.time = AverageMeter()
        self.data = AverageMeter()

    def iter_callback(self, epoch, iteration, d):
        self.top1.update(d['top1'], d['size'])
        self.top5.update(d['top5'], d['size'])
        self.loss.update(d['loss'], d['size'])
        self.time.update(d['time'], d['size'])
        self.data.update(d['data'], d['size'])

        if iteration % self.print_freq == 0:
            print('{0}:\t{1} [{2}/{3}]\t'
                  'Time {time.val:.3f} ({time.avg:.3f})\t'
                  'Data time {data.val:.3f} ({data.avg:.3f})\t'
                  'Speed {4:.3f} ({5:.3f})\t'
                  'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                  self.name, epoch, iteration, self.total_iterations,
                  self.args.world_size * self.args.batch_size / self.time.val,
                  self.args.world_size * self.args.batch_size / self.time.avg,
                  time=self.time,
                  data=self.data,
                  loss=self.loss,
                  top1=self.top1,
                  top5=self.top5))

    def epoch_callback(self, epoch):
        print('{0} epoch {1} summary:\t'
              'Time {time.avg:.3f}\t'
              'Data time {data.avg:.3f}\t'
              'Speed {2:.3f}\t'
              'Loss {loss.avg:.4f}\t'
              'Prec@1 {top1.avg:.3f}\t'
              'Prec@5 {top5.avg:.3f}'.format(
              self.name, epoch,
              self.args.world_size * self.args.batch_size / self.time.avg,
              time=self.time, data=self.data,
              loss=self.loss, top1=self.top1, top5=self.top5))

        self.top1.reset()
        self.top5.reset()
        self.loss.reset()
        self.time.reset()
        self.data.reset()


class PrintLogger(object):
    def __init__(self, train_iterations, val_iterations, args):
        self.train_logger = EpochLogger("Train", train_iterations, args)
        self.val_logger = EpochLogger("Eval", val_iterations, args)

    def train_iter_callback(self, epoch, iteration, d):
        self.train_logger.iter_callback(epoch, iteration, d)

    def train_epoch_callback(self, epoch):
        self.train_logger.epoch_callback(epoch)
        
    def val_iter_callback(self, epoch, iteration, d):
        self.val_logger.iter_callback(epoch, iteration, d)

    def val_epoch_callback(self, epoch):
        self.val_logger.epoch_callback(epoch)
        
    def experiment_timer(self, exp_duration):
        print("Experiment took {} seconds".format(exp_duration))

    def end_callback(self):
        pass


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# }}}

def should_backup_checkpoint(args):
    def _sbc(epoch):
        return args.gather_checkpoints and (epoch < 10 or epoch % 10 == 0)
    return _sbc

def save_checkpoint(state, is_best, epoch):
    if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
        if is_best:
            print("SAVING")
            if not args.sparse:
                torch.save(state, args.save_dir + 'checkpoint-dense-{}-{}-seed-{}.pth.tar'.format(str(epoch), args.arch, str(args.seed), ))
            else:
                torch.save(state, args.save_dir + 'checkpoint-{}-{}-{}-{}-seed-{}.pth.tar'.format(str(epoch), args.arch, str(args.sparse_init), str(args.density), str(args.seed), ))


def timed_generator(gen):
    start = time.time()
    for g in gen:
        end = time.time()
        t = end - start
        yield g, t
        start = time.time()


def timed_function(f):
    def _timed_function(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        return ret, time.time() - start
    return _timed_function


def adjust_learning_rate(args):
    def _alr(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        if epoch < args.warmup:
            lr = args.lr * (epoch + 1) / (args.warmup + 1)

        else:
            if epoch < 30*args.multiplier:
                p = 0
            elif epoch < 60*args.multiplier:
                p = 1
            elif epoch < 80*args.multiplier:
                p = 2
            else:
                p = 3
            lr = args.lr * (0.1 ** p)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM, async_op=False)
    rt /= torch.distributed.get_world_size()
    return rt

def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    update_iter = 0
    add_parser_arguments(parser)
    args = parser.parse_args()
    cudnn.benchmark = True
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    main()
