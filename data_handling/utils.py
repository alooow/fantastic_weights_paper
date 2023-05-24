import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from typing import Union
import argparse
from copy import deepcopy


def str2bool(v: Union[bool, str]) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

class DatasetSplitter(torch.utils.data.Dataset):
    """This splitter makes sure that we always use the same training/validation split"""
    def __init__(self,parent_dataset,split_start=-1,split_end= -1):
        split_start = split_start if split_start != -1 else 0
        split_end = split_end if split_end != -1 else len(parent_dataset)
        assert split_start <= len(parent_dataset) - 1 and split_end <= len(parent_dataset) and     split_start < split_end , "invalid dataset split"

        self.parent_dataset = parent_dataset
        self.split_start = split_start
        self.split_end = split_end

    def __len__(self):
        return self.split_end - self.split_start


    def __getitem__(self,index):
        assert index < len(self),"index out of bounds in split_datset"
        return self.parent_dataset[index + self.split_start]


def get_cifar100_dataloaders(args, validation_split=0.0, max_threads=10, worker_init_fn=None):
    """Creates augmented train, validation, and test data loaders."""
    cifar_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    cifar_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    trainset = torchvision.datasets.CIFAR100(root=os.environ['CIFAR100_PYTORCH'], train=True, download=True,
                                             transform=transform_train)

    if worker_init_fn is None:
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    else:
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2,
                                                   worker_init_fn=worker_init_fn)

    testset = torchvision.datasets.CIFAR100(root=os.environ['CIFAR100_PYTORCH'], train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, test_loader

def get_cifar10_dataloaders(args, validation_split=0.0, max_threads=10, transform=True, worker_init_fn=None):
    """Creates augmented train, validation, and test data loaders."""

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))

    if transform:
        train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                        (4,4,4,4),mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
         normalize
    ])

    full_dataset = datasets.CIFAR10(os.environ['CIFAR10_PYTORCH'], True, train_transform, download=True)
    test_dataset = datasets.CIFAR10(os.environ['CIFAR10_PYTORCH'], False, test_transform, download=False)


    # we need at least two threads
    max_threads = 2 if max_threads < 2 else max_threads
    if max_threads >= 6:
        val_threads = 2
        #train_threads = max_threads - val_threads
        train_threads = args.workers
    else:
        val_threads = 1
        train_threads = max_threads - 1


    valid_loader = None
    if validation_split > 0.0:
        split = int(np.floor((1.0-validation_split) * len(full_dataset)))
        train_dataset = DatasetSplitter(full_dataset,split_end=split)
        val_dataset = DatasetSplitter(full_dataset,split_start=split)
        if worker_init_fn is None:
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                args.batch_size,
                num_workers=train_threads,
                pin_memory=True, shuffle=True)
        else:
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                args.batch_size,
                num_workers=train_threads,
                pin_memory=True, shuffle=True, worker_init_fn=worker_init_fn)
        valid_loader = torch.utils.data.DataLoader(
            val_dataset,
            args.test_batch_size,
            num_workers=val_threads,
            pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            full_dataset,
            args.batch_size,
            num_workers=4,
            pin_memory=True, shuffle=True)

    print('Train loader length', len(train_loader))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        args.test_batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True)

    return train_loader, valid_loader, test_loader

def get_tinyimagenet_dataloaders(args, validation_split=0.0, worker_init_fn=None):
    traindir = os.path.join(os.environ["TINY_IMAGENET"], 'train')
    valdir = os.path.join(os.environ["TINY_IMAGENET"], 'val')
    testdir = os.path.join(os.environ["TINY_IMAGENET"], 'test')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            #transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    workers=4

    if worker_init_fn is None:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=workers, pin_memory=True, sampler=train_sampler)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=workers, pin_memory=True, sampler=train_sampler, worker_init_fn=worker_init_fn)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            #transforms.Resize(256),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    print("Warning: Using validation dataset as test")
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            #transforms.Resize(256),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def get_ImageNet_loaders(args, distributed=False):
    traindir = os.path.join(os.environ['IMAGENET_PYTORCH'], 'train')
    valdir = os.path.join(os.environ['IMAGENET_PYTORCH'], 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    max_threads = 2 if args.max_threads < 2 else args.max_threads
    if max_threads >= 6:
        val_threads = 2
        train_threads = max_threads - val_threads
    else:
        val_threads = 1
        train_threads = max_threads - 1

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=train_threads, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=val_threads, pin_memory=True, sampler=val_sampler)
    return train_loader, val_loader, val_loader


def get_fashionmnist_dataloader(args, validation_split=0.0, worker_init_fn=None):
    """Creates augmented train, validation, and test data loaders."""
    normalize = transforms.Normalize((0.5,), (0.5,))
    transform = transform=transforms.Compose([transforms.ToTensor(),normalize])

    full_dataset = datasets.FashionMNIST(os.environ['FASHION_MNIST_PYTORCH'], train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(os.environ['FASHION_MNIST_PYTORCH'], train=False, transform=transform)

    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    valid_loader = None
    if validation_split > 0.0:
        split = int(np.floor((1.0-validation_split) * len(full_dataset)))
        train_dataset = DatasetSplitter(full_dataset,split_end=split)
        val_dataset = DatasetSplitter(full_dataset,split_start=split)
        if worker_init_fn is None:
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                args.batch_size,
                num_workers=8,
                pin_memory=True, shuffle=True)
        else:
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                args.batch_size,
                num_workers=8,
                pin_memory=True, shuffle=True, worker_init_fn=worker_init_fn)
        valid_loader = torch.utils.data.DataLoader(
            val_dataset,
            args.test_batch_size,
            num_workers=2,
            pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            full_dataset,
            args.batch_size,
            num_workers=8,
            pin_memory=True, shuffle=True)

    print('Train loader length', len(train_loader))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        args.test_batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True)

    return train_loader, valid_loader, test_loader

def get_kmnist_dataloader(args, validation_split=0.0, worker_init_fn=None):
    """Creates augmented train, validation, and test data loaders."""
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    transform = transform=transforms.Compose([transforms.ToTensor(),normalize])

    full_dataset = datasets.KMNIST(os.environ['KMNIST_PYTORCH'], train=True, download=True, transform=transform)
    test_dataset = datasets.KMNIST(os.environ['KMNIST_PYTORCH'], train=False, transform=transform)

    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    valid_loader = None
    if validation_split > 0.0:
        split = int(np.floor((1.0-validation_split) * len(full_dataset)))
        train_dataset = DatasetSplitter(full_dataset,split_end=split)
        val_dataset = DatasetSplitter(full_dataset,split_start=split)
        if worker_init_fn is None:
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                args.batch_size,
                num_workers=8,
                pin_memory=True, shuffle=True)
        else:
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                args.batch_size,
                num_workers=8,
                pin_memory=True, shuffle=True,
                worker_init_fn=worker_init_fn)
        valid_loader = torch.utils.data.DataLoader(
            val_dataset,
            args.test_batch_size,
            num_workers=2,
            pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            full_dataset,
            args.batch_size,
            num_workers=8,
            pin_memory=True, shuffle=True)

    print('Train loader length', len(train_loader))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        args.test_batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True)

    return train_loader, valid_loader, test_loader



def get_mnist_dataloaders(args, validation_split=0.0):
    """Creates augmented train, validation, and test data loaders."""
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    transform = transform=transforms.Compose([transforms.ToTensor(),normalize])

    full_dataset = datasets.MNIST(os.environ['MNIST_PYTORCH'], train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(os.environ['MNIST_PYTORCH'], train=False, transform=transform)

    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    valid_loader = None
    if validation_split > 0.0:
        split = int(np.floor((1.0-validation_split) * len(full_dataset)))
        train_dataset = DatasetSplitter(full_dataset,split_end=split)
        val_dataset = DatasetSplitter(full_dataset,split_start=split)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            args.batch_size,
            num_workers=8,
            pin_memory=True, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(
            val_dataset,
            args.test_batch_size,
            num_workers=2,
            pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            full_dataset,
            args.batch_size,
            num_workers=8,
            pin_memory=True, shuffle=True)

    print('Train loader length', len(train_loader))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        args.test_batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True)

    return train_loader, valid_loader, test_loader


def get_higgs_dataloaders(args, worker_init_fn=None):
    """ Creates dataloaders for the Higgs dataset """
    import openml
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, TensorDataset

    # set the directory where the data is stored
    openml.config.cache_directory = os.environ['TABULAR_DATA_DIR']
    # download the OpenML task (Higgs dataset has task_id 361069)
    task = openml.tasks.get_task(task_id=361069)
    dataset = task.get_dataset()
    X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="dataframe",
                                                                    target=dataset.default_target_attribute)
    y = y.astype(int)  # convert y to ints

    # split into train and test with sklearn, where an equal number of 0's and 1's are in each (by stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    # split the test into validation and test
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)
    # we now have: train (70%), validation (15%), test (15%)

    # make dataset objects
    train_dataset = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32),
                                  torch.tensor(y_train.values, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val.values, dtype=torch.float32),
                                torch.tensor(y_val.values, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test.values, dtype=torch.float32),
                                 torch.tensor(y_test.values, dtype=torch.float32))

    # make dataloaders
    if worker_init_fn is None:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def plot_class_feature_histograms(args, model, device, test_loader, optimizer):
    if not os.path.exists('./results'): os.mkdir('./results')
    model.eval()
    agg = {}
    num_classes = 10
    feat_id = 0
    sparse = not args.dense
    model_name = 'alexnet'
    #model_name = 'vgg'
    #model_name = 'wrn'


    densities = None
    for batch_idx, (data, target) in enumerate(test_loader):
        if batch_idx % 100 == 0: print(batch_idx,'/', len(test_loader))
        with torch.no_grad():
            #if batch_idx == 10: break
            data, target = data.to(device), target.to(device)
            for cls in range(num_classes):
                #print('=='*50)
                #print('CLASS {0}'.format(cls))
                model.t = target
                sub_data = data[target == cls]

                output = model(sub_data)

                feats = model.feats
                if densities is None:
                    densities = []
                    densities += model.densities

                if len(agg) == 0:
                    for feat_id, feat in enumerate(feats):
                        agg[feat_id] = []
                        #print(feat.shape)
                        for i in range(feat.shape[1]):
                            agg[feat_id].append(np.zeros((num_classes,)))

                for feat_id, feat in enumerate(feats):
                    map_contributions = torch.abs(feat).sum([0, 2, 3])
                    for map_id in range(map_contributions.shape[0]):
                        #print(feat_id, map_id, cls)
                        #print(len(agg), len(agg[feat_id]), len(agg[feat_id][map_id]), len(feats))
                        agg[feat_id][map_id][cls] += map_contributions[map_id].item()

                del model.feats[:]
                del model.densities[:]
                model.feats = []
                model.densities = []

    if sparse:
        np.save('./results/{0}_sparse_density_data'.format(model_name), densities)

    for feat_id, map_data in agg.items():
        data = np.array(map_data)
        #print(feat_id, data)
        full_contribution = data.sum()
        #print(full_contribution, data)
        contribution_per_channel = ((1.0/full_contribution)*data.sum(1))
        #print('pre', data.shape[0])
        channels = data.shape[0]
        #data = data[contribution_per_channel > 0.001]

        channel_density = np.cumsum(np.sort(contribution_per_channel))
        print(channel_density)
        idx = np.argsort(contribution_per_channel)

        threshold_idx = np.searchsorted(channel_density, 0.05)
        print(data.shape, 'pre')
        data = data[idx[threshold_idx:]]
        print(data.shape, 'post')

        #perc = np.percentile(contribution_per_channel[contribution_per_channel > 0.0], 10)
        #print(contribution_per_channel, perc, feat_id)
        #data = data[contribution_per_channel > perc]
        #print(contribution_per_channel[contribution_per_channel < perc].sum())
        #print('post', data.shape[0])
        normed_data = np.max(data/np.sum(data,1).reshape(-1, 1), 1)
        #normed_data = (data/np.sum(data,1).reshape(-1, 1) > 0.2).sum(1)
        #counts, bins = np.histogram(normed_data, bins=4, range=(0, 4))
        np.save('./results/{2}_{1}_feat_data_layer_{0}'.format(feat_id, 'sparse' if sparse else 'dense', model_name), normed_data)
        #plt.ylim(0, channels/2.0)
        ##plt.hist(normed_data, bins=range(0, 5))
        #plt.hist(normed_data, bins=[(i+20)/float(200) for i in range(180)])
        #plt.xlim(0.1, 0.5)
        #if sparse:
        #    plt.title("Sparse: Conv2D layer {0}".format(feat_id))
        #    plt.savefig('./output/feat_histo/layer_{0}_sp.png'.format(feat_id))
        #else:
        #    plt.title("Dense: Conv2D layer {0}".format(feat_id))
        #    plt.savefig('./output/feat_histo/layer_{0}_d.png'.format(feat_id))
        #plt.clf()

def combine_with_defaults(config, defaults):
    res = deepcopy(defaults)
    for k, v in config.items():
        assert k in res.keys(), "{} not in default values".format(k)
        res[k] = v
    return res
