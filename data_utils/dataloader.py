import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, SVHN, FashionMNIST, EMNIST
import torchvision.transforms as transforms
from data_utils.sampling import *



def get_datasets(args):
    """
    Load datasets based on the input arguments.

    Args:
        args: Arguments containing dataset configuration.

    Returns:
        train_set: Training dataset.
        val_set: Validation dataset (currently None).
        test_set: Test dataset.
    """
    if args.data == 'cifar10':
        normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                         std=(0.2023, 0.1994, 0.2010))
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        train_set = CIFAR10(root=args.data_root, train=True, download=True, transform=train_transform)
        test_set = CIFAR10(root=args.data_root, train=False, transform=test_transform)
    
    elif args.data == 'svhn':
        normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                         std=(0.2023, 0.1994, 0.2010))
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        train_set = SVHN(root=args.data_root, split='train', download=True, transform=train_transform)
        test_set = SVHN(root=args.data_root, split='test', download=True, transform=test_transform)

    elif args.data == 'fmnist':
        normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                         std=(0.2023, 0.1994, 0.2010))
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        train_set = FashionMNIST(root=args.data_root, train=True, download=True, transform=train_transform)
        test_set = FashionMNIST(root=args.data_root, train=False, transform=test_transform)

    elif args.data == 'cifar100':
        normalize = transforms.Normalize(mean=(0.5071, 0.4867, 0.4408),
                                         std=(0.2675, 0.2565, 0.2761))
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        train_set = CIFAR100(root=args.data_root, train=True, download=True, transform=train_transform)
        test_set = CIFAR100(root=args.data_root, train=False, transform=test_transform)

    else:
        raise NotImplementedError(f"Dataset {args.data} is not implemented.")

    return train_set, None, test_set


def get_dataloaders(args, batch_size, dataset):
    """
    Create dataloaders for train, validation, and test splits.

    Args:
        args: Arguments containing dataloader configuration.
        batch_size: Batch size for training dataloader.
        dataset: Tuple containing train, validation, and test datasets.

    Returns:
        train_loader: DataLoader for training set.
        val_loader: DataLoader for validation set.
        test_loader: DataLoader for test set.
    """
    train_set, val_set, test_set = dataset

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True) if 'train' in args.splits else None

    test_loader = DataLoader(test_set, batch_size=500, shuffle=False,
                            num_workers=args.workers, pin_memory=True) if 'test' in args.splits else None

    val_loader = test_loader if 'val' in args.splits else None

    return train_loader, val_loader, test_loader


def get_user_groups(train_set, args):
    """
    Create non-IID user groups for federated learning.

    Args:
        train_set: Training dataset.
        args: Arguments containing user group configuration.

    Returns:
        train_user_groups: Dictionary mapping user IDs to data indices.
    """
    return create_noniid_users(train_set, args, args.alpha)


def get_client_dataloader(dataset, idxs, args, batch_size):
    """
    Create a DataLoader for a specific client based on given indices.

    Args:
        dataset: Dataset from which to sample.
        idxs: List of indices for the client.
        args: Arguments containing dataloader configuration.
        batch_size: Batch size for the client DataLoader.

    Returns:
        DataLoader: DataLoader for the client's data.
    """
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(idxs)),
        sampler=torch.utils.data.sampler.SubsetRandomSampler(idxs),
        num_workers=args.workers,
        pin_memory=True
    )



