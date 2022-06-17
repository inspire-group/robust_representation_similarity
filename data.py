import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def cifar10(datadir, batch_size, image_size=32, shuffle_test=True):
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    transform_test = transforms.ToTensor()

    trainset = datasets.CIFAR10(
        root=os.path.join(datadir, "cifar10"),
        train=True,
        download=True,
        transform=transform_train,
    )
    testset = datasets.CIFAR10(
        root=os.path.join(datadir, "cifar10"),
        train=False,
        download=True,
        transform=transform_test,
    )

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=shuffle_test,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader


def imagenet(datadir, batch_size, image_size=224, shuffle_test=True):
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(int(1.14 * image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )

    trainset = datasets.ImageFolder(
        os.path.join(datadir, "train"), transform=transform_train
    )
    testset = datasets.ImageFolder(
        os.path.join(datadir, "val"), transform=transform_test
    )

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=shuffle_test,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader


imagenette = imagewoof = imagenet  # just change the dataset dir to load these datasets
