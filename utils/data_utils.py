import logging

import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
logger = logging.getLogger(__name__)
def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset_path = args.dataset_path

    train_dir = dataset_path+"/train/"
    test_dir = dataset_path+"/test/"
    train_data = datasets.ImageFolder(train_dir, transform=transform_train)
    test_data = datasets.ImageFolder(test_dir, transform=transform_test)
    if args.local_rank == 0:
        torch.distributed.barrier()
    train_loader = DataLoader(train_data,
                              batch_size=args.train_batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(test_data,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             shuffle=False,
                             pin_memory=True)
    return train_loader, test_loader
