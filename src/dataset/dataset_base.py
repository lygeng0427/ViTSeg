# encoding:utf-8

import cv2
import numpy as np
import src.dataset.transform as transform
from torch.utils.data import Dataset
from .utils import make_dataset
from .classes import get_split_classes, filter_classes
import torch
import random
import argparse
from typing import List
from torchvision import transforms as T
from torch.utils.data.distributed import DistributedSampler


def get_train_loader(args, return_path=False):
    """
        Build the train loader. This is a episodic loader.
    """
    assert args.split in [0, 1, 2, 3]
    padding = [v*255 for v in args.mean] if args.get('padding') == 'avg' else None
    aug_dic = {
        'randscale': transform.RandScale([args.scale_min, args.scale_max]),
        'randrotate': transform.RandRotate(
            [args.rot_min, args.rot_max],
            padding=[0 for x in args.mean],
            ignore_label=255
        ),
        'hor_flip': transform.RandomHorizontalFlip(),
        'vert_flip': transform.RandomVerticalFlip(),
        'crop': transform.Crop(
            [args.image_size, args.image_size], crop_type='rand',
            padding=[0 for x in args.mean], ignore_label=255
        ),
        'resize': transform.Resize(args.image_size, padding=padding),
    }

    train_transform = [aug_dic[name] for name in args.augmentations]
    train_transform += [transform.ToTensor(), transform.Normalize(mean=args.mean, std=args.std)]
    train_transform = transform.Compose(train_transform)

    split_classes = get_split_classes(args)     # 只用了 args.use_split_coco 这个参数， 返回coco和pascal所有4个split, dict of dict
    class_list = split_classes[args.data_name][args.split]['train']   # list of all meta train class labels

    # ====== Build loader ======
    train_data = StandardData(
        transform=train_transform,
        class_list=class_list,
        return_paths=return_path,
        data_list_path=args.train_list,
        args=args
        )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        sampler=None,
        drop_last=True
    )
    return train_loader, None


class StandardData(Dataset):
    def __init__(self, args: argparse.Namespace,
                 transform: transform.Compose,
                 data_list_path: str,
                 class_list: List[int],
                 return_paths: bool):
        self.data_root = args.data_root
        self.class_list = class_list
        self.data_list, _ = make_dataset(args.data_root, data_list_path, class_list)
        self.transform = transform
        self.return_paths = return_paths

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        label_class = np.unique(label).tolist()
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)
        new_label_class = []
        undesired_class = []
        for c in label_class:
            if c in self.class_list:
                new_label_class.append(c)
            else:
                undesired_class.append(c)
        label_class = new_label_class
        assert len(label_class) > 0

        new_label = np.zeros_like(label)  # background
        for lab in label_class:
            indexes = np.where(label == lab)
            new_label[indexes[0], indexes[1]] = self.class_list.index(lab) + 1  # Add 1 because class 0 is for bg
        for lab in undesired_class:
            indexes = np.where(label == lab)
            new_label[indexes[0], indexes[1]] = 255

        ignore_pix = np.where(new_label == 255)
        new_label[ignore_pix[0], ignore_pix[1]] = 255

        if self.transform is not None:
            image, new_label = self.transform(image, new_label)
        if self.return_paths:
            return image, new_label, image_path, label_path
        else:
            return image, new_label
