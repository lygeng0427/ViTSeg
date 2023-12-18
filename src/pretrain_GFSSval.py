# encoding:utf-8

# This code is to pretrain model backbone on the base train data
import pdb
import os
import yaml
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from typing import Dict
from torch import Tensor
from tensorboardX import SummaryWriter
from src.model.pspnet import get_resnet

from .optimizer import get_optimizer, get_scheduler
from .dataset.dataset import get_val_loader
from .dataset.dataset_base import get_train_loader
from .util import intersection_and_union, AverageMeter
from .util import load_cfg_from_cfg_file, merge_cfg_from_list
from .util import ensure_path, set_log_path, log
from .test_21way import validate21ways, exp_validate
import argparse
from .classifier import SemiClassifier, StandardClassifier
from src.seghead.upernet import UPerNet
from src.models import SegmentModule,get_model_optimizer
from src.model.listencoder import ListEncoder
from src.mmpretrain.mmpretrain.apis.model import get_model
from src.dino_vision_transformer import vit_base
import matplotlib.pyplot as plt

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Training classifier weight transformer')
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg


def main(args: argparse.Namespace) -> None:

    sv_path = 'new_pretrain_{}/{}{}{}_{}/split{}_shot{}/{}{}{}'.format(
        args.data_name, "pretrain"if args.pretrained else "", args.arch, args.layers, args.seghead, args.split, args.shot, args.exp_name, "Episodic" if args.episodic_val else "", "Debug" if args.debug else "")
    sv_path = os.path.join('./results', sv_path)
    ensure_path(sv_path)
    set_log_path(path=sv_path)
    log('save_path {}'.format(sv_path))
    yaml.dump(args, open(os.path.join(sv_path, 'config.yaml'), 'w'))

    log(args)
    writer = SummaryWriter(os.path.join(sv_path, 'model'))

    if args.manual_seed is not None:
        cudnn.benchmark = False  # 为True的话可以对网络结构固定、网络的输入形状不变的 模型提速
        cudnn.deterministic = True
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)

    # =========Get model and optimizer========
    model,optimizer = get_model_optimizer(args)

    # ========== Validation ==================
    validate_fn = validate21ways if args.episodic_val else standard_validate
    log(f"==> Using {'episodic' if args.episodic_val else 'standard'} validation\n")

    # ========= Data  ==========
    train_loader, train_sampler = get_train_loader(args)
    log("{} {}".format(len(train_loader),train_loader.batch_size))
    val_loader = get_val_loader(args, episodic=args.episodic_val)  # mode='train' means that we will validate on images from validation set, but with the bases classes

    # ========== Scheduler  ================
    scheduler = get_scheduler(args, optimizer, len(train_loader))

    # ====== Metrics initialization ======
    max_val_mIoU = 0.
    iter_per_epoch = 10 if args.debug else len(train_loader)
    log_iter = 5 if args.debug else int(iter_per_epoch / args.log_freq) + 1

    metrics: Dict[str, Tensor] = {"val_mIou": torch.zeros((args.epochs, 1)).type(torch.float32),
                                  "val_loss": torch.zeros((args.epochs, 1)).type(torch.float32),
                                  "train_mIou": torch.zeros((args.epochs, log_iter)).type(torch.float32),
                                  "train_loss": torch.zeros((args.epochs, log_iter)).type(torch.float32),
                                  }

    # ====== Training  ======
    log('==> Start training')
    for epoch in range(args.epochs):

        loss_meter = AverageMeter()
        iterable_train_loader = iter(train_loader)

        for i in range(1, iter_per_epoch+1):
            model.train()

            images, gt = next(iterable_train_loader)  # q: [1, 3, 473, 473], s: [1, 1, 3, 473, 473]
            if torch.cuda.is_available():
                images = images.cuda()  # [1, 1, 3, h, w]
                gt = gt.cuda()  # [1, 1, h, w]

            loss = compute_loss(args=args, model=model, images=images, targets=gt.long(), num_classes=args.num_classes_tr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.scheduler == 'cosine':
                scheduler.step()

            if i % args.log_freq == 0:
                model.eval()
                logits = model(images)
                intersection, union, target = intersection_and_union(logits.argmax(1), gt, args.num_classes_tr, 255)  # gt [B, 473, 473]

                allAcc = (intersection.sum() / (target.sum() + 1e-10))  # scalar
                mAcc = (intersection / (target + 1e-10)).mean()
                mIoU = (intersection / (union + 1e-10)).mean()
                loss_meter.update(loss.item())

                log('iter {}/{}: loss {:.2f}, running loss {:.2f}, Acc {:.4f}, mAcc {:.4f}, mIoU {:.4f}'.format(
                    i, epoch, loss.item(), loss_meter.avg, allAcc, mAcc, mIoU))

        log('============ Epoch {}=============: running loss {:.2f}'.format(epoch, loss_meter.avg))
        writer.add_scalar('train_loss', loss_meter.avg, epoch)
        writer.add_scalar("mean_iou/train", mIoU, epoch)
        writer.add_scalar("pixel accuracy/train", allAcc, epoch)
        loss_meter.reset()

        # pdb.set_trace()
        if epoch % args.epi_frq == 0:
            val_Iou = validate_fn(args=args, val_loader=val_loader, model=model)
            writer.add_scalar("mean_iou/val", val_Iou, epoch)
            # writer.add_scalar("pixel accuracy/val", val_loss, epoch)

            # Model selection
            if val_Iou.item() > max_val_mIoU:
                max_val_mIoU = val_Iou.item()

                filename = os.path.join(sv_path, f'best.pth')
                if args.save_models:
                    log('=> Max_mIoU = {:.3f}, Saving checkpoint to: {}'.format(max_val_mIoU, filename))
                    torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict()}, filename)
        print("GPU memory usage: {} MB".format(torch.cuda.memory_allocated() / 1024 / 1024))

    if args.save_models:  # 所有跑完，存last epoch
        filename = os.path.join(sv_path, 'final.pth')
        torch.save( {'epoch': args.epochs, 'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}, filename )

    val_loader = get_val_loader(args, episodic=True)
    validate21ways(args=args, val_loader=val_loader, model=model,sv_path = sv_path)
    # exp_validate(args=args, val_loader=val_loader, model=model,sv_path = sv_path)


def cross_entropy(logits: torch.tensor, one_hot: torch.tensor, targets: torch.tensor, mean_reduce: bool = True,
                  ignore_index: int = 255) -> torch.tensor:
    """
    inputs: one_hot  : shape [batch_size, num_classes, h, w]
            logits : shape [batch_size, num_classes, h, w]
            targets : shape [batch_size, h, w]
    returns:loss: shape [batch_size] or [] depending on mean_reduce
    """
    # pdb.set_trace()
    assert logits.size() == one_hot.size()
    log_prb = F.log_softmax(logits, dim=1)
    non_pad_mask = targets.ne(ignore_index)
    loss = -(one_hot * log_prb).sum(dim=1)
    loss = loss.masked_select(non_pad_mask)
    if mean_reduce:
        return loss.mean()  # average later
    else:
        return loss


def compute_loss(args, model, images, targets, num_classes):
    """
    inputs:  images  : shape [batch_size, C, h, w]
             logits : shape [batch_size, num_classes, h, w]
             targets : shape [batch_size, h, w]
    returns: loss: shape []
             logits: shape [batch_size]
    """
    batch, h, w = targets.size()
    one_hot_mask = torch.zeros(batch, num_classes, h, w).cuda()
    new_target = targets.clone().unsqueeze(1)
    new_target[new_target == 255] = 0

    one_hot_mask.scatter_(1, new_target, 1).long()
    if args.smoothing:
        eps = 0.1
        one_hot = one_hot_mask * (1 - eps) + (1 - one_hot_mask) * eps / (num_classes - 1)
    else:
        one_hot = one_hot_mask  # [batch_size, num_classes, h, w]

    if args.mixup:
        alpha = 0.2
        lam = np.random.beta(alpha, alpha)
        rand_index = torch.randperm(images.size()[0]).cuda()
        one_hot_a = one_hot
        targets_a = targets

        one_hot_b = one_hot[rand_index]
        target_b = targets[rand_index]
        mixed_images = lam * images + (1 - lam) * images[rand_index]

        logits = model(mixed_images)
        loss = cross_entropy(logits, one_hot_a, targets_a) * lam  \
            + cross_entropy(logits, one_hot_b, target_b) * (1. - lam)
    else:
        logits = model(images)
        loss = cross_entropy(logits, one_hot, targets)
    return loss


def standard_validate(args, val_loader, model):

    loss_meter = AverageMeter()
    intersections = torch.zeros(args.num_classes_tr).cuda()
    unions = torch.zeros(args.num_classes_tr).cuda()

    model.eval()
    for i, (images, gt) in enumerate(val_loader):

        if torch.cuda.is_available():
            images = images.cuda()  # [1, 1, 3, h, w]
            gt = gt.cuda()          # [1, 1, h, w]

        with torch.no_grad():
            logits = model(images).detach()
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
            loss = loss_fn(logits, gt)
            loss_meter.update(loss.item())

        intersection, union, target = intersection_and_union(logits.argmax(1), gt, args.num_classes_tr, 255)  # gt [B, 473, 473]
        intersections += intersection  # [16]
        unions += union                # [16]

    mIoU = (intersections / (unions + 1e-10)).mean()
    acc = intersections.sum() / unions.sum()
    log(f'Testing results: running loss {loss_meter.avg:.2f}, Acc {acc:.4f}, mIoU {mIoU:.4f}\n')

    # return mIoU, loss_meter.avg
    return mIoU


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        args.epochs = 2
        args.batch_size = 8
        args.log_freq = 5
        args.batch_size_val = 8
        args.n_runs = 1
        args.test_num = 64
        args.epi_frq = 1
    if args.layers>50:
        args.batch_size = 4
    world_size = len(args.gpus)
    args.distributed = (world_size > 1)
    main(args)