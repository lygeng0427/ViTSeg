
import os
import time
import argparse
from tqdm import tqdm
from typing import Tuple

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from .classifier import DIaMClassifier
from .model.pspnet import get_pspnet
from .dataset.dataset import get_val_loader
import pdb
from .util import (
    setup,
    cleanup,
    get_cfg,
    setup_seed,
    get_model_dir,
    find_free_port,
    resume_random_state,
    fast_intersection_and_union,
    intersection_and_union,
    visualize,Masker
)
import random
import torch.nn.functional as F
from src.util import get_model_dir, fast_intersection_and_union, setup_seed, resume_random_state, find_free_port, setup, \
    cleanup, get_cfg, intersection_and_union, ensure_path, set_log_path, log, visualize, Masker
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    return get_cfg(parser)


def main_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    print(f"==> Running evaluation script")
    setup(args, rank, world_size)
    setup_seed(args.manual_seed)

    # ========== Data  ==========
    val_loader = get_val_loader(args)

    # ========== Model ==========
    model = get_pspnet(args).to(rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])

    root = get_model_dir(args)

    print("=> Creating the model")
    if args.ckpt_used is not None:
        filepath = os.path.join(root, f'{args.ckpt_used}.pth')
        assert os.path.isfile(filepath), filepath
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> Loaded weight '{}'".format(filepath))
    else:
        print("=> Not loading anything")

    # ========== Test  ==========
    validate(args=args, val_loader=val_loader, model=model)
    cleanup()


def validate(args: argparse.Namespace, val_loader: torch.utils.data.DataLoader, model: DDP) -> Tuple[torch.tensor, torch.tensor]:
    arch = args.arch + str(args.layers) if args.arch=='resnet' else args.arch
    sv_path = 'test_DIaM'
    sv_path = os.path.join('./results', sv_path)
    ensure_path(sv_path)
    set_log_path(path=sv_path)
    log('save_path {}'.format(sv_path))

    print('\n==> Start testing ({} runs)'.format(args.n_runs), flush=True)
    random_state = setup_seed(args.manual_seed, return_old_state=True)
    device = torch.device('cuda:{}'.format(dist.get_rank()))
    model.eval()

    c = model.module.bottleneck_dim
    h = model.module.feature_res[0]
    w = model.module.feature_res[1]


    visual_num, visual_max = 0, 7

    nb_episodes = len(val_loader) if args.test_num == -1 else int(args.test_num / args.batch_size_val)
    runtimes = torch.zeros(args.n_runs)
    base_mIoU, novel_mIoU = [torch.zeros(args.n_runs, device=device) for _ in range(2)]

    # ========== Perform the runs  ==========
    for run in range(args.n_runs):
        print('Run', run + 1, 'of', args.n_runs)
        # The order of classes in the following tensors is the same as the order of classifier (novels at last)
        cls_intersection = torch.zeros(args.num_classes_tr + args.num_classes_val)
        cls_union = torch.zeros(args.num_classes_tr + args.num_classes_val)
        cls_target = torch.zeros(args.num_classes_tr + args.num_classes_val)

        runtime = 0
        features_s, gt_s = None, None
        if not args.generate_new_support_set_for_each_task:
            with torch.no_grad():
                spprt_imgs, s_label = val_loader.dataset.generate_support([], remove_them_from_query_data_list=True)
                nb_episodes = len(val_loader) if args.test_num == -1 else nb_episodes  # Updates nb_episodes since some images were removed by generate_support
                spprt_imgs = spprt_imgs.to(device, non_blocking=True)
                s_label = s_label.to(device, non_blocking=True)
                features_s = model.module.extract_feat(spprt_imgs).detach().view((args.num_classes_val, args.shot, c, h, w))
                gt_s = s_label.view((args.num_classes_val, args.shot, args.image_size, args.image_size))

        for _ in tqdm(range(nb_episodes), leave=True):
            t0 = time.time()
            with torch.no_grad():
                try:
                    loader_output = next(iter_loader)
                except (UnboundLocalError, StopIteration):
                    iter_loader = iter(val_loader)
                    loader_output = next(iter_loader)
                qry_img, q_label, q_valid_pix, img_path = loader_output

                qry_img = qry_img.to(device, non_blocking=True)
                q_label = q_label.to(device, non_blocking=True)
                features_q = model.module.extract_feat(qry_img).detach().unsqueeze(1)
                valid_pixels_q = q_valid_pix.unsqueeze(1).to(device)
                gt_q = q_label.unsqueeze(1)

                query_image_path_list = list(img_path)
                if args.generate_new_support_set_for_each_task:
                    spprt_imgs, s_label = val_loader.dataset.generate_support(query_image_path_list)
                    spprt_imgs = spprt_imgs.to(device, non_blocking=True)
                    s_label = s_label.to(device, non_blocking=True)
                    features_s = model.module.extract_feat(spprt_imgs).detach().view((args.num_classes_val, args.shot, c, h, w))
                    gt_s = s_label.view((args.num_classes_val, args.shot, args.image_size, args.image_size))

            # =========== Initialize the classifier and run the method ===============
            base_weight = model.module.classifier.weight.detach().clone().T
            base_bias = model.module.classifier.bias.detach().clone()
            # print("base_weight: ", base_weight.shape, "base_bias: ", base_bias.shape)
            # print("features_s: ", features_s.shape, "gt_s: ", gt_s.shape)
            # print("features_q: ", features_q.shape, "gt_q: ", gt_q.shape)
            # print("valid_q: ", valid_q.shape, "gt_q: ", gt_q.shape)
            classifier = DIaMClassifier(args, base_weight, base_bias, n_tasks=features_q.size(0))
            classifier.init_prototypes(features_s, gt_s)
            classifier.compute_pi(features_q, valid_pixels_q, gt_q)  # gt_q won't be used in optimization if pi estimation strategy is self or uniform
            classifier.optimize(features_s, features_q, gt_s, valid_pixels_q)

            runtime += time.time() - t0

            # =========== Perform inference and compute metrics ===============
            logits = classifier.get_logits(features_q).detach()
            probas = classifier.get_probas(logits)

            for i in range(qry_img.shape[0]):
                if visual_num<visual_max:
                    if random.random()<0.05:
                        q_pred = probas.squeeze(1).argmax(1)
                        qry_img = qry_img.clone().to("cpu").detach()
                        q_label = q_label.clone().to("cpu").detach()
                        q_pred = q_pred.clone().to("cpu").detach()
                        visual_num+=1
                        q_mask = F.interpolate(q_label.unsqueeze(1).float(), size=qry_img.shape[-2:], mode='nearest').squeeze(1)  # [1,1,h,w]
                        q_pred = F.interpolate(q_pred.unsqueeze(1).float(), size=qry_img.shape[-2:], mode='nearest').squeeze(1)  # [1,1,h,w]
                        visualize(qry_img[i],sv_path,f"{visual_num}IMG.png")

                        masker = Masker()
                        mask = masker.mask_to_rgb(q_mask[i].squeeze())
                        # plt.imshow(mask)
                        plt.imsave(sv_path+f"/{visual_num}LABEL.png", mask)
                        mask = masker.mask_to_rgb(q_pred[i].squeeze())
                        # plt.imshow(mask)
                        plt.imsave(sv_path+f"/{visual_num}PRED.png", mask)

            intersection, union, target = fast_intersection_and_union(probas, gt_q)  # [batch_size_val, 1, num_classes]
            intersection, union, target = intersection.squeeze(1).cpu(), union.squeeze(1).cpu(), target.squeeze(1).cpu()
            cls_intersection += intersection.sum(0)
            cls_union += union.sum(0)
            cls_target += target.sum(0)

        base_count, novel_count, sum_base_IoU, sum_novel_IoU = 4 * [0]
        for i, class_ in enumerate(val_loader.dataset.all_classes):
            if cls_union[i] == 0:
                continue
            IoU = cls_intersection[i] / (cls_union[i] + 1e-10)
            print("Class {}: \t{:.4f}".format(class_, IoU))
            if class_ in val_loader.dataset.base_class_list:
                sum_base_IoU += IoU
                base_count += 1
            elif class_ in val_loader.dataset.novel_class_list:
                sum_novel_IoU += IoU
                novel_count += 1

        avg_base_IoU, avg_novel_IoU = sum_base_IoU / base_count, sum_novel_IoU / novel_count
        print('Mean base IoU: {:.4f}, Mean novel IoU: {:.4f}'.format(avg_base_IoU, avg_novel_IoU), flush=True)

        base_mIoU[run], novel_mIoU[run] = avg_base_IoU, avg_novel_IoU
        runtimes[run] = runtime

    agg_mIoU = (base_mIoU.mean() + novel_mIoU.mean()) / 2
    print('==>')
    print('Average of base mIoU: {:.4f}\tAverage of novel mIoU: {:.4f} \t(over {} runs)'.format(
        base_mIoU.mean(), novel_mIoU.mean(), args.n_runs))
    print('Mean --- {:.4f}'.format(agg_mIoU), flush=True)
    print('Average runtime / run --- {:.1f}\n'.format(runtimes.mean()))

    resume_random_state(random_state)
    return agg_mIoU


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpus)
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    if args.debug:
        args.test_num = 64
        args.n_runs = 2
        args.batch_size_val = 8
    world_size = len(args.gpus)
    distributed = world_size > 1
    assert not distributed, 'Testing should not be done in a distributed way'
    args.distributed = distributed
    args.port = find_free_port()
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    mp.spawn(main_worker,
             args=(world_size, args),
             nprocs=world_size,
             join=True)
