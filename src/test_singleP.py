import argparse
import os
import time
from tqdm import tqdm
from PIL import Image
from typing import Tuple


import os
import pdb
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from src.classifier import SemiClassifier, train_semi
from src.dataset.dataset import get_val_loader
from src.dataset.dataset_semi import get_semi_val_loader
from src.model.pspnet import get_model
from src.util import get_model_dir, fast_intersection_and_union, setup_seed, resume_random_state, find_free_port, setup, \
    cleanup, get_cfg, intersection_and_union, ensure_path, set_log_path, log


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    return get_cfg(parser)


def main_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    arch = args.arch + str(args.layers) if args.arch=='resnet' else args.arch
    sv_path = 'test_{}/{}/shot{}_split{}/{}'.format(args.data_name, arch, args.shot, args.split, args.exp_name)
    sv_path = os.path.join('./results', sv_path)
    ensure_path(sv_path)
    set_log_path(path=sv_path)
    log('save_path {}'.format(sv_path))

    log(args)

    log(f"==> Running evaluation script")
    # setup(args, rank, world_size)
    setup_seed(args.manual_seed)

    # ========== Data  ==========
    val_loader = get_val_loader(args)
    # semi_val_loader = get_semi_val_loader(args)

    # ========== Model  ==========
    if args.arch == 'dino':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').cuda()
    else: 
        model = get_model(args).to('cuda')
        print("=> Creating the model")
        if args.ckpt_used is not None:
            root = get_model_dir(args)
            filepath = os.path.join(root, f'{args.ckpt_used}.pth')
            assert os.path.isfile(filepath), filepath
            checkpoint = torch.load(filepath)
            # checkpoint = torch.load(filepath, map_location=lambda storage, location: storage)
            model_weight = {}
            for name, params in checkpoint['state_dict'].items():
                name = name[7:]
                model_weight[name] = params
            model.load_state_dict(model_weight)
            log("=> Loaded weight '{}'".format(filepath))
        else:
            log("=> Not loading anything")

    # ========== Test  ==========
    validate(args=args, val_loader=val_loader, model=model)
    # cleanup()


def validate(args: argparse.Namespace, val_loader: torch.utils.data.DataLoader, model: DDP) -> Tuple[torch.tensor, torch.tensor]:
    log('\n==> Start testing ({} runs)'.format(args.n_runs))
    random_state = setup_seed(args.manual_seed, return_old_state=True)

    device = torch.device('cuda')
    model.eval()

    nb_episodes = len(val_loader) if args.test_num == -1 else int(args.test_num / args.batch_size_val)
    runtimes = torch.zeros(args.n_runs)
    novel_mIoU = torch.zeros(args.n_runs, device=device)

    # ========== Perform the runs  ==========
    for run in range(args.n_runs):
        print('Run', run + 1, 'of', args.n_runs)

        # The order of classes in the following tensors is the same as the order of classifier (novels at last)
        cls_intersection = torch.zeros(1 + args.num_classes_val)
        cls_union = torch.zeros(1 + args.num_classes_val)
        cls_target = torch.zeros(1 + args.num_classes_val)
        cls_IOU = torch.zeros(1 + args.num_classes_val)
        cls_count = torch.zeros(1 + args.num_classes_val)

        if not args.generate_new_support_set_for_each_task:
            with torch.no_grad():
                s_imgs, s_label = val_loader.dataset.generate_support([], remove_them_from_query_data_list=True)
                nb_episodes = len(val_loader) if args.test_num == -1 else nb_episodes  # Updates nb_episodes since some images were removed by generate_support
                s_imgs = s_imgs.to(device, non_blocking=True)
                s_label = s_label.to(device, non_blocking=True)

                if args.arch == 'dino':
                    s_out = model.forward_features(s_imgs)
                    s_feat = s_out['x_norm_patchtokens']
                    s_feat = s_feat.view(args.num_classes_val*args.shot, args.image_size//args.patch_size, args.image_size//args.patch_size, -1)
                    s_feat = s_feat.permute(0, 3, 1, 2)
                else: 
                    s_feat = model.extract_features(s_imgs)

        t0 = time.time()
        for _ in tqdm(range(nb_episodes), leave=True):

            with torch.no_grad():
                try:
                    loader_output = next(iter_loader)
                except:
                    iter_loader = iter(val_loader)
                    loader_output = next(iter_loader)
                q_img, q_label, q_valid_pix, img_path = loader_output

                q_img = q_img.to(device, non_blocking=True)
                q_label = q_label.to(device, non_blocking=True)
                q_valid_pix = q_valid_pix.to(device, non_blocking=True)

                if args.arch == 'dino':
                    q_out = model.forward_features(q_img)
                    q_feat = q_out['x_norm_patchtokens']
                    q_feat = q_feat.view(q_img.size(0), args.image_size // args.patch_size, args.image_size // args.patch_size, -1)
                    q_feat = q_feat.permute(0, 3, 1, 2)
                else: 
                    q_feat = model.extract_features(q_img)

                query_image_path_list = list(img_path)
                if args.generate_new_support_set_for_each_task:
                    s_imgs, s_label = val_loader.dataset.generate_support(query_image_path_list)
                    s_imgs = s_imgs.to(device, non_blocking=True)
                    s_label = s_label.to(device, non_blocking=True)

                    if args.arch == 'dino': 
                        s_out = model.forward_features(s_imgs)
                        s_feat = s_out['x_norm_patchtokens']
                        s_feat = s_feat.view(args.num_classes_val * args.shot, args.image_size // args.patch_size, args.image_size // args.patch_size, -1)
                        s_feat = s_feat.permute(0, 3, 1, 2)
                    else: 
                        s_feat = model.extract_features(s_imgs)

            # ===========  semi-supervised learning method ==============
            
            '''Use n_shot support  image to train a classifier for pseudo-labelling'''
            classifier_semi = SemiClassifier(args, feat_dim=s_feat.shape[1]).to(device)
            classifier_semi.train()
            train_semi(classifier_semi, s_feat, s_label.clone(), iterations=100, lr_semi=0.01)

            # =========== Perform inference and compute metrics ===============
            classifier_semi.eval()
            with torch.no_grad():
                q_logit = classifier_semi(q_feat, q_label.shape[-2:])
                q_pred = q_logit.argmax(1)

            q_label[torch.where(q_label <= 15)] =0
            q_label[torch.where((q_label > 15) & (q_label < 255))] -= 15

            intersection, union, target = intersection_and_union(q_pred,  q_label, num_classes=args.num_classes_val+1, ignore_index = 255)
            intersection, union, target = intersection.cpu(), union.cpu(), target.cpu()
            cls_intersection += intersection
            cls_union += union
            cls_target += target

        novel_count, sum_novel_IoU = 2 * [0]
        classwise_msg = ''
        for i, class_ in enumerate([0] + val_loader.dataset.novel_class_list):
            if cls_union[i] == 0:
                continue
            IoU = cls_intersection[i] / (cls_union[i] + 1e-10)
            cls_IOU[i] += IoU
            cls_count[i] += 1
            # log("Class {}: \t{:.4f}".format(class_, IoU))
            classwise_msg += 'Class {}: {:.4f}|'.format(class_, IoU)

            if class_ in val_loader.dataset.novel_class_list:
                sum_novel_IoU += IoU
                novel_count += 1

        runtime = time.time() - t0
        avg_novel_IoU = sum_novel_IoU / novel_count
        log('==> Run {}, Mean novel IoU: {:.4f}'.format(run+1, avg_novel_IoU))
        log('\t Classwise IoU: ' + classwise_msg)

        novel_mIoU[run] = avg_novel_IoU
        runtimes[run] = runtime

    for i, class_ in enumerate([0] + val_loader.dataset.novel_class_list):
        log("Class {}: \t{:.4f}".format(class_, cls_IOU[i]/cls_count[i]))
    log('==> ')
    log('Average of novel mIoU: {:.4f} \t(over {} runs)'.format(novel_mIoU.mean(), args.n_runs))
    log('Average runtime / run --- {:.1f}\n'.format(runtimes.mean()))

    resume_random_state(random_state)
    return novel_mIoU.mean()


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
    # args.distributed = distributed
    args.distributed = False
    args.port = find_free_port()
    args.port = None
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    # mp.spawn(main_worker,
    #          args=(world_size, args),
    #          nprocs=world_size,
    #          join=True)

    main_worker(rank=0, world_size=0, args=args)



