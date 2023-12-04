import pdb
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
import torch.utils.data
import torch.nn.parallel
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from src.dataset.dataset import get_val_loader
from src.model.model_factory import get_model, create_decoder
from src.classifier_semi import FullClassifier, train_decoder
from src.util import load_pretrain_weight, load_base_weight,  setup_seed, resume_random_state, find_free_port, \
     get_cfg, intersection_and_union, ensure_path, set_log_path, log

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    return get_cfg(parser)


def main_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    arch = args.arch + str(args.layers) if args.arch == 'resnet' else args.arch
    sv_path = 'test_{}/{}/shot{}_split{}/{}'.format(args.data_name, arch, args.shot, args.split, args.exp_name)
    sv_path = os.path.join('./results', sv_path)
    ensure_path(sv_path)
    set_log_path(path=sv_path)
    log('save_path {}'.format(sv_path))

    log(args)

    log(f"==> Running evaluation script")
    # setup(args, rank, world_size)
    setup_seed(args.manual_seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # ========== Data  ==========
    val_loader = get_val_loader(args, episodic=True)
    val_loader1 = get_val_loader(args, episodic=False)
    # semi_val_loader = get_semi_val_loader(args)

    # ========== Model  ==========
    model = get_model(args).to(device)
    #pdb.set_trace()
    model = load_pretrain_weight(model, args)

    # ========== Test  ==========
    episodic_validate(args=args, val_loader=val_loader, model=model, val_loader1=val_loader1)
    # cleanup()


def episodic_validate(args, val_loader, model, val_loader1=None):

    model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    nb_episodes = len(val_loader) if args.test_num == -1 else int(args.test_num / args.batch_size_val)
    runtimes = torch.zeros(args.n_runs)
    novel_mIoU = torch.zeros(args.n_runs, device=device)
    base_mIoU = torch.zeros(args.n_runs, device=device)

    # ========== Perform the runs  ==========
    for run in range(args.n_runs):
        log('===>Run {} / {}'.format(run + 1, args.n_runs))

        # The order of classes in the following tensors is the same as the order of classifier (novels at last)
        cls_intersection = torch.zeros(args.num_classes_tr + args.num_classes_val)
        cls_union = torch.zeros(args.num_classes_tr + args.num_classes_val)
        
        cls_intersection1 = torch.zeros(args.num_classes_tr + args.num_classes_val)
        cls_union1 = torch.zeros(args.num_classes_tr + args.num_classes_val)
        

        if not args.generate_new_support_set_for_each_task:
            with torch.no_grad():
                s_imgs, s_label = val_loader.dataset.generate_support([], remove_them_from_query_data_list=True)
                nb_episodes = len(val_loader) if args.test_num == -1 else nb_episodes  # Updates nb_episodes since some images were removed by generate_support
                s_imgs = s_imgs.to(device, non_blocking=True)
                s_label = s_label.to(device, non_blocking=True)
                s_feat = model.extract_features(s_imgs)

        run_start_time = time.time()
        for _ in tqdm(range(nb_episodes), leave=True):

            with torch.no_grad():
                try:
                    loader_output = next(iter_loader)
                except:
                    iter_loader = iter(val_loader)
                    loader_output = next(iter_loader)
                q_img, q_label, q_valid_pix, img_path = loader_output
                
                try:
                    loader_output1 = next(iter_loader1)
                except:
                    iter_loader1 = iter(val_loader1)
                    loader_output1 = next(iter_loader1)
                q_img1, q_label1 = loader_output1
                q_img1, q_label1 = q_img1.cuda(), q_label1.cuda()
                q_feat1 = model.extract_features(q_img1)
                

                q_img = q_img.to(device, non_blocking=True)
                q_label = q_label.to(device, non_blocking=True)
                q_valid_pix = q_valid_pix.to(device, non_blocking=True)
                q_feat = model.extract_features(q_img)

                if args.generate_new_support_set_for_each_task:
                    query_image_path_list = list(img_path)
                    s_imgs, s_label = val_loader.dataset.generate_support(query_image_path_list)
                    s_imgs = s_imgs.to(device, non_blocking=True)
                    s_label = s_label.to(device, non_blocking=True)
                    s_feat = model.extract_features(s_imgs)  # dino & PSPNet output different shapes

            # ===========  train classifier with support set ==============
            if args.arch == 'resnet':
                decoder = FullClassifier(args, feat_dim=s_feat.shape[1]).to(device)
            else:
                decoder = create_decoder(args, d_model=768, patch_size=args.patch_size, num_classes=args.num_classes_val + args.num_classes_tr).to(device)
            decoder = load_base_weight(decoder, model, args)

            decoder.train()
            train_decoder(decoder, s_feat, s_label.clone(), iterations=100, lr=0.01, args=args)

            # =========== Perform inference and compute metrics ===============
            decoder.eval()
            with torch.no_grad():
                q_logit = decoder(q_feat, q_label.shape[-2:])
                q_pred = q_logit.argmax(1)   # [batch_size_val, H, W]
                
                q_logit1 = model.decoder(q_feat, q_label.shape[-2:])
                q_pred1 = q_logit1.argmax(1)
            
            # pdb.set_trace()

            intersection, union, target = intersection_and_union(q_pred, q_label, num_classes=args.num_classes_val + args.num_classes_tr,
                                                                 ignore_index=255)
            intersection, union, target = intersection.cpu(), union.cpu(), target.cpu()
            cls_intersection += intersection
            cls_union += union
            
            # print(cls_intersection/(cls_union+1e-10))
            
            intersection, union, target = intersection_and_union(q_pred1, q_label, num_classes=args.num_classes_val + args.num_classes_tr, ignore_index=255)
            intersection, union, target = intersection.cpu(), union.cpu(), target.cpu()
            cls_intersection1 += intersection
            cls_union1 += union
    
        a0 = cls_intersection/(cls_union+1e-10)
        a1 = cls_intersection1/(cls_union1+1e-10)
        print('--performance of new  decoder--{}'.format(torch.mean(a0[:args.num_classes_tr])))
        print(a0)
        print('--performance of base decoder--{}'.format(torch.mean(a1[:args.num_classes_tr])))
        print(a1)


        base_count, novel_count, sum_base_IoU, sum_novel_IoU = 4 * [0]
        classwise_msg = ''
        for i, class_ in enumerate(val_loader.dataset.all_classes):
            if cls_union[i] == 0:
                continue
            IoU = cls_intersection[i] / (cls_union[i] + 1e-10)
            classwise_msg += 'Class {}: {:.4f}|'.format(class_, IoU)

            if class_ in val_loader.dataset.base_class_list:
                sum_base_IoU += IoU
                base_count += 1
            elif class_ in val_loader.dataset.novel_class_list:
                sum_novel_IoU += IoU
                novel_count += 1

        runtime = time.time() - run_start_time
        avg_base_IoU, avg_novel_IoU = sum_base_IoU / base_count, sum_novel_IoU / novel_count
        base_mIoU[run], novel_mIoU[run] = avg_base_IoU, avg_novel_IoU
        runtimes[run] = runtime
        log('Run {}, Avg base IoU: {:.4f}, Avg novel IoU {:.4f}'.format(run + 1,avg_base_IoU, avg_novel_IoU))
        log('\t Classwise IoU: ' + classwise_msg)

    log('==> Multi-Run: Avg base mIoU: {:.4f}, Avg novel mIoU: {:.4f}, Mean: {:.4f} -- over {} runs'.format(
        base_mIoU.mean(), novel_mIoU.mean(), (base_mIoU.mean() + novel_mIoU.mean())/2, args.n_runs))

    return (base_mIoU.mean() + novel_mIoU.mean())/2


if __name__ == "__main__":
    args = parse_args()
    if args.arch == 'dinov2':
        args.image_size = 448
        args.patch_size = 14
    elif args.arch == 'dino':
        args.image_size = 448
        args.patch_size = 16
    if args.debug:
        args.test_num = 64
        args.n_runs = 2

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpus)
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    world_size = len(args.gpus)
    distributed = world_size > 1
    assert not distributed, 'Testing should not be done in a distributed way'
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



