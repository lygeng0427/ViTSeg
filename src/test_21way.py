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


from src.models import SegmentModule,get_model_optimizer
from src.classifier import DIaMClassifier,SemiClassifier
from src.seghead.upernet import UPerNet
import torch.nn.functional as F
from src.dataset.dataset import get_val_loader
from src.dataset.dataset_semi import get_semi_val_loader
from src.model.pspnet import get_resnet
from src.util import get_model_dir, fast_intersection_and_union, setup_seed, resume_random_state, find_free_port, setup, \
    cleanup, get_cfg, intersection_and_union, ensure_path, set_log_path, log, visualize, Masker

import random

from src.model.listencoder import ListEncoder

import matplotlib.pyplot as plt


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    return get_cfg(parser)


def main_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    arch = args.arch + str(args.layers) if args.arch=='resnet' else args.arch
    sv_path = 'test_{}/{}/shot{}_split{}/{}{}{}'.format(args.data_name, arch, args.shot, args.split, args.exp_name, args.clsfer,"debug" if args.debug else "")
    sv_path = os.path.join('./0CV_reports', sv_path)
    ensure_path(sv_path)
    set_log_path(path=sv_path)
    log('save_path {}'.format(sv_path))

    log(args)

    log(f"==> Running evaluation script")
    # setup(args, rank, world_size)
    setup_seed(args.manual_seed)

    # ========== Model  ==========

    if args.arch == 'dino':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').cuda()
    else: 
        model,_ = get_model_optimizer(args)
        model = model.to("cuda")
        print("=> Creating the model")
        if args.ckpt_used is not None:
            root = get_model_dir(args)
            filepath = os.path.join(root, f'{args.ckpt_used}.pth')
            assert os.path.isfile(filepath), filepath
            checkpoint = torch.load(filepath)
            # checkpoint = torch.load(filepath, map_location=lambda storage, location: storage)
            model.load_state_dict(checkpoint["state_dict"])
            # model_weight = {}
            # for name, params in checkpoint['state_dict'].items():
            #     name = name[7:]
            #     model_weight[name] = params
            # model.load_state_dict(model_weight)
            # log("=> Loaded weight '{}'".format(filepath))
        else:
            log("=> Not loading anything")
    
    # ========== Data  ==========
    val_loader = get_val_loader(args)
    # semi_val_loader = get_semi_val_loader(args)

    # ========== Test  ==========
    validate21ways(args=args, val_loader=val_loader, model=model, sv_path = None)
    # exp_validate(args=args, val_loader=val_loader, model=model,sv_path = sv_path)
    # cleanup()


def validate21ways(args: argparse.Namespace, val_loader: torch.utils.data.DataLoader, model: DDP, sv_path = None) -> Tuple[torch.tensor, torch.tensor]:
    log('\n==> Start testing ({} runs)'.format(args.n_runs))
    random_state = setup_seed(args.manual_seed, return_old_state=True)

    device = torch.device('cuda')
    model.eval()

    nb_episodes = len(val_loader) if args.test_num == -1 else int(args.test_num / args.batch_size_val)
    runtimes = torch.zeros(args.n_runs)
    novel_mIoU = torch.zeros(args.n_runs, device=device)
    base_mIoU = torch.zeros(args.n_runs, device=device)

    visual_num = 0
    visual_max = 7
    # ========== Perform the runs  ==========
    for run in range(args.n_runs):
        print('Run', run + 1, 'of', args.n_runs)

        # The order of classes in the following tensors is the same as the order of classifier (novels at last)
        cls_intersection = torch.zeros(args.num_classes_tr + args.num_classes_val)
        cls_union = torch.zeros(args.num_classes_tr + args.num_classes_val)
        cls_target = torch.zeros(args.num_classes_tr + args.num_classes_val)
        cls_IOU = torch.zeros(args.num_classes_tr + args.num_classes_val)
        cls_count = torch.zeros(args.num_classes_tr + args.num_classes_val)

        if not args.generate_new_support_set_for_each_task:
            with torch.no_grad():
                s_imgs, s_label = val_loader.dataset.generate_support([], remove_them_from_query_data_list=True)
                nb_episodes = len(val_loader) if args.test_num == -1 else nb_episodes  # Updates nb_episodes since some images were removed by generate_support
                s_imgs = s_imgs.to(device, non_blocking=True)
                s_label = s_label.to(device, non_blocking=True)
                gt_s = s_label.view((args.num_classes_val, args.shot, args.image_size, args.image_size))

                s_feat = model.extract_feat(s_imgs)

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

                q_feat = model.extract_feat(q_img)

                query_image_path_list = list(img_path)
                if args.generate_new_support_set_for_each_task:
                    s_imgs, s_label = val_loader.dataset.generate_support(query_image_path_list)
                    s_imgs = s_imgs.to(device, non_blocking=True)
                    s_label = s_label.to(device, non_blocking=True)
                    gt_s = s_label.view((args.num_classes_val, args.shot, args.image_size, args.image_size))

                    s_feat = model.extract_feat(s_imgs)

            # ===========  semi-supervised learning method ==============
            base_weight = model.classifier.conv1.weight.detach().clone()
            base_bias = model.classifier.conv1.bias.detach().clone()
            # classifier = Classifier21(args, feat_dim=s_feat.shape[1]).to(device)
            if args.seghead == "None": 
                if args.arch == "pspnet":
                    in_dim = 512
                elif args.arch == "resnet":
                    in_dim = 2048
                    if args.layers<=34:
                        in_dim = 512
                else:
                    in_dim = 768
            elif args.seghead == "UPerNet":
                in_dim = 256

            classifier = SemiClassifier(classes = args.num_classes_tr + args.num_classes_val, feat_dim=in_dim).to(device)
            classifier.init_weight(base_weight, base_bias,args.num_classes_val, channel_num = in_dim)
            classifier.train_val(s_feat, gt_s.clone(), args.num_classes_tr, args.num_classes_val, iterations=200,
                        lr_semi=0.01)
            # =========== Perform inference and compute metrics ===============
            print("GPU memory usage: {} MB".format(torch.cuda.memory_allocated() / 1024 / 1024))
            classifier.eval()
            with torch.no_grad():
                q_logit = classifier(q_feat, q_label.shape[-2:])
                q_pred = q_logit.argmax(1)
            # pdb.set_trace()
            # q_label[torch.where(q_label <= 15)] =0
            # q_label[torch.where((q_label > 15) & (q_label < 255))] -= 15

            intersection, union, target = intersection_and_union(q_pred,  q_label, num_classes=args.num_classes_val+args.num_classes_tr, ignore_index = 255)
            intersection, union, target = intersection.cpu(), union.cpu(), target.cpu()
            cls_intersection += intersection
            cls_union += union
            cls_target += target

        base_count, novel_count, sum_base_IoU, sum_novel_IoU = 4 * [0]
        classwise_msg = ''
        for i, class_ in enumerate(val_loader.dataset.all_classes):
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
            if class_ in val_loader.dataset.base_class_list:
                sum_base_IoU += IoU
                base_count += 1

        runtime = time.time() - t0
        avg_novel_IoU = sum_novel_IoU / novel_count
        avg_base_IoU = sum_base_IoU / base_count
        log('==> Run {}, Mean novel IoU: {:.4f}, Mean base IoU: {:.4f}'.format(run+1, avg_novel_IoU,avg_base_IoU))
        log('\t Classwise IoU: ' + classwise_msg)

        novel_mIoU[run] = avg_novel_IoU
        base_mIoU[run] = avg_base_IoU
        runtimes[run] = runtime

    for i, class_ in enumerate(val_loader.dataset.all_classes):
        log("Class {}: \t{:.8f}".format(class_, cls_IOU[i]/cls_count[i]))
    log('==> ')
    log('Average of novel mIoU: {:.4f} \t(over {} runs)'.format(novel_mIoU.mean(), args.n_runs))
    log('Average of base mIoU: {:.4f} \t(over {} runs)'.format(base_mIoU.mean(), args.n_runs))
    log('Average runtime / run --- {:.1f}\n'.format(runtimes.mean()))

    resume_random_state(random_state)
    return novel_mIoU.mean()


def exp_validate(args: argparse.Namespace, val_loader: torch.utils.data.DataLoader, model: DDP, sv_path) -> Tuple[torch.tensor, torch.tensor]:
    log('\n==> Start testing ({} runs)'.format(args.n_runs))
    random_state = setup_seed(args.manual_seed, return_old_state=True)

    device = torch.device('cuda')
    model.eval()

    nb_episodes = len(val_loader) if args.test_num == -1 else int(args.test_num / args.batch_size_val)
    runtimes = torch.zeros(args.n_runs)
    novel_mIoU = torch.zeros(args.n_runs, device=device)
    base_mIoU = torch.zeros(args.n_runs, device=device)

    visual_num = 0
    visual_max = 7
    # ========== Perform the runs  ==========
    for run in range(args.n_runs):
        print('Run', run + 1, 'of', args.n_runs)

        # The order of classes in the following tensors is the same as the order of classifier (novels at last)
        cls_intersection = torch.zeros(args.num_classes_tr + args.num_classes_val)
        cls_union = torch.zeros(args.num_classes_tr + args.num_classes_val)
        cls_target = torch.zeros(args.num_classes_tr + args.num_classes_val)
        cls_IOU = torch.zeros(args.num_classes_tr + args.num_classes_val)
        cls_count = torch.zeros(args.num_classes_tr + args.num_classes_val)

        if not args.generate_new_support_set_for_each_task:
            with torch.no_grad():
                s_imgs, s_label = val_loader.dataset.generate_support([], remove_them_from_query_data_list=True)
                nb_episodes = len(val_loader) if args.test_num == -1 else nb_episodes  # Updates nb_episodes since some images were removed by generate_support
                s_imgs = s_imgs.to(device, non_blocking=True)
                s_label = s_label.to(device, non_blocking=True)
                gt_s = s_label.view((args.num_classes_val, args.shot, args.image_size, args.image_size))


                s_feat = model.extract_feat(s_imgs)

        

            # ===========  semi-supervised learning method ==============
            
            '''Use n_shot support  image to train a classifier for pseudo-labelling'''
        base_weight = model.classifier.conv1.weight.detach().clone()
        base_bias = model.classifier.conv1.bias.detach().clone()
        if args.seghead == "None":
            if args.arch == "pspnet":
                in_dim = 512
            elif args.arch == "resnet":
                in_dim = 2048
            else:
                in_dim = 768
        elif args.seghead == "UPerNet":
            in_dim = 256

        #track IoU, loss during iterations for visualization
        epi_novel_IoU = torch.zeros(50)
        epi_base_IoU = torch.zeros(50)
        rep = 0
        freq = 20 if not args.debug else 100

        #===========start meta-testing epoch============
        for it in range(0,201,freq):            
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
                    q_feat = model.extract_feat(q_img)
                    query_image_path_list = list(img_path)
                
                    if args.generate_new_support_set_for_each_task:
                        s_imgs, s_label = val_loader.dataset.generate_support(query_image_path_list)
                        s_imgs = s_imgs.to(device, non_blocking=True)
                        s_label = s_label.to(device, non_blocking=True)
                        gt_s = s_label.view((args.num_classes_val, args.shot, args.image_size, args.image_size))

                        s_feat = model.extract_feat(s_imgs)
                if args.clsfer == "DIaM":
                    gt_q = q_label.unsqueeze(1)
                    base_weight = model.classifier.conv1.weight.detach().clone().T
                    base_bias = model.classifier.conv1.bias.detach().clone()

                    q_feat = q_feat.unsqueeze(1)
                    s_feat = s_feat.unsqueeze(1)
                    valid_pixels_q = q_valid_pix.unsqueeze(1)
                    classifier = DIaMClassifier(args, base_weight, base_bias, n_tasks=q_feat.size(0))
                    classifier.init_prototypes(s_feat, gt_s)
                    classifier.compute_pi(q_feat, valid_pixels_q, gt_q)  # gt_q won't be used in optimization if pi estimation strategy is self or uniform
                    classifier.optimize(s_feat, q_feat, gt_s, valid_pixels_q, iterations = it)
                elif args.clsfer == "Linear":
                    classifier = SemiClassifier(classes = args.num_classes_tr + args.num_classes_val, feat_dim=in_dim).to(device)
                    classifier.init_weight(base_weight, base_bias,args.num_classes_val, channel_num = in_dim)
                    loss = classifier.train_val(s_feat, gt_s.clone(), args.num_classes_tr, args.num_classes_val, iterations=it,
                                lr_semi=0.01)
                # s_imgs, s_label = val_loader.dataset.generate_support([], remove_them_from_query_data_list=True)
                # nb_episodes = len(val_loader) if args.test_num == -1 else nb_episodes  # Updates nb_episodes since some images were removed by generate_support
                # s_imgs = s_imgs.to(device, non_blocking=True)
                # s_label = s_label.to(device, non_blocking=True)
                # gt_s = s_label.view((args.num_classes_val, args.shot, args.image_size, args.image_size))

                # =========== Perform inference and compute metrics ===============
                print("GPU memory usage: {} MB".format(torch.cuda.memory_allocated() / 1024 / 1024))
                # classifier.eval() #diam has no eval()
                with torch.no_grad():
                    if args.clsfer == "DIaM":
                        q_logit = classifier.get_logits(q_feat).detach()
                        q_proba = classifier.get_probas(q_logit)
                        intersection, union, target = fast_intersection_and_union(q_proba, gt_q)  # [batch_size_val, 1, num_classes]
                        intersection, union, target = intersection.squeeze(1).cpu(), union.squeeze(1).cpu(), target.squeeze(1).cpu()
                        cls_intersection += intersection.sum(0)
                        cls_union += union.sum(0)
                        cls_target += target.sum(0)
                        # print("q_proba:", q_proba.shape)
                        q_pred = q_proba.squeeze(1).argmax(1)
                        # pdb.set_trace()
                    else:
                        q_logit = classifier(q_feat, q_label.shape[-2:])
                        # q_logit = new_model.classifier(new_model.segmenthead(q_feat), q_label.shape[-2:])
                        q_pred = q_logit.argmax(1)
                        intersection, union, target = intersection_and_union(q_pred,  q_label, num_classes=args.num_classes_val+args.num_classes_tr, ignore_index = 255)
                        intersection, union, target = intersection.cpu(), union.cpu(), target.cpu()
                        cls_intersection += intersection
                        cls_union += union
                        cls_target += target

                    q_img = q_img.clone().to("cpu").detach()
                    q_label = q_label.clone().to("cpu").detach()
                    q_pred = q_pred.clone().to("cpu").detach()
                    for i in range(q_img.shape[0]):
                        if it%50 == 0:
                            id = (int(q_img[i].sum().item())%1024)
                            imgname = (query_image_path_list[i][-7:])
                            if id<10:
                                svim_path = f"{sv_path}/img/{imgname}"
                                if not os.path.exists(svim_path):
                                    ensure_path(svim_path)
                                visual_num+=1
                                q_mask = F.interpolate(q_label.unsqueeze(1).float(), size=q_img.shape[-2:], mode='nearest').squeeze(1)  # [1,1,h,w]
                                q_pred = F.interpolate(q_pred.unsqueeze(1).float(), size=q_img.shape[-2:], mode='nearest').squeeze(1)  # [1,1,h,w]
                                visualize(q_img[i],svim_path,f"IMG{it}.png")

                                masker = Masker()
                                mask = masker.mask_to_rgb(q_mask[i].squeeze())
                                # plt.imshow(mask)
                                plt.imsave(svim_path+f"/LABEL.png", mask)
                                mask = masker.mask_to_rgb(q_pred[i].squeeze())
                                # plt.imshow(mask)
                                plt.imsave(svim_path+f"/{run}PRED{it}.png", mask)

            with torch.no_grad():
                base_count, novel_count, sum_base_IoU, sum_novel_IoU = 4 * [0]
                classwise_msg = ''
                for i, class_ in enumerate(val_loader.dataset.all_classes):
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
                    if class_ in val_loader.dataset.base_class_list:
                        sum_base_IoU += IoU
                        base_count += 1

                runtime = time.time() - t0
                avg_novel_IoU = sum_novel_IoU / novel_count
                avg_base_IoU = sum_base_IoU / base_count
                log('==> Run {} iteration {}, Mean novel IoU: {:.4f}, Mean base IoU: {:.4f}'.format(run+1, it, avg_novel_IoU,avg_base_IoU))
                log('\t Classwise IoU: ' + classwise_msg)

                novel_mIoU[run] = avg_novel_IoU
                base_mIoU[run] = avg_base_IoU
                runtimes[run] = runtime
                
                epi_novel_IoU[rep] = avg_novel_IoU
                epi_base_IoU[rep] = avg_base_IoU
                rep += 1
        
        # pdb.set_trace()
        # plt.figure(figsize=(10,10))
        # plt.plot(epi_loss.detach().numpy(),marker='o',label = "loss")
        # plt.ylabel('rate', fontsize=22)
        # plt.legend()
        # plt.savefig(sv_path+f"/loss{run}.png")

        plt.figure(figsize=(10,10))
        plt.plot(epi_novel_IoU.detach().numpy()[:rep],marker='o',label = "novel_IoU")
        # plt.plot(epi_base_IoU.detach().numpy(),marker='o',label = "base_IoU")
        plt.ylabel('rate', fontsize=22)
        plt.legend()
        plt.savefig(sv_path+f"/novelIoU{run}.png")

        plt.figure(figsize=(10,10))
        # plt.plot(epi_novel_IoU.detach().numpy(),marker='o',label = "novel_IoU")
        plt.plot(epi_base_IoU.detach().numpy()[:rep],marker='o',label = "base_IoU")
        plt.ylabel('rate', fontsize=22)
        plt.legend()
        plt.savefig(sv_path+f"/baseIoU{run}.png")

        for i, class_ in enumerate(val_loader.dataset.all_classes):
            log("Class {}: \t{:.8f}".format(class_, cls_IOU[i]/cls_count[i]))
    log(f'iteration: {it} ==> ')
    log('Average of novel mIoU: {:.4f} \t(over {} runs)'.format(novel_mIoU.mean(), args.n_runs))
    log('Average of base mIoU: {:.4f} \t(over {} runs)'.format(base_mIoU.mean(), args.n_runs))
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



