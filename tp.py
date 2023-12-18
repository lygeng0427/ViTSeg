

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

from src.classifier import Classifier
from src.model.pspnet import get_model
from src.dataset.dataset import get_val_loader
from src.util import (
    setup,
    cleanup,
    get_cfg,
    setup_seed,
    get_model_dir,
    find_free_port,
    resume_random_state,
    merge_cfg_from_list,
    load_cfg_from_cfg_file,
    fast_intersection_and_union,
)

# ======= load config =======
parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--config', type=str, required=True, help='config file')
parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
args = parser.parse_args('--config configs/pascal.yaml'.split())
assert args.config is not None
cfg = load_cfg_from_cfg_file(args.config)
if args.opts is not None:
    cfg = merge_cfg_from_list(cfg, args.opts)
args=cfg

# =======change config ==========

args.image_size = 448
args.batch_size_val = 10

# ========= model ===========
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

# ========== Data  ==========
val_loader = get_val_loader(args)

# ========== Eval ============
model.eval()
nb_episodes = len(val_loader) if args.test_num == -1 else int(args.test_num / args.batch_size_val)

if not args.generate_new_support_set_for_each_task:
    with torch.no_grad():
        s_imgs, s_label = val_loader.dataset.generate_support([], remove_them_from_query_data_list=True)
        nb_episodes = len(val_loader) if args.test_num == -1 else nb_episodes  # Updates nb_episodes since some images were removed by generate_support
        s_out = model.forward_features(s_imgs)
        s_feat = s_out['x_norm_patchtokens']
        s_feat = s_feat.view(args.num_classes_val * args.shot, args.image_size // args.patch_size,
                             args.image_size // args.patch_size, -1)

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
        features_q = model.module.extract_features(qry_img).detach().unsqueeze(1)
        valid_pixels_q = q_valid_pix.unsqueeze(1).to(device)
        gt_q = q_label.unsqueeze(1)

        query_image_path_list = list(img_path)
        if args.generate_new_support_set_for_each_task:
            spprt_imgs, s_label = val_loader.dataset.generate_support(query_image_path_list)
            spprt_imgs = spprt_imgs.to(device, non_blocking=True)
            s_label = s_label.to(device, non_blocking=True)
            features_s = model.module.extract_features(spprt_imgs).detach().view(
                (args.num_classes_val, args.shot, c, h, w))
            gt_s = s_label.view((args.num_classes_val, args.shot, args.image_size, args.image_size))
