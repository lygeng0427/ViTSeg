import torch
import math
import timm
from torch import nn
from src.model.pspnet import PSPNet
from src.model.decoder import DecoderLinear, MaskTransformer
from src.model.encoder_decoder import EncoderDecoder
from mmengine.runner.checkpoint import load_state_dict


def get_model(args) -> nn.Module:
    if args.arch == 'resnet':
        return PSPNet(args, zoom_factor=8, use_ppm=True)
    else:
        if args.arch == 'dino':
            backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
            patch_size = 16
        elif args.arch == 'dinov2':
            backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            patch_size = 14

        decoder = create_decoder(args, d_model=768, patch_size=patch_size, num_classes=args.num_classes_tr)
        segmentor = EncoderDecoder(backbone, decoder, args=args, neck=None)
        return segmentor


def create_decoder(args, d_model=768, patch_size=14, num_classes=None):
    decoder_cfg = args.decoder_cfg.copy()

    decoder_cfg["d_encoder"] = d_model
    decoder_cfg["patch_size"] = patch_size
    if 'n_cls' not in decoder_cfg:
        decoder_cfg['n_cls'] = num_classes if num_classes is not None else args.num_classes_tr

    if args.decoder == 'linear':
        decoder_cfg = {k: v for k, v in decoder_cfg.items() if k in ['n_cls', 'patch_size', 'd_encoder']}
        decoder = DecoderLinear(**decoder_cfg)
    elif args.decoder == "mask_transformer":
        dim = d_model
        n_heads = dim // 64
        decoder_cfg["n_heads"] = n_heads
        decoder_cfg["d_model"] = dim
        decoder_cfg["d_ff"] = 4 * dim
        decoder = MaskTransformer(**decoder_cfg)
    else:
        raise ValueError(f"Unknown decoder: {args.decoder}")
    return decoder


"""
if args.arch == 'vit':
    cfg_path = 'configs/vit/vit_vit-b16_mln_upernet_8xb2-80k_ade20k-512x512.py'
    cfg = Config.fromfile(cfg_path)
    cfg.model.pop('data_preprocessor')
    cfg.model['pretrained'] = 'initmodel/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth'
    model = MODELS.build(cfg.model)
    model = init_model(cfg, device='cpu')
    
    
    cfg_backbone = cfg.model['backbone']
    cfg_backbone['pretrained'] = 'initmodel/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth'
    backbone = MODELS.build(cfg_backbone)

"""


def init_backbone_weight(model, ckpt_path):
    checkpoint = torch.load(ckpt_path)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    if 'backbone.pos_embed' in state_dict.keys():
        if model.backbone.pos_embed.shape != state_dict['backbone.pos_embed'].shape:
            print(f'Resize the pos_embed shape from 'f'{state_dict["backbone.pos_embed"].shape} to 'f'{model.backbone.pos_embed.shape}')
            h, w = model.backbone.img_size
            pos_size = int(math.sqrt(state_dict['backbone.pos_embed'].shape[1] - 1))
            state_dict['backbone.pos_embed'] = model.backbone.resize_pos_embed(
                state_dict['backbone.pos_embed'],
                (h // model.backbone.patch_size, w // model.backbone.patch_size),
                (pos_size, pos_size), model.backbone.interpolate_mode)

    load_state_dict(model, state_dict, strict=False, logger=None)




