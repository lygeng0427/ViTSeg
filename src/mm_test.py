from src.mmpretrain.mmpretrain.apis.model import get_model
from src.dino_vision_transformer import vit_base
import torch
import pdb

inputs = torch.rand(1,3,473,473)
backbone_name = 'ViT'

if backbone_name == 'Dino':
    backbone = vit_base(16)
    dino_vitb16_param = torch.load('src/dino_vitbase16_pretrain.pth')
    # Todo 1,768,785 ---> 1,768,28,28
    backbone.load_state_dict(dino_vitb16_param)
else:
    from src.mmpretrain.mmpretrain.apis.model import get_model
    if backbone_name == 'DeiT':
        backbone = get_model('deit-base_16xb64_in1k', pretrained=True) #torch.Size([1, 768, 28, 28])
    elif backbone_name == 'ViT':
        backbone = get_model('vit-base-p16_32xb128-mae_in1k', pretrained=True) #output size torch.Size([1, 768, 28, 28])
    elif backbone_name == 'Dino2':
        backbone = get_model('vit-base-p14_dinov2-pre_3rdparty', pretrained=True) #torch.Size([1, 768, 32, 32])

pdb.set_trace()
print(feats.shape)
print(len(feats))
last_out = feats[-1]
print(last_out.shape)