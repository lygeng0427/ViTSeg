# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class EncoderDecoder(nn.Module):
    """Encoder Decoder segmentors.
    """

    def __init__(self, backbone, decoder, args=None, neck=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.neck = neck
        self.args = args

    def extract_features(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        if self.args.arch == 'dino':
            x = self.backbone.get_intermediate_layers(inputs, n=1)
            x = x[-1]
            x = x[:, 1:]  # drop cls_token
        elif self.args.arch == 'dinov2':
            x = self.backbone.get_intermediate_layers(inputs, n=1, reshape=False, norm=True, return_class_token=False)
            x = x[-1]  # cls_token already dropped

        if self.neck is not None:
            x = self.neck(x)
        return x

    def forward(self, inputs: Tensor) -> Tensor:
        """Network forward process.
        """
        H, W = inputs.size(2), inputs.size(3)

        if self.args.arch == 'dino':
            x = self.backbone.get_intermediate_layers(inputs, n=1)
            x = x[-1]
            x = x[:, 1:]  # drop cls_token
        elif self.args.arch == 'dinov2':
            x = self.backbone.get_intermediate_layers(inputs, n=1, reshape=False, norm=True, return_class_token=False)
            x = x[-1]  # cls_token already dropped

        if self.neck is not None:
            x = self.neck(x)

        masks = self.decoder(x, (H, W))
        # masks = F.interpolate(masks, size=(H, W), mode="bilinear")

        return masks