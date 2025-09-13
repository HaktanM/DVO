# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from enum import Enum
from typing import Optional, Tuple

import torch
from urllib.parse import urlparse
from pathlib import Path

import sys
import os



from torch import Tensor, nn

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "dinov3"))
print(project_root)
sys.path.append(project_root)

from .models.encoder import DinoVisionTransformerWrapper, PatchSizeAdaptationStrategy

from dinov3.hub.backbones import (
    Weights as BackboneWeights,
    dinov3_vit7b16,
    dinov3_vitb16,
    dinov3_vith16plus,
    dinov3_vitl16,
    dinov3_vits16,
    dinov3_vits16plus,
    convert_path_or_url_to_url,
)


_BACKBONE_DICT = {
    "dinov3_vit7b16": dinov3_vit7b16,
    "dinov3_vith16plus": dinov3_vith16plus,
    "dinov3_vitl16": dinov3_vitl16,
    
    "dinov3_vitb16": dinov3_vitb16,
    "dinov3_vits16plus": dinov3_vits16plus,
    "dinov3_vits16": dinov3_vits16,
}


model_list = [
    "dinov3_vitl16",
    "dinov3_vitb16",
    "dinov3_vits16plus"
]

dino_weight_dict = {
    model_list[0] : "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd",
    model_list[1] : "dinov3_vitb16",
    model_list[2] : "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa",
}


class Config:
    DINO_MODEL          = None
    PATH_DINO_WEIGHTS   = None

    @property
    def PATH_DINO_WEIGHTS(self):
        return f'dinov3/weights/{dino_weight_dict[self.DINO_MODEL]}.pth'



class DVOHead(torch.nn.Module):
    def __init__(
        self,
        encoder: torch.nn.Module,
        encoder_dtype=torch.float,
        number_of_layers = None
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_dtype = encoder_dtype

        self.fnet = torch.nn.Conv2d(in_channels=number_of_layers*384, out_channels=128, kernel_size=1)
        self.inet = torch.nn.Conv2d(in_channels=number_of_layers*384, out_channels=384, kernel_size=1)

        self.is_cuda = torch.cuda.is_available()

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        # b : batch size
        # n : number of images
        # c1 : channel size of the input
        b, n, c1, h1, w1 = x.shape
        x = x.view(b*n, c1, h1, w1)

        self.encoder.eval()

        x = x.reshape(-1, *x.shape[-3:])
        x = x.to(self.encoder_dtype)
        with torch.no_grad():
            x = self.encoder(x)

        # Concatenate the feature maps and apply a 1D convolution
        feature_maps = []
        for layer in x:
            # layer[0] holds the feature map of the image  ( 384 x (H/4) x (W/4) )
            # layer[1] holds the vector representation of the whole image (381x1)
            feature_maps.append(layer[0]) 
        concatenated_fmap = torch.cat(feature_maps, dim=1)

        fmap = self.fnet(concatenated_fmap)
        imap = self.inet(concatenated_fmap)

        _, cf, hf, wf = fmap.shape
        _, ci, hi, wi = imap.shape
        
        return fmap.view(b, n, cf, hf, wf), imap.view(b, n, ci, hi, wi)


def getDinoHead(
        cfg
    ):

    # Get the DinoV3 model 
    backbone: torch.nn.Module = _BACKBONE_DICT[cfg.DINO_MODEL](
        pretrained=True,
        weights=cfg.PATH_DINO_WEIGHTS,
    )

    # Create the encoder, get the desired layers as outpus
    encoder = DinoVisionTransformerWrapper(
        backbone_model=backbone,
        backbone_out_layers=cfg.ENCODER_LAYERS,
        use_backbone_norm=True,
        adapt_to_patch_size=PatchSizeAdaptationStrategy.CENTER_PADDING,
    )

    # Finally, return our DVO head
    # DVO head convers Dino vectors into fmap and gmap
    return DVOHead(encoder=encoder, number_of_layers=len(cfg.ENCODER_LAYERS))


    