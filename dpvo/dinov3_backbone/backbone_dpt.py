# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from enum import Enum
from typing import Optional, Tuple

import torch
from .models import build_head
from urllib.parse import urlparse
from pathlib import Path

DINOV3_BASE_URL = "https://dl.fbaipublicfiles.com/dinov3"
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "dinov3"))
sys.path.append(project_root)
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



def _get_out_layers(backbone_name):
    """
    For 4 yaer
    ViT/S (12 blocks): [2, 5, 8, 11]
    ViT/B (12 blocks): [2, 5, 8, 11]
    ViT/L (24 blocks): [5, 11, 17, 23] (correct), [4, 11, 17, 23] (incorrect)
    ViT/g (40 blocks): [9, 19, 29, 39]
    """
    if "vits" in backbone_name:
        return [3, 7, 11]
    elif "vitb" in backbone_name:
        return [3, 7, 11]
    elif "vitl" in backbone_name:
        return [7, 15, 23]
    elif "vith" in backbone_name:
        return [9, 19, 31]
    elif "vit7b" in backbone_name:
        return [12, 25, 39]
    else:
        raise ValueError(f"Unrecognized backbone name {backbone_name}")


def _get_post_process_channels(backbone_name):

    if "vits" in backbone_name:
        return [384, 384, 384]
    elif "vitb" in backbone_name:
        return [384, 384, 384]
    elif "vitl" in backbone_name:
        return [512, 512, 512]
    elif "vith" in backbone_name:
        return [1280, 1280, 1280]
    elif "vit7b" in backbone_name:
        return [2048, 2048, 2048]
    else:
        raise ValueError(f"Unrecognized backbone name {backbone_name}")


_BACKBONE_DICT = {
    "dinov3_vit7b16": dinov3_vit7b16,
    "dinov3_vith16plus": dinov3_vith16plus,
    "dinov3_vitl16": dinov3_vitl16,
    
    "dinov3_vitb16": dinov3_vitb16,
    "dinov3_vits16plus": dinov3_vits16plus,
    "dinov3_vits16": dinov3_vits16,
}


def make_dinov3_head(
    *,
    backbone_name: str = "dinov3_vitb16",
    pretrained: bool = True,
    backbone_weights: BackboneWeights | str = BackboneWeights.LVD1689M,
    backbone_dtype: torch.dtype = torch.float32,
    channels=512,
    use_backbone_norm=True,
    use_batchnorm=False,
    use_cls_token=False,
    **kwargs,
):
    
    print(f"backbone_weights : {backbone_weights}")
    backbone: torch.nn.Module = _BACKBONE_DICT[backbone_name](
        pretrained=pretrained,
        weights=backbone_weights,
    )
    out_index = _get_out_layers(backbone_name)
    post_process_channels = _get_post_process_channels(backbone_name)

    depther = build_head(
        backbone,
        backbone_out_layers=out_index,
        use_backbone_norm=use_backbone_norm,
        use_batchnorm=use_batchnorm,
        use_cls_token=use_cls_token,
        head_type="dpt",
        encoder_dtype=backbone_dtype,
        # DPTHead args
        channels=channels,
        post_process_channels=post_process_channels,
        **kwargs,
    )

    # if pretrained:
    #     if isinstance(depther_weights, DepthWeights):
    #         assert depther_weights == DepthWeights.SYNTHMIX, f"Unsupported depther weights {depther_weights}"
    #         weights_name = depther_weights.value.lower()
    #         hash = kwargs["hash"] if "hash" in kwargs else "02040be1"
    #         url = DINOV3_BASE_URL + f"/{backbone_name}/{backbone_name}_{weights_name}_dpt_head-{hash}.pth"
    #     else:
    #         url = convert_path_or_url_to_url(depther_weights)
    #     checkpoint = torch.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=check_hash)
    #     depther[0].decoder.load_state_dict(checkpoint, strict=True)
    return depther




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from PIL import Image
    from torchvision import transforms
    def get_img():
        
        path = "/home/ogam1080ti/Desktop/Onder/git2/DPVO/img.jpg"
        image = Image.open(path).convert("RGB")
        return image

    def make_transform(resize_size: int | list[int] = 768):

        to_tensor = transforms.ToTensor()
        resize = transforms.Resize((resize_size, resize_size//2), antialias=True)
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        return transforms.Compose([to_tensor, resize, normalize])
    model = make_dinov3_head(backbone_name="dinov3_vitb16",
                             channels= 718,
                             pretrained=True,
                             backbone_weights="/home/ogam1080ti/Desktop/Onder/git2/DPVO/dinov3/weights/dinov3_vitb16.pth")

    #model.cuda()
    img_size = 640
    img = get_img()
    transform = make_transform(img_size)
    with torch.inference_mode():
        with torch.autocast('cpu', dtype=torch.float32):
            batch_img = transform(img)[None, None]
            batch_img = batch_img.cpu()
            embed = model(batch_img)
            pass
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(img)
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(embed[0,0].mean(0).cpu(), cmap=colormaps["Spectral"])
    plt.axis("off")
    plt.show()
    pass


