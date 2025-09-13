# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import torch
from torch import nn
from ..utils import cast_to

from .dpt_head import DPTHead
from .encoder import BackboneLayersSet, DinoVisionTransformerWrapper, PatchSizeAdaptationStrategy
import copy
from .cnn_extractor import BasicEncoder4



def make_head(
    embed_dims: int | list[int],
    use_batchnorm: bool = False,
    use_cls_token: bool = False,
    channels=512,
    convmodule_norm_cfg=None,
    **kwargs,
) -> torch.nn.Module:

    if isinstance(embed_dims, int):
        embed_dims = [embed_dims]

    decoder = DPTHead(
        in_channels=embed_dims,
        readout_type="project" if use_cls_token else "ignore",
        use_batchnorm=use_batchnorm,
        channels=channels,
        convmodule_norm_cfg=convmodule_norm_cfg,
        **kwargs,  # TODO add here post_process_channels, n_hidden_channels
    )

    return decoder

class Fusion(torch.nn.Module):
    def __init__(self,
                 dim1: int,
                 dim2: int,
                 out_dim: int,
                 norm_fn: str):
        super().__init__()
        

        if norm_fn == 'instance':
            # Create two independent norm layers for use in the sequence
            norm1 = nn.InstanceNorm2d(out_dim)
            norm2 = nn.InstanceNorm2d(out_dim)
        elif norm_fn == 'none':
            norm1 = nn.Identity()
            norm2 = nn.Identity()
        else:
            raise ValueError(f"Unknown norm_fn: {norm_fn}")

        self.net = nn.Sequential(
            nn.Conv2d(dim1 + dim2, out_dim, kernel_size=1, stride=1, bias=False),
            norm1,
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, stride=1, bias=False),
            norm2,
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, stride=1, bias=False)
        )

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        return self.net(x)
            
        
    

class EncoderDecoderFusion(torch.nn.Module):
    def __init__(
        self,
        encoder: torch.nn.Module,
        fdecoder: torch.nn.Module,
        idecoder: torch.nn.Module,
        fnet_cnn: torch.nn.Module,
        inet_cnn: torch.nn.Module,
        f_fusion: torch.nn.Module,
        i_fusion: torch.nn.Module,
        encoder_dtype=torch.float,
        decoder_dtype=torch.float,
    ):
        super().__init__()
        self.encoder = encoder
        self.fdecoder = fdecoder
        self.idecoder = idecoder
        self.fnet_cnn = fnet_cnn
        self.inet_cnn = inet_cnn
        self.f_fusion = f_fusion
        self.i_fusion = i_fusion
        self.encoder_dtype = encoder_dtype
        self.decoder_dtype = decoder_dtype

        self.is_cuda = torch.cuda.is_available()

        for param in self.encoder.parameters():
            param.requires_grad = False


    def forward(self, x):
        self.encoder.eval()

        prefix_shape = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])
        x = x.to(self.encoder_dtype)
        f_cnn = self.fnet_cnn(x)
        i_cnn = self.inet_cnn(x)
        with torch.no_grad():
            x = self.encoder(x)
        x = cast_to(x, self.decoder_dtype)
        
        # Get CNN features
        f_cnn = cast_to(f_cnn, self.decoder_dtype)
        i_cnn = cast_to(i_cnn, self.decoder_dtype)
        
        # Get DPT Features
        f_dpt = self.fdecoder(x)
        i_dpt = self.idecoder(x)

        # Fuse
        f_x = self.f_fusion(f_dpt, f_cnn)
        i_x = self.i_fusion(i_dpt, i_cnn)
        
        f_x = f_x.reshape(*prefix_shape, *f_x.shape[-3:])
        i_x = i_x.reshape(*prefix_shape, *i_x.shape[-3:])

        return f_x, i_x


def build_head(
    backbone: torch.nn.Module,
    f_dim: int,
    i_dim: int,
    backbone_out_layers: list[int] | BackboneLayersSet,
    use_backbone_norm: bool = False,
    use_batchnorm: bool = False,
    use_cls_token: bool = False,
    adapt_to_patch_size: PatchSizeAdaptationStrategy = PatchSizeAdaptationStrategy.CENTER_PADDING,
    encoder_dtype: torch.dtype = torch.float,
    decoder_dtype: torch.dtype = torch.float,
    # depth args
    **kwargs,
):
    encoder = DinoVisionTransformerWrapper(
        backbone_model=backbone,
        backbone_out_layers=backbone_out_layers,
        use_backbone_norm=use_backbone_norm,
        adapt_to_patch_size=adapt_to_patch_size,
    )
    encoder = encoder.to(encoder_dtype)


    fdecoder = make_head(
        encoder.embed_dims,
        use_batchnorm=use_batchnorm,
        use_cls_token=use_cls_token,
        channels=f_dim,
        convmodule_norm_cfg={"type":"instance"},
        **kwargs,
    )
    
    idecoder = make_head(
        encoder.embed_dims,
        use_batchnorm=use_batchnorm,
        use_cls_token=use_cls_token,
        channels=i_dim,
        convmodule_norm_cfg=None,
        **kwargs,
    )
    
    fnet_cnn = BasicEncoder4(output_dim=f_dim, norm_fn='instance')
    inet_cnn = BasicEncoder4(output_dim=i_dim, norm_fn='none')

    fnet_cnn = fnet_cnn.to(encoder_dtype)
    inet_cnn = inet_cnn.to(encoder_dtype)

    f_fusion = Fusion(f_dim, f_dim, f_dim,norm_fn="instance")
    i_fusion = Fusion(i_dim, i_dim, i_dim,norm_fn="none")

    encoder_decoder = EncoderDecoderFusion(
            encoder = encoder,
            fdecoder= fdecoder,
            idecoder= idecoder,
            fnet_cnn= fnet_cnn,
            inet_cnn= inet_cnn,
            f_fusion= f_fusion,
            i_fusion= i_fusion,
            encoder_dtype=encoder_dtype,
            decoder_dtype=decoder_dtype,
        )

    return encoder_decoder
