# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from torchview import draw_graph
import torch
from dpvo.dinov3_backbone.DinoHead import getDinoHead

model_list = [
    "dinov3_vitl16",
    "dinov3_vitb16",
    "dinov3_vits16plus"
]

dino_weight_dict = {
    model_list[0] : "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd",
    model_list[1] : "dinov3_vitb16",
    model_list[2] : "dinov3_vits16plus",
}


class Config:
    DINO_MODEL          = None
    ENCODER_LAYERS      = [0, 3, 5]

    @property
    def PATH_DINO_WEIGHTS(self):
        return f'dinov3/weights/{dino_weight_dict[self.DINO_MODEL]}.pth'


if __name__ == "__main__":
    model = model_list[2]

    cfg = Config()
    cfg.DINO_MODEL = model



    
    dvo_head = getDinoHead(cfg)
    
    # x = torch.rand(1,1,3,300,300)
    # fmap, imap = dvo_head.forward(x)

    # # --- Check which parameters are trainable ---
    # print("Trainable parameters in the model:")
    # trainable_params_count = 0
    # for name, param in dvo_head.named_parameters():
    #     print(f"Name: {name}, requires_grad: {param.requires_grad}, shape: {param.shape}")
    #     if param.requires_grad:
    #         trainable_params_count += param.numel()

    # print(f"\nTotal number of trainable parameters: {trainable_params_count}")

    # Assume 'backbone' is your instantiated DINOv3 model
    model_graph = draw_graph(
        dvo_head, 
        input_size=(1, 1, 3, 224, 224), 
        expand_nested=True # This shows the internal layers of blocks
    )

    # To view the graph
    model_graph.visual_graph

    # To save the graph as a file (e.g., a PNG)
    model_graph.visual_graph.render("dinov3_backbone", format="svg")