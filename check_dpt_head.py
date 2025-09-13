from dpvo.dinov3_backbone import make_dinov3_head
import torch

if __name__=="__main__":
    dino_head = make_dinov3_head(   backbone_name="dinov3_vitb16",
                                    f_dim=128,
                                    i_dim=384,
                                    pretrained=True,
                                    backbone_weights="dinov3/weights/dinov3_vitb16.pth")
    

    img = torch.rand(1,1,3,256,256)
    out = dino_head(img)

    pass