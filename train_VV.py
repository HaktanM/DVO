import cv2
import os
import argparse
import numpy as np
from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dpvo.data_readers.factory import dataset_factory

from dpvo.lietorch import SE3
from dpvo.logger import Logger
import torch.nn.functional as F

from dpvo.net import VONet
from evaluate_tartan import evaluate as validate


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()

def image2gray(image):
    image = image.mean(dim=0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()

def kabsch_umeyama(A, B):
    n, m = A.shape
    EA = torch.mean(A, axis=0)
    EB = torch.mean(B, axis=0)
    VarA = torch.mean((A - EA).norm(dim=1)**2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = torch.svd(H)

    c = VarA / torch.trace(torch.diag(D))
    return c


def train(args):
    """ main training loop """

    # legacy ddp code
    rank = 0

    db = dataset_factory(['tartan'], datapath=args.h5_path, n_frames=args.n_frames)
    train_loader = DataLoader(db, batch_size=1, shuffle=True, num_workers=4)

    net = VONet(args)
    net.train()
    net.cuda()

    if args.ckpt is not None:
        state_dict = torch.load(args.ckpt)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v
        net.load_state_dict(new_state_dict, strict=False)

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-6)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
        args.lr, args.steps, pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    if rank == 0:
        logger = Logger(args.name, scheduler)

    total_steps = 0

    while 1:
        for data_blob in train_loader:
            images, poses, disps, intrinsics = [x.cuda().float() for x in data_blob]
            optimizer.zero_grad()

            # fix poses to gt for first 1k steps
            so = False


            poses = SE3(poses).inv()
            if not args.test:
                traj, stats, logging, _ = net(images, poses, disps, intrinsics, M=1024, STEPS=STEPS, structure_only=so, wtd_loss=args.all_flows_loss, total_steps=total_steps)
            else:
                with torch.no_grad():
                    traj, stats, logging, _ = net(images, poses, disps, intrinsics, M=1024, STEPS=STEPS, structure_only=so, wtd_loss=args.all_flows_loss, total_steps=total_steps)

            tr_list = []
            ro_list = []
            ef_list = []
            loss = 0.0
            pose_loss = 0.0
            flow_loss = 0.0
            ro_loss = 0.0
            tr_loss = 0.0
            for i, (v, x, y, P1, P2, _, wtk, _, _, vf, _, xf, yf, _) in enumerate(traj):
                wtk_d = wtk[:, :, None, None, :].detach()
                e = ((x-y)*wtk_d).norm(dim=-1)
                e = e.reshape(-1, net.P**2)[(v > 0.5).reshape(-1)].min(dim=-1).values


                ef = (xf - yf).norm(dim=-1)
                ef = ef.reshape(-1, net.P**2)[(vf > 0.5).reshape(-1)].min(dim=-1).values

                N = P1.shape[1]
                ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))
                ii = ii.reshape(-1).cuda()
                jj = jj.reshape(-1).cuda()

                k = ii != jj
                ii = ii[k]
                jj = jj[k]

                P1 = P1.inv()
                P2 = P2.inv()

                t1 = P1.matrix()[...,:3,3]
                t2 = P2.matrix()[...,:3,3]

                s = kabsch_umeyama(t2[0], t1[0]).detach().clamp(max=10.0)
                P1 = P1.scale(s.view(1, 1))

                dP = P1[:,ii].inv() * P1[:,jj]
                dG = P2[:,ii].inv() * P2[:,jj]

                e1 = (dP * dG.inv()).log()
                tr = e1[...,0:3].norm(dim=-1)
                ro = e1[...,3:6].norm(dim=-1)

                tr_list.append(tr)
                ro_list.append(ro)
                ef_list.append(ef)
                
                flow_loss += args.flow_weight * e.mean()
                if not so and (i >= 2 or args.all_poses_loss):
                    ro_l = ro.mean()
                    tr_l = tr.mean()
                    pose_loss += args.pose_weight * ( tr_l + ro_coeff*ro_l )
                    ro_loss += ro_l
                    tr_loss += tr_l

            if total_steps > 0 and total_steps % 20 == 0 and not args.test and not so:
                flow_grad = max(torch.autograd.grad(flow_loss, net.update.gru[1].res[0].weight, retain_graph=True)[0].norm().item(), 1e-6)
                pose_grad = max(torch.autograd.grad(pose_loss, net.update.gru[1].res[0].weight, retain_graph=True)[0].norm().item(), 1e-6)
                tr_grad = max(torch.autograd.grad(tr_loss, net.update.gru[1].res[0].weight, retain_graph=True)[0].norm().item(), 1e-6)
                ro_grad = max(torch.autograd.grad(ro_loss, net.update.gru[1].res[0].weight, retain_graph=True)[0].norm().item(), 1e-6)
                
                ro_coeff_ratios = ro_coeff_ratios[-50:] + [max(min(tr_grad/ro_grad, 10*ro_coeff), 0.1*ro_coeff)]
                ro_coeff = np.mean(ro_coeff_ratios)
                flow_coeff_ratios = flow_coeff_ratios[-50:] + [max(min(pose_grad/flow_grad, 10*flow_coeff), 0.1*flow_coeff)]
                flow_coeff = np.mean(flow_coeff_ratios)

            loss = flow_coeff*flow_loss + pose_loss
            
       
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            optimizer.step()
            scheduler.step()

            total_steps += 1

            metrics = {
                "loss": loss.item(),
                "px1": (e < .25).float().mean().item(),
                "ro": ro.float().mean().item(),
                "tr": tr.float().mean().item(),
                "r1": (ro < .001).float().mean().item(),
                "r2": (ro < .01).float().mean().item(),
                "t1": (tr < .001).float().mean().item(),
                "t2": (tr < .01).float().mean().item(),
                "coeffs/ro": ro_coeff,
                "coeffs/flow": flow_coeff
            }

            if rank == 0:
                logger.push(metrics)

            if total_steps % 10000 == 0:
                torch.cuda.empty_cache()

                if rank == 0:
                    PATH = 'checkpoints/%s_%06d.pth' % (args.name, total_steps)
                    torch.save(net.state_dict(), PATH)

                # Validation will be caried out on EuroC sequence
                # validation_results = validate(None, net)
                # if rank == 0:
                #     logger.write_dict(validation_results)

                torch.cuda.empty_cache()
                net.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',    default='/DVO_baseline', help='name your experiment')
    parser.add_argument('--ckpt',    help='checkpoint to restore')
    parser.add_argument('--h5_path', default="/data", help='path to h5 files')
    parser.add_argument('--steps', type=int, default=240000)
    parser.add_argument('--lr', type=float, default=0.00008)
    parser.add_argument('--clip', type=float, default=10.0)
    parser.add_argument('--n_frames', type=int, default=15)
    parser.add_argument('--pose_weight', type=float, default=10.0)
    parser.add_argument('--flow_weight', type=float, default=0.1)

    ## Patch size and radius of correlation window
    parser.add_argument('--P', type=int, default=3)
    parser.add_argument('--R', type=int, default=3)

    ## Dino related arguments
    parser.add_argument('--DINE_MODEL', type=str, default="dinov3_vits16plus")
    parser.add_argument('--PATH_DINO_WEIGHTS', type=str, default=f"dinov3/weights/dinov3_vits16plus.pth")
    parser.add_argument('--ENCODER_LAYERS', type=list, default=[0, 3, 5])
    args = parser.parse_args()

    train(args)
