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
from evaluate_euroc import run as runEuRoC
from evaluate_euroc import write_trajectory
from dpvo.config import cfg

import evo.main_ape as main_ape
from evo.core import sync
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

import glob

def load_weights(network):
    
    if isinstance(network, str):
        from collections import OrderedDict
        state_dict = torch.load(network)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if "update.lmbda" not in k:
                new_state_dict[k.replace('module.', '')] = v
        
        network = VONet(args)
        network.load_state_dict(new_state_dict)

    else:
        network = network

    # steal network attributes
    network.cuda()
    network.eval()

    return network


def validateEuRoC(args, cfg, net):

    euroc_scenes = [
        "MH_01_easy",
        "MH_02_easy",
        "MH_03_medium",
        "MH_04_difficult",
        "MH_05_difficult",
        "V1_01_easy",
        "V1_02_medium",
        "V1_03_difficult",
        "V2_01_easy",
        "V2_02_medium",
        "V2_03_difficult",
    ]

    for trial_id in range(args.trials):
        for scene in euroc_scenes:

            # Get tge estimated path
            path_to_data = os.path.join(args.eurocdir, scene, "mav0", "cam0", "data")
            traj_est_raw, _ = runEuRoC(cfg=cfg, network=net, imagedir=path_to_data, calib="calib/euroc.txt", stride=args.stride, viz=False, show_img=False)

            # Get the timestamps
            images_list = sorted(glob.glob(os.path.join(path_to_data, "*.png")))[::args.stride]
            tstamps = [float(x.split('/')[-1][:-4]) for x in images_list]

            traj_est = PoseTrajectory3D(
                positions_xyz=traj_est_raw[:,:3],
                orientations_quat_wxyz=traj_est_raw[:, [6, 3, 4, 5]],
                timestamps=tstamps)


            # Get the groundtruth
            groundtruth = "datasets/euroc_groundtruth/{}.txt".format(scene) 
            traj_ref = file_interface.read_tum_trajectory_file(groundtruth)
            
            # Sync estimated trajectory with groundtruth
            traj_ref_sync, traj_est_sync = sync.associate_trajectories(traj_ref, traj_est)

            result = main_ape.ape(traj_ref_sync, traj_est_sync, est_name='traj', 
                    pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
            

            # Finally print and save the result
            ate_score = result.stats["rmse"]
            network_config = args.cp.split("/")[-1].split(".")[0]
            print(f"{network_config} in {scene} : {ate_score}")
            write_trajectory(path=os.path.join("EstimatedTrajectories",network_config), seq=scene, traj=traj_est_raw, times=tstamps, trial_id=trial_id)


if __name__=="__main__":

    

    model_list = [
        "DVOl16",
        "DVOb16",
        "DVOs16plus",
    ]

    # model_list = [
    #     "DVOs16plus",
    # ]

    dino_weight_dict = {
        model_list[0] : "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd",
        model_list[1] : "dinov3_vitb16",
        model_list[2] : "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa",
    }

    dino_model_dict = {
        model_list[0] : "dinov3_vitl16",
        model_list[1] : "dinov3_vitb16",
        model_list[2] : "dinov3_vits16plus",
    }


    iteration_list = [
        "040000",
        "050000",
        "070000",
    ]

    for iteration in iteration_list:
        for model in model_list:
            checkpoint_dpvo = model + "_" + iteration + ".pth"
            parser = argparse.ArgumentParser()
            parser.add_argument('--cp', default=f"/home/haktanito/icra2026/checkpoints/{checkpoint_dpvo}")
            parser.add_argument('--P', default=3)
            parser.add_argument('--R', default=3)
            parser.add_argument('--DINO_MODEL', default=dino_model_dict[model])
            parser.add_argument('--PATH_DINO_WEIGHTS', default=f'dinov3/weights/{dino_weight_dict[model]}.pth')

            parser.add_argument('--dataset', default='EuRoC')


            parser.add_argument('--config', default="config/default.yaml")
            parser.add_argument('--stride', type=int, default=2)
            parser.add_argument('--trials', type=int, default=1)
            parser.add_argument('--eurocdir', default="/media/haktanito/HDD/EuroC")
            parser.add_argument('--backend_thresh', type=float, default=64.0)
            parser.add_argument('--plot', action="store_true")
            parser.add_argument('--opts', nargs='+', default=[])
            parser.add_argument('--save_trajectory', action="store_true")
            args = parser.parse_args()

            cfg.merge_from_file(args.config)
            cfg.BACKEND_THRESH = args.backend_thresh
            cfg.merge_from_list(args.opts)
            
            cfg.P = args.P
            cfg.R = args.R
            cfg.DINO_MODEL = args.DINO_MODEL
            cfg.PATH_DINO_WEIGHTS = args.PATH_DINO_WEIGHTS
            

            args = parser.parse_args()

            net = load_weights(args.cp)
            validateEuRoC(args, cfg, net)