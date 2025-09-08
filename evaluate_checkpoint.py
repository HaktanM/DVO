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


def load_weights(network):
    
    if isinstance(network, str):
        from collections import OrderedDict
        state_dict = torch.load(network)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if "update.lmbda" not in k:
                new_state_dict[k.replace('module.', '')] = v
        
        network = VONet()
        network.load_state_dict(new_state_dict)

    else:
        network = network

    # steal network attributes
    network.cuda()
    network.eval()

    return network


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cp')
    args = parser.parse_args()

    net = load_weights(args.cp)
    validation_results = validate(None, net)

    for key in validation_results:
        print(f"{key} : {validation_results[key]}")