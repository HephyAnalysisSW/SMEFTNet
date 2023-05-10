import warnings
warnings.filterwarnings("ignore")

import os
import math
import numpy as np
import torch
import pickle
import glob
import itertools
import operator
import functools

from models.EFTModel import EFTModel

import SMEFTNet
#import sys
#sys.path.insert(0, '..')
#sys.path.insert(0, '../..')
#sys.path.insert(0, '../../..')
import tools.user as user

conv_params     = ( [(0.0, [20, 20])] )
readout_params  = (0.0, [32, 32])
dRN = 0.4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_model = EFTModel(events_per_parampoint=1)

from SMEFTNet import SMEFTNet
model = SMEFTNet(
    dRN=dRN,
    conv_params=conv_params,
    readout_params=readout_params,
    learn_from_gamma=True,
    num_classes = 2,
    regression=True,
    ).to(device)

base_points = [ 1, 2]
def loss( out, truth, weights):
    return sum([(weights[:,0]*( theta*(out[:,0] - weights[:,1]/weights[:,0]) + .5*theta**2*( out[:,1] - weights[:,2]/weights[:,0] ))**2).sum() for theta in base_points])
