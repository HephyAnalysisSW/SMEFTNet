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
import ROOT 
import tools.helpers as helpers
ROOT.gROOT.SetBatch(True)

from models.WZModel import WZModel
import matplotlib.pyplot as plt 

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

scalar_features = ["genJet_pt",'parton_lepV_pt']

data_model_train = WZModel(what='VV_eflow', scalar_features = scalar_features, operator='cWtil', nsplit=500, train=True  , linear_term_range=[-2000,2000])
data_model_test  = WZModel(what='VV_eflow', scalar_features = scalar_features, operator='cWtil', nsplit=500, train=False , linear_term_range=[-2000,2000])


from SMEFTNet import SMEFTNet
model = SMEFTNet(
    dRN=dRN,
    conv_params=conv_params,
    readout_params=readout_params,
    learn_from_gamma=True,
    num_scalar_features=len(scalar_features),
    num_classes = 1,
    regression=True,
   ).to(device)


def loss( out, truth, weights=None):
    weight_sm = weights[:,0]
    target    = weights[:,1]
    return torch.mean( weight_sm*( out[:,0] - target )**2)

truth_var_names = ['phi', 'hadV_pt', 'lepV_pt']
ranges = [(-math.pi, math.pi, 50), (300, 1000,15), (300,1000,15)]
def plot_chunk( out, truth, weights,  scalar_features=None, chunk_number=0):
    weight_sm = weights[:,0]
    target    = weights[:,1]

    ret={}
    for var, (varname, (xlow,xhigh, bins)) in enumerate(zip(truth_var_names,ranges)):
        score = ROOT.TH1F("score_%s_%d"%(varname,chunk_number) , "score", bins, xlow, xhigh)
        pred  = ROOT.TH1F("pred_%s_%d" %(varname,chunk_number) , "pred" , bins, xlow, xhigh)
        norm  = ROOT.TH1F("norm_%s_%d" %(varname,chunk_number) , "norm" , bins, xlow, xhigh)

        
        score.Add( helpers.make_TH1F( np.histogram(truth[:,var], np.linspace(xlow, xhigh, bins+1), weights=weight_sm*target) ))
        pred .Add( helpers.make_TH1F( np.histogram(truth[:,var], np.linspace(xlow, xhigh, bins+1), weights=weight_sm*out[:,0]) ))
        norm .Add( helpers.make_TH1F( np.histogram(truth[:,var], np.linspace(xlow, xhigh, bins+1), weights=weight_sm)) )
        ret[varname]=[score, pred, norm]
    return ret

def add_chunks( chunks ):
    ret={}
    for var in chunks[0]:
        ret[var]=chunks[0][var]
        for i in range(1,len(chunks)):
            for j in range(3):
                ret[var][j].Add(chunks[i][var][j])
    return ret

def save( objects, model_directory, epoch ):

    outf=ROOT.TFile.Open(os.path.join( model_directory, f'test_closure_{epoch}.root'), "recreate")
    for varname in objects:
        score,pred,norm=objects[varname]
    
        score.Divide(norm)
        pred.Divide(norm)

        c1 = ROOT.TCanvas()
        score.SetLineStyle(2)
        score.GetXaxis().SetTitle(varname)
        score.Draw()
        pred.Draw("same")
        c1.Print( '%s/closure_%s_%03i.png'%(model_directory,varname,epoch))
        
        outf.WriteTObject( score, "true_%s"%varname)
        outf.WriteTObject( pred , "surrogate_%s"%varname)

    outf.Close()
