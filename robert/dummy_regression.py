import torch
import torch.nn as nn
import uproot
from torch.utils.data import DataLoader
#from dataloader import RootDataset 
from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt 
import numpy as np
import ROOT
import math
import functools
import operator

from glob import glob
from torch.utils.data import Dataset

import sys, os
sys.path.insert(0, '..')
import tools.syncer as syncer
import tools.user as user
import tools.helpers as helpers

prefix = 'dummy_v1'
epochs = 300

torch.set_num_threads(8)

branches  = ['parton_hadV_angle_phi', 'p_C', 'delphesJet_dR_hadV_maxq1q2', 'delphesJet_dR_matched_hadV_parton']
selection = lambda ar: (ar['delphesJet_dR_hadV_maxq1q2']<0.6) & (ar['delphesJet_dR_matched_hadV_parton']<0.6)

from tools.WeightInfo import WeightInfo
reweight_pkl = '/eos/vbc/group/cms/robert.schoefbeck/gridpacks/ParticleNet/WZto2L_HT300_reweight_card.pkl'
weightInfo = WeightInfo(reweight_pkl)
weightInfo.set_order(2)
default_eft_parameters = {p:0 for p in weightInfo.variables}
def make_eft(**kwargs):
    result = { key:val for key, val in default_eft_parameters.items() }
    for key, val in kwargs.items():
        if not key in weightInfo.variables+["Lambda"]:
            raise RuntimeError ("Wilson coefficient not known.")
        else:
            result[key] = float(val)
    return result

def getWeights( eft, coeffs, lin=False):

    if lin:
        combs = list(filter( lambda c:len(c)<2, weightInfo.combinations))
    else:
        combs = weightInfo.combinations
    fac = np.array( [ functools.reduce( operator.mul, [ (float(eft[v]) - weightInfo.ref_point_coordinates[v]) for v in comb ], 1 ) for comb in combs], dtype='float')
    return np.matmul(coeffs[:,:len(combs)], fac)

class RootDataset(Dataset):
    def __init__(self, files, branches):
        self.tree_name='Events'
        self.input_files = [ f for f in glob(files)]
        self.branches = branches
        array = uproot.concatenate([f+':'+self.tree_name for f in self.input_files], self.branches, library='np')
        mask_selection = selection(array)
        self.dphi = array['parton_hadV_angle_phi'][mask_selection]
        coeffs = np.stack(array['p_C'][mask_selection],axis=0)

        self.weight= coeffs[:,0]
        self.truth = 0.5*( getWeights( make_eft( cW=1 ), coeffs) - getWeights( make_eft( cW=-1 ), coeffs))/self.weight

        clip_mask = (np.abs(self.truth) < 1e3)
        self.dphi = self.dphi[clip_mask]
        self.truth = self.truth[clip_mask]

    def __len__(self):
        return self.truth.shape[0]

    def __getitem__( self, idx ):
        return self.dphi[idx], 10**4*self.weight[idx], self.truth[idx]

model = nn.Sequential(
    nn.Linear(1,5),
    nn.ReLU(),
    nn.Linear(5,5),
    nn.ReLU(),
    nn.Linear(5,1),
)

dataset=RootDataset("/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v6/WZto2L_HT300_Ref/*.root", branches = branches)
train,test = torch.utils.data.random_split( dataset, [0.8,0.2], generator=torch.Generator().manual_seed(42))
train_loader      = DataLoader(train, batch_size=len(train))
test_loader       = DataLoader(test,  batch_size=len(test))
train_loader_eval = DataLoader(train, batch_size=len(train))

#optimizer = optim.RMSprop(model.parameters(), lr=0.005, momentum=0)
optimizer = optim.Adam(model.parameters(), lr=0.005)

#def loss( estimate, weight, truth):
#    if not estimate.ndim==weight.ndim==truth.ndim:
#        raise RuntimeError("Unintentional broadcasting! %i %i %i "%( estimate.ndim, weight.ndim, truth.ndim) )
#    return torch.mean(weight*(estimate-truth)**2)
def loss( estimate, weight, truth):
    if not estimate.ndim==weight.ndim==truth.ndim:
        raise RuntimeError("Unintentional broadcasting! %i %i %i "%( estimate.ndim, weight.ndim, truth.ndim) )
    return torch.mean(weight*torch.abs(estimate-truth))

loss_train=[]
loss_test=[]
# nsteps=100
# for phi, truth in train_loader_eval:
#     mean = torch.mean(truth).item()
#     print("mean is", mean, 'and std is', torch.std(truth).item())
#     rng=torch.linspace(-mean, 3*mean, steps=nsteps)
#     WW,RR=torch.meshgrid( truth, rng)
#     loss=torch.mean( (WW-RR)**2, axis=0)
#     print(loss.shape)
#     plt.plot( rng.numpy(),  loss.numpy())
#     plt.savefig('loss_scan.png')
#     plt.clf()
#     print(kk)

for epoch in range(epochs):

    for phi, weight, truth in tqdm( train_loader ):
        optimizer.zero_grad()
        theloss = loss(model(phi.view(-1,1))[:,0], weight, truth)
        theloss.backward()
        optimizer.step()

    with torch.no_grad():
        for phi, weight, truth in test_loader:
            estimate = model(phi.view(-1,1))[:,0]
            loss_test .append( loss(estimate, weight, truth).item())
            #hist,bins,_=plt.hist( estimate.view(-1,1).numpy(), bins=500, label='Test')

        score = ROOT.TH1F("score", "score", 50, -math.pi, math.pi)
        pred  = ROOT.TH1F("pred", "pred", 50, -math.pi, math.pi)
        norm  = ROOT.TH1F("norm", "norm", 50, -math.pi, math.pi)

        for phi, weight, truth in train_loader_eval:

            estimate = model(phi.view(-1,1))[:,0]

            score.Add( helpers.make_TH1F( np.histogram(phi, np.linspace(-math.pi, math.pi, 50+1), weights=weight*truth) ))
            pred.Add( helpers.make_TH1F( np.histogram(phi, np.linspace(-math.pi, math.pi, 50+1), weights=weight*estimate) ))
            norm .Add( helpers.make_TH1F( np.histogram(phi, np.linspace(-math.pi, math.pi, 50+1), weights=weight)) )

            print ("Loss", theloss.item(), "tot-true", (weight*truth).sum(), 'pred', (weight*estimate).sum() )

            loss_train.append( loss(estimate, weight, truth).item())

            #plt.hist( (weight*(estimate.view(-1)-truth)**2 ).numpy(), bins=200)
            #plt.yscale('log')
            #plt.savefig(os.path.join(user.plot_directory, prefix, f'loss_per_event_{epoch}.png'))
            #plt.clf() 

            #hist,bins,_=plt.hist( estimate.numpy(), bins=bins, label='Train')

        #plt.legend()
        #plt.savefig(os.path.join(user.plot_directory, prefix, f"hist_epoch_{epoch}.png"))
        #plt.clf()

        score.Divide(norm)
        pred.Divide(norm)

        c1 = ROOT.TCanvas()
        score.SetLineStyle(2)
        score.Draw()
        pred.Draw("same")
        c1.Print(os.path.join(user.plot_directory, prefix, 'closure_%03i.png'%epoch))
        syncer.sync()

syncer.makeRemoteGif(os.path.join(user.plot_directory, prefix), pattern="closure_*.png", name=fname, delay=delay)

plt.plot(loss_train[1:], label='train')
plt.plot(loss_test[1:] , label='test')
plt.savefig(os.path.join(user.plot_directory, prefix,'training.png'))
plt.clf()

helpers.copyIndexPHP( os.path.join( user.plot_directory, prefix) )
syncer.sync()
