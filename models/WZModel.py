import pickle
import random
import ROOT
import os
import numpy as np 

if __name__=="__main__":
    import sys
    #sys.path.append('/work/sesanche/SMEFTNet')
    sys.path.append('..')
    sys.path.append('../..')

from tools.DataGenerator import DataGenerator
from tools.WeightInfo    import WeightInfo
import torch
import tools.user as user

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

selection = lambda ar: (ar.genJet_pt>500) & (ar.dR_genJet_maxq1q2 < 0.6) & (ar.genJet_SDmass > 70) & (ar.genJet_SDmass < 110)

reweight_pkl = os.path.join( user.pkl_directory, 'WZto2LNoRef_HT300_reweight_card.pkl' )
weightInfo = WeightInfo(reweight_pkl)
weightInfo.set_order(2)

def angle( x, y):
    return torch.arctan2( y, x)
def dphi(phi1,phi2):
    dph=phi1-phi2
    return dph + 2*np.pi*(dph < -np.pi) - 2*np.pi*(dph > np.pi)

class WZModel:
    def __init__( self, charged=False,  scalar_features = [], what='lab'):
        self.what = what
        self.charged = charged
        if self.what == 'lab': 
            branches=[
                "genJet_pt", "genJet_SDmass",'dR_genJet_maxq1q2',
                "ngen", 'gen_pt_lab', "gen_Deta_lab", "gen_Dphi_lab",
                "parton_hadV_q2_phi", "parton_hadV_q2_eta",
                "parton_hadV_q1_phi", "parton_hadV_q1_eta",
                'p_C',
            ]
        elif self.what == 'VV':
            branches=["genJet_pt", "genJet_SDmass",'dR_genJet_maxq1q2',
               "ngen", 'gen_pt_lab', 'gen_Theta_VV', 'gen_phi_VV',
               'parton_hadV_angle_phi',
               'p_C',
            ]
        else:
            raise NotImplementedError

        if self.charged:
            branches+=["gen_charge"]

        if scalar_features is not None and len(scalar_features)>0:
            self.scalar_features = scalar_features
            if not type(self.scalar_features)==type([]): raise RuntimeError ("Need a list of scalar features")
            for feature in scalar_features:
                if feature not in branches:
                    branches.append( feature)
        else:
            self.scalar_features = None 
        

        self.data_generator =  DataGenerator(
            input_files = [os.path.join( user.data_directory, "v6/WZto2L_HT300/*.root" )],
            n_split             = 200,
            splitting_strategy  = "files",
            selection           = selection,
            branches            = branches,
        )
        

    def getEvents(self, data):
        padding = 40
        pts = DataGenerator.vector_branch( data, 'gen_pt_lab',padding_target=padding ) 
        ptmask = torch.ones_like( torch.Tensor(pts) ).to(device) #  (pts > 5)
        pts    = torch.Tensor(pts).to(device)   * ptmask  # 0-pad the pt < 5

        weights = torch.Tensor(DataGenerator.vector_branch(data, 'p_C', padding_target=len(weightInfo.combinations))).to(device)

        if self.charged:
            charge   = DataGenerator.vector_branch( data, 'gen_charge',padding_target=padding ) # [ptmask
            features = (torch.Tensor(charge).to(device) * ptmask).view(-1,padding,1)  
        else:
            features = None

        if self.scalar_features: 
            scalar_features = torch.Tensor(DataGenerator.scalar_branches(data, self.scalar_features)).to(device)
        else:
            scalar_features = None

        if self.what == 'lab':
            detas = torch.Tensor(DataGenerator.vector_branch( data, 'gen_Deta_lab',padding_target=padding )).to(device)
            dphis = torch.Tensor(DataGenerator.vector_branch( data, 'gen_Dphi_lab',padding_target=padding )).to(device)
            angles = torch.stack(( detas*ptmask, dphis*ptmask),axis=2)

            q12_dphi = dphi(torch.Tensor(DataGenerator.scalar_branches(data, ['parton_hadV_q2_phi'])).to(device),torch.Tensor(DataGenerator.scalar_branches(data, ['parton_hadV_q1_phi'])).to(device))
            q12_deta =      torch.Tensor(DataGenerator.scalar_branches(data, ['parton_hadV_q2_eta'])).to(device)-torch.Tensor(DataGenerator.scalar_branches(data, ['parton_hadV_q1_eta'])).to(device)
            truth = angle(q12_dphi, q12_deta)[:,0]

        elif self.what == 'VV':

            thetas = torch.Tensor(DataGenerator.vector_branch( data, 'gen_Theta_VV',padding_target=padding )).to(device)
            dphis  = torch.Tensor(DataGenerator.vector_branch( data, 'gen_phi_VV',padding_target=padding )).to(device)
            angles = torch.stack(( thetas*ptmask, dphis*ptmask),axis=2)

            parton_hadV_q1_phi = torch.Tensor(DataGenerator.scalar_branches(data, ['parton_hadV_angle_phi'])).to(device)
            truth = parton_hadV_q1_phi[:,0]

        return pts, angles, features, scalar_features, weights, truth 
            

if __name__=="__main__":

    # reading file by file (because n_split is -1); choose n_split = 10 for 10 chunks, or 1 if you want to read the whole dataset
    model = WZModel()
    total = 0
    for data in model.data_generator:
        pts, gamma, features, scalar_features, weights, truth   = model.getEvents(data)
        print ("len(pts)", len(pts))
        total += len(pts)

    print ("Read in total",total,"events")
    print ("Reading all events at once: Got", len(model.getEvents(model.data_generator[-1])[0]) )
     

