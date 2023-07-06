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
import torch
import tools.user as user

selection = lambda ar: (ar.genJet_pt>500) & (ar.dR_genJet_maxq1q2 < 0.6) & (ar.genJet_SDmass > 70) & (ar.genJet_SDmass < 110)
vector_branches = ['gen_pt_lab', "gen_Deta_lab", "gen_Dphi_lab"]
feature_names = [ "genJet_pt", "genJet_SDmass" ]
targets = [
    "parton_hadV_q2_phi", "parton_hadV_q2_eta",
    "parton_hadV_q1_phi", "parton_hadV_q1_eta",
]

def angle( x, y):
    return torch.arctan2( torch.Tensor(y), torch.Tensor(x))
def dphi(phi1,phi2):
    dph=phi1-phi2
    return dph + 2*np.pi*(dph < -np.pi) - 2*np.pi*(dph > np.pi)

class WZModel:
    def __init__( self, charged=False):
        self.charged = charged
        self.data_generator =  DataGenerator(
            input_files = [os.path.join( user.data_directory, "v6/WZto2L_HT300/*.root" )],
            n_split = 200,
            splitting_strategy = "files",
            selection   = selection,
            branches = [
                "genJet_pt", "genJet_SDmass",'dR_genJet_maxq1q2',
                "ngen", 'gen_pt_lab', "gen_Deta_lab", "gen_Dphi_lab", "gen_charge", 
                "parton_hadV_q2_phi", "parton_hadV_q2_eta",
                "parton_hadV_q1_phi", "parton_hadV_q1_eta",
            ]
        )

    #@staticmethod
    def getEvents(self, data):
        #print("Getting events")
        #self.data_generator._load(-1)
        #print("Loaded")
        q12_dphi = dphi(DataGenerator.scalar_branches(data, ['parton_hadV_q2_phi']),DataGenerator.scalar_branches(data, ['parton_hadV_q1_phi']))
        #print("Gotten branch")

        q12_deta = DataGenerator.scalar_branches(data, ['parton_hadV_q2_eta'])-DataGenerator.scalar_branches(data, ['parton_hadV_q1_eta'])
        #print("Gotten branch2")
        padding = 40
        pts = DataGenerator.vector_branch( data, 'gen_pt_lab',padding_target=padding ) 
        ptmask = torch.ones_like( torch.Tensor(pts) ) #  (pts > 5)
        detas = DataGenerator.vector_branch( data, 'gen_Deta_lab',padding_target=padding ) # [ptmask]
        dphis = DataGenerator.vector_branch( data, 'gen_Dphi_lab',padding_target=padding ) # [ptmask
        detas = torch.Tensor(detas) * ptmask  # 0-pad the pt < 5
        pts   = torch.Tensor(pts)   * ptmask  # 0-pad the pt < 5
        dphis = torch.Tensor(dphis) * ptmask  # 0-pad the pt < 5
        #print(torch.stack(( detas, dphis)).shape, torch.Tensor(pts).shape)
        if self.charged:
            charge   = DataGenerator.vector_branch( data, 'gen_charge',padding_target=padding ) # [ptmask
            features = (torch.Tensor(charge) * ptmask).view(-1,padding,1)  
        else:
            features = None 
        return (torch.Tensor(pts), torch.stack(( detas, dphis),axis=2), features, None, angle(q12_dphi, q12_deta)[:,0])


if __name__=="__main__":

    # reading file by file (because n_split is -1); choose n_split = 10 for 10 chunks, or 1 if you want to read the whole dataset
    model = WZModel()
    total = 0
    for data in model.data_generator:
        pts, gamma, _, truth   = model.getEvents(data)
        print ("len(pts)", len(pts))
        total += len(pts)

    print ("Read in total",total,"events")
    print ("Reading all events at once: Got", len(model.getEvents(model.data_generator[-1])[0]) )
     

