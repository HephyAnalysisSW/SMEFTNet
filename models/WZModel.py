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

selection = lambda ar: (ar.genJet_pt>500) & (ar.dR_genJet_maxq1q2 < 0.6) & (ar.genJet_SDmass > 70) & (ar.genJet_SDmass < 110)

reweight_pkl = '/pnfs/psi.ch/cms/trivcat/store/user/sesanche/HadronicSMEFT/gridpacks/WZto2LNoRef_HT300_reweight_card.pkl'
weightInfo = WeightInfo(reweight_pkl)
weightInfo.set_order(2)

def angle( x, y):
    return torch.arctan2( torch.Tensor(y), torch.Tensor(x))
def dphi(phi1,phi2):
    dph=phi1-phi2
    return dph + 2*np.pi*(dph < -np.pi) - 2*np.pi*(dph > np.pi)
class WZModel:
    def __init__( self, what='lab'):
        self.what = what

        if self.what == 'lab': 
            variables=[
                "genJet_pt", "genJet_SDmass",'dR_genJet_maxq1q2',
                "ngen", 'gen_pt_lab', "gen_Deta_lab", "gen_Dphi_lab", 
                "parton_hadV_q2_phi", "parton_hadV_q2_eta",
                "parton_hadV_q1_phi", "parton_hadV_q1_eta",
                'p_C',
            ]
        elif self.what == 'VV':
            variables=["genJet_pt", "genJet_SDmass",'dR_genJet_maxq1q2',
                       "ngen", 'gen_pt_lab', 'gen_Theta_VV', 'gen_phi_VV',
                       'parton_hadV_angle_phi',
                       'p_C',
            ]

        self.data_generator =  DataGenerator(
            input_files = [os.path.join( user.data_directory, "v6/WZto2L_HT300/*.root" )],
            n_split = 200,
            splitting_strategy = "files",
            selection   = selection,
            branches = variables,
        )

    @staticmethod
    def getEvents(data, what):

        if what == 'lab':
            q12_dphi = dphi(DataGenerator.scalar_branches(data, ['parton_hadV_q2_phi']),DataGenerator.scalar_branches(data, ['parton_hadV_q1_phi']))
            q12_deta = DataGenerator.scalar_branches(data, ['parton_hadV_q2_eta'])-DataGenerator.scalar_branches(data, ['parton_hadV_q1_eta'])
            pts = DataGenerator.vector_branch( data, 'gen_pt_lab',padding_target=40 ) 
            ptmask = torch.ones_like( torch.Tensor(pts) ) #  (pts > 5)
            detas = DataGenerator.vector_branch( data, 'gen_Deta_lab',padding_target=40 ) # [ptmask]
            dphis = DataGenerator.vector_branch( data, 'gen_Dphi_lab',padding_target=40 ) # [ptmask
            detas = torch.Tensor(detas) * ptmask  # 0-pad the pt < 5
            pts   = torch.Tensor(pts)   * ptmask  # 0-pad the pt < 5
            dphis = torch.Tensor(dphis) * ptmask  # 0-pad the pt < 5
            coeffs = torch.Tensor(DataGenerator.vector_branch(data, 'p_C', padding_target=len(weightInfo.combinations)))
            return (torch.Tensor(pts), torch.stack(( detas, dphis),axis=2), coeffs, angle(q12_dphi, q12_deta)[:,0])

        elif what == 'VV':
            pts = torch.Tensor(DataGenerator.vector_branch( data, 'gen_pt_lab',padding_target=40 ) )
            thetas = torch.Tensor(DataGenerator.vector_branch( data, 'gen_Theta_VV',padding_target=40 ))
            dphis  = torch.Tensor(DataGenerator.vector_branch( data, 'gen_phi_VV',padding_target=40 ))
            parton_hadV_q1_phi = torch.Tensor(DataGenerator.scalar_branches(data, ['parton_hadV_angle_phi']))
            coeffs = torch.Tensor(DataGenerator.vector_branch(data, 'p_C', padding_target=len(weightInfo.combinations)))
            return (torch.Tensor(pts), torch.stack(( thetas, dphis),axis=2), coeffs, parton_hadV_q1_phi[:,0])
            

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
     

