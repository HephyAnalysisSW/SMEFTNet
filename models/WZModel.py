import pickle
import random
import ROOT

if __name__=="__main__":
    import sys
    sys.path.append('/work/sesanche/SMEFTNet')

from tools.DataGenerator import DataGenerator
import torch

selection = lambda ar: (ar.genJet_pt>500) & (ar.dR_genJet_maxq1q2 < 0.6) & (ar.genJet_SDmass > 70) & (ar.genJet_SDmass < 110)
vector_branches = ['gen_pt_lab', "gen_Deta_lab", "gen_Dphi_lab"]
feature_names = [ "genJet_pt", "genJet_SDmass" ]
targets = [
    "parton_hadV_q2_phi", "parton_hadV_q2_eta",
    "parton_hadV_q1_phi", "parton_hadV_q1_eta",
]

def angle( x, y):
    angle= torch.arctan2( torch.Tensor(y), torch.Tensor(x))
    return torch.stack((torch.cos(angle), torch.sin(angle)),axis=2)

class WZModel:
    def __init__( self ):

        self.dg =  DataGenerator(
            input_files = ["/pnfs/psi.ch/cms/trivcat/store/user/sesanche/SMEFTNet/v6/*.root"],
            n_split = 1,
            splitting_strategy = "files",
            selection   = selection,
            branches = [
                "genJet_pt", "genJet_SDmass",'dR_genJet_maxq1q2',
                "ngen", 'gen_pt_lab', "gen_Deta_lab", "gen_Dphi_lab", 
                "parton_hadV_q2_phi", "parton_hadV_q2_eta",
                "parton_hadV_q1_phi", "parton_hadV_q1_eta",
            ]
        )


    def getEvents( self, nTraining ):
        print("Getting events")
        self.dg.load(-1, small=nTraining )
        print("Loaded")
        q12_dphi = self.dg.scalar_branches(['parton_hadV_q2_phi'])-self.dg.scalar_branches(['parton_hadV_q1_phi'])
        print("Gotten branch")

        q12_deta = self.dg.scalar_branches(['parton_hadV_q2_eta'])-self.dg.scalar_branches(['parton_hadV_q1_eta'])
        print("Gotten branch2")

        pts = self.dg.vector_branch( 'gen_pt_lab',padding_target=40 ) 
        ptmask = torch.ones_like( torch.Tensor(pts) ) #  (pts > 5)
        detas = self.dg.vector_branch( 'gen_Deta_lab',padding_target=40 ) # [ptmask]
        dphis = self.dg.vector_branch( 'gen_Dphi_lab',padding_target=40 ) # [ptmask
        detas = torch.Tensor(detas) * ptmask  # 0-pad the pt < 5
        pts   = torch.Tensor(pts)   * ptmask  # 0-pad the pt < 5
        dphis = torch.Tensor(dphis) * ptmask  # 0-pad the pt < 5
        return (torch.Tensor(pts), angle( detas, dphis), None, angle(q12_dphi, q12_deta)[:,0,:])


if __name__=="__main__":
    model=WZModel()
    # load some events and their weights 
    print( model.getEvents(1000))

