import uproot
import torch 
import numpy as np 
from glob import glob
from torch.utils.data import Dataset

class RootDataset(Dataset):
    def __init__(self, files, branches):
        self.tree_name='Events'
        self.input_files = [ f for f in glob(files)]
        self.branches = branches
        array = uproot.concatenate([f+':'+self.tree_name for f in self.input_files], self.branches, library='np')
        self.dphi = array['parton_hadV_angle_phi']
        self.weight=np.stack(array['p_C'],axis=0)
        self.weight = self.weight[:,12]/self.weight[:,0]
        clip_mask = (np.abs(self.weight) < 1e3)
        self.dphi = self.dphi[clip_mask]
        self.weight = self.weight[clip_mask]

    def __len__(self):
        return self.weight.shape[0]

    def __getitem__( self, idx ):
        return self.dphi[idx], self.weight[idx]
        
if __name__=='__main__':
    dl=RootDataset("/pnfs/psi.ch/cms/trivcat/store/user/sesanche/SMEFTNet/v6/WZto2L_HT300/*0.root", ['parton_hadV_angle_phi', 'p_C'])
    phi, weight=dl[:]
    print(weight)
    import matplotlib.pyplot as plt 
    plt.hist( weight, bins=500 ) 
    plt.savefig( 'weight.png')
    plt.clf()
