import pickle
import random
import ROOT
import os
from math import pi

if __name__=="__main__":
    import sys
    sys.path.append('..')
    sys.path.append('../..')

from tools.DataGenerator import DataGenerator
from tools.WeightInfo    import WeightInfo
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

reweight_pkl = '/eos/vbc/group/cms/robert.schoefbeck/gridpacks/ParticleNet/t-sch-RefPoint-noWidthRW_reweight_card.pkl'
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

wilson_coefficients = ['ctWRe']

random_eft = make_eft(**{v:random.random() for v in weightInfo.variables} )
sm         = make_eft()

def make_combinations( coefficients ):
    combinations = []
    for comb in weightInfo.combinations:
        good = True
        for k in comb:
            if k not in wilson_coefficients:
                good = False
                break
        if good:
            combinations.append(comb)
    return combinations

selection = lambda ar: (ar.genJet_pt>500) & (ar.genJet_SDmass>0) & (abs(ar.dR_genJet_maxq1q2b)<0.6) & (ar.genJet_SDsubjet1_mass>=0)
# -> https://schoef.web.cern.ch/schoef/pytorch/choleskyNN/genTops/training_plots/choleskyNN_genTops_ctWRe_nTraining_519075/lin/epoch.gif
import tools.user as user

scalar_features = [   
    "parton_top_decayAngle_theta", "parton_top_decayAngle_phi", "genJet_eta", "genJet_pt", "dR_genJet_maxq1q2b", "genJet_mass", "genJet_nConstituents", "genJet_SDmass", "genJet_SDsubjet0_deltaEta", "genJet_SDsubjet0_deltaPhi", "genJet_SDsubjet0_deltaR", "genJet_SDsubjet0_mass", "genJet_SDsubjet1_deltaEta", "genJet_SDsubjet1_deltaPhi", "genJet_SDsubjet1_deltaR", "genJet_SDsubjet1_mass", "genJet_tau1", "genJet_tau2", "genJet_tau3", "genJet_tau4", "genJet_tau21", "genJet_tau32", "genJet_ecf1", "genJet_ecf2", "genJet_ecf3", "genJet_ecfC1", "genJet_ecfC2", "genJet_ecfC3", "genJet_ecfD", "genJet_ecfDbeta2", "genJet_ecfM1", "genJet_ecfM2", "genJet_ecfM3", "genJet_ecfM1beta2", "genJet_ecfM2beta2", "genJet_ecfM3beta2", "genJet_ecfN1", "genJet_ecfN2", "genJet_ecfN3", "genJet_ecfN1beta2", "genJet_ecfN2beta2", "genJet_ecfN3beta2", "genJet_ecfU1", "genJet_ecfU2", "genJet_ecfU3", "genJet_ecfU1beta2", "genJet_ecfU2beta2", "genJet_ecfU3beta2", 
    "parton_q1_pt", "parton_q2_pt", "parton_b_pt", "parton_W_pt",
                    ]

data_generator = DataGenerator(
            input_files = [os.path.join( user.data_directory, "v6_tsch/tschRefPointNoWidthRW/*.root")],
                n_split = -1,
                splitting_strategy = "files",
                selection   = selection,
                branches = [
                    "genJet_pt", "dR_genJet_maxq1q2b", "genJet_SDmass", "genJet_mass", "genJet_nConstituents", "genJet_SDsubjet1_mass", 
                    "ngen", "gen_pt", "gen_etarel", "gen_phirel", "gen_charge", "p_C",
                    "ctWRe_coeff"
                 ] + scalar_features
            )

def angle( x, y):
    return torch.arctan2( torch.Tensor(y), torch.Tensor(x))

#def dphi(phi1,phi2):
#    dph=phi1-phi2
#    return dph + 2*np.pi*(dph < -np.pi) - 2*np.pi*(dph > np.pi)

class genTopsModel:
    def __init__( self, min_pt = 0, padding=100, small=False, features = [], truth_interval = None, truth = 'ctWRe'):
        self.scalar_features = scalar_features
        self.features = features
        self.padding  = padding
        self.min_pt   = min_pt
        self.truth_interval = truth_interval
        self.data_generator = data_generator 
        self.data_generator.branches += features 
        self.truth = truth

    def set_truth_mask( self, truth):
        self.mask = torch.ones(len(truth),dtype=bool).to(device)
        if self.truth_interval is not None:
            if self.truth_interval[0] is not None:
                self.mask*=(truth>self.truth_interval[0])
            if self.truth_interval[1] is not None:
                self.mask*=(truth<=self.truth_interval[1])

    #@staticmethod
    def getEvents(self, data):
        ctwRe_coeff = DataGenerator.vector_branch( data, 'ctWRe_coeff',padding_target=3 )
        weights = ctwRe_coeff[:, 0]

        if self.truth == 'ctWRe':
            truth = torch.Tensor(ctwRe_coeff[:, 1]).to(device)
        else:
            truth = torch.Tensor(DataGenerator.scalar_branches( data, [self.truth] )[:,0]).to(device)

        self.set_truth_mask(truth)
        #truth = -0.15*torch.ones_like(truth)

        pts = DataGenerator.vector_branch( data, 'gen_pt',padding_target=self.padding )
        ptmask =  torch.Tensor((pts >= self.min_pt)).to(device) #  (pts > 5)
        detas = DataGenerator.vector_branch( data, 'gen_etarel',padding_target=self.padding ) # [ptmask]
        dphis = DataGenerator.vector_branch( data, 'gen_phirel',padding_target=self.padding ) # [ptmask
        pts   = torch.Tensor(pts).to(device)   * ptmask  
        detas = torch.Tensor(detas).to(device) * ptmask  
        dphis = torch.Tensor(dphis).to(device) * ptmask  
        #return pts, torch.stack( (dphis, detas), axis=-1), torch.tensor(weights).to(device), torch.tensor(truth).to(device)

        
        features = None
        for feature in self.features: 
            branch_data = DataGenerator.vector_branch( data, feature, padding_target=self.padding ) 
            if features is None:
                features = (torch.Tensor(branch_data).to(device) * ptmask).view(-1,self.padding,1)
            else:
                features = torch.cat( (features, (torch.Tensor(branch_data).to(device)* ptmask).view(-1,self.padding,1)), dim=2)

        return  pts[self.mask],\
                torch.stack( (dphis, detas), axis=-1)[self.mask],\
                features[self.mask] if features is not None else None,\
                None,\
                torch.Tensor( weights ).to(device)[self.mask],\
                truth[self.mask]

    def getWeightDict( self, data ):
        ctwRe_coeff = DataGenerator.vector_branch( data, 'ctWRe_coeff',padding_target=3 )
        truth = torch.Tensor(ctwRe_coeff[:, 1]).to(device)
        self.set_truth_mask(truth)
        combinations = make_combinations( wilson_coefficients )
        coeffs = data_generator.vector_branch(data, 'p_C')
        return {comb:coeffs[:,weightInfo.combinations.index(comb)][self.mask] for comb in combinations}

    def getScalarFeatures( self, data, branches = None):
        ctwRe_coeff = DataGenerator.vector_branch( data, 'ctWRe_coeff',padding_target=3 )
        truth = torch.Tensor(ctwRe_coeff[:, 1]).to(device)
        self.set_truth_mask(truth)
        return DataGenerator.scalar_branches( data, self.scalar_features if branches is None else branches)[self.mask]

tex = {"ctWRe":"C_{tW}^{Re}", "ctWIm":"C_{tW}^{Im}", "ctBIm":"C_{tB}^{Im}", "ctBRe":"C_{tB}^{Re}", "cHt":"C_{Ht}", 'cHtbRe':'C_{Htb}^{Re}', 'cHtbIm':'C_{Htb}^{Im}', 'cHQ3':'C_{HQ}^{(3)}'}

#['ctWRe', 'ctBRe', 'cHQ3', 'cHt', 'cHtbRe', 'ctWIm', 'ctBIm', 'cHtbIm']
eft_plot_points = [
    {'color':ROOT.kBlack,       'eft':sm, 'tex':"SM"},
    {'color':ROOT.kMagenta-4,   'eft':make_eft(ctWRe=1),   'tex':"c_{tW}^{Re}=1",},
    {'color':ROOT.kMagenta+2,   'eft':make_eft(ctWRe=-1),   'tex':"c_{tW}^{Re}=-1",},
    {'color':ROOT.kGreen-4,     'eft':make_eft(ctWRe=3),   'tex':"c_{tW}^{Re}=3",},
    {'color':ROOT.kGreen+2,     'eft':make_eft(ctWRe=-3),   'tex':"c_{tW}^{Re}=-3",},
    {'color':ROOT.kBlue+2,      'eft':make_eft(ctWRe=5),   'tex':"c_{tW}^{Re}=5",},
    {'color':ROOT.kBlue-4,      'eft':make_eft(ctWRe=-5),   'tex':"c_{tW}^{Re}=-5",},
    ]

plot_options =  {
    "genJet_pt"                 :{'binning':[50,500,2000], 'tex':'p_{T}(jet)'},
    "genJet_eta"                :{'binning':[50,-3,3], 'tex':'#eta(jet)'},
    "genJet_mass"               :{'binning':[50,150,200], 'tex':'M(jet) unpruned'},
    "genJet_nConstituents"      :{'binning':[50,30,230], 'tex':'n-constituents'},
    "genJet_SDmass"             :{'binning':[50,150,200], 'tex':'M_{SD}(jet)'},
    "dR_genJet_maxq1q2b"        :{'binning':[50,0,1], 'tex':'max #Delta R(jet, q1/q2/b})'},
    "parton_top_decayAngle_theta":{'binning':[50,0,pi], 'tex':'#theta(t-decay)'},
    "parton_top_decayAngle_phi" :{'binning':[50,-pi,pi], 'tex':'#phi(t-decay)'},
    "parton_q1_pt"              :{'binning':[50,0,300], 'tex':'p_{T}(q_1)'},
    "parton_q2_pt"              :{'binning':[50,0,300], 'tex':'p_{T}(q_2)'}, 
    "parton_b_pt"               :{'binning':[50,0,300], 'tex':'p_{T}(b)'},
    "parton_W_pt"               :{'binning':[50,0,300], 'tex':'p_{T}(W)'},
    "genJet_SDsubjet0_deltaEta" :{'binning':[50,-0.6,0.6], 'tex':'#Delta#eta(jet,jet_{SD,0})'},
    "genJet_SDsubjet0_deltaPhi" :{'binning':[50,-0.6,0.6], 'tex':'#Delta#phi(jet,jet_{SD,0})'},
    "genJet_SDsubjet0_deltaR"   :{'binning':[50,0,0.7], 'tex':'#Delta R(jet,jet_{SD,0})'},
    "genJet_SDsubjet0_mass"     :{'binning':[50,0,200], 'tex':'M_{SD}(jet_{0})'},
    "genJet_SDsubjet1_deltaEta" :{'binning':[50,-0.6,0.6], 'tex':'#Delta#eta(jet,jet_{SD,1})'},
    "genJet_SDsubjet1_deltaPhi" :{'binning':[50,-0.6,0.6], 'tex':'#Delta#phi(jet,jet_{SD,1})'},
    "genJet_SDsubjet1_deltaR"   :{'binning':[50,0,0.7], 'tex':'#Delta R(jet,jet_{SD,1})'},
    "genJet_SDsubjet1_mass"     :{'binning':[50,0,200], 'tex':'M_{SD}(jet_{1})'},
    "genJet_tau1"               :{'binning':[50,0,1], 'tex':'#tau_{1}'},
    "genJet_tau2"               :{'binning':[50,0,.5],'tex':'#tau_{2}'},
    "genJet_tau3"               :{'binning':[50,0,.3],'tex':'#tau_{3}'},
    "genJet_tau4"               :{'binning':[50,0,.3],'tex':'#tau_{4}'},
    "genJet_tau21"              :{'binning':[50,0,1], 'tex':'#tau_{21}'},
    "genJet_tau32"              :{'binning':[50,0,1], 'tex':'#tau_{32}'},
#https://github.com/cms-sw/cmssw/blob/6d2f66057131baacc2fcbdd203588c41c885b42c/RecoJets/JetProducers/python/ECF_cff.py
    "genJet_ecf1"               :{'binning':[50,0,2000], 'tex':"ecf1"},
    "genJet_ecf2"               :{'binning':[50,0,400000], 'tex':"ecf2"},
    "genJet_ecf3"               :{'binning':[50,0,4000000], 'tex':"ecf3"},
    "genJet_ecfC1"              :{'binning':[50,0,.5], 'tex':"ecfC1"},
    "genJet_ecfC2"              :{'binning':[50,0,.5], 'tex':"ecfC2"},
    "genJet_ecfC3"              :{'binning':[50,0,.5], 'tex':"ecfC3"},
    "genJet_ecfD"               :{'binning':[50,0,8], 'tex':"ecfD"},
    "genJet_ecfDbeta2"          :{'binning':[50,0,20], 'tex':"ecfDbeta2"},
    "genJet_ecfM1"              :{'binning':[50,0,0.35], 'tex':"ecfM1"},
    "genJet_ecfM2"              :{'binning':[50,0,0.2], 'tex':"ecfM2"},
    "genJet_ecfM3"              :{'binning':[50,0,0.2], 'tex':"ecfM3"},
    "genJet_ecfM1beta2"         :{'binning':[50,0,0.35], 'tex':"ecfM1beta2"},
    "genJet_ecfM2beta2"         :{'binning':[50,0,0.2], 'tex':"ecfM2beta2"},
    "genJet_ecfM3beta2"         :{'binning':[50,0,0.2], 'tex':"ecfM3beta2"},
    "genJet_ecfN1"              :{'binning':[50,0,0.5], 'tex':"ecfN1"},
    "genJet_ecfN2"              :{'binning':[50,0,0.5], 'tex':"ecfN2"},
    "genJet_ecfN3"              :{'binning':[50,0,5], 'tex':"ecfN3"},
    "genJet_ecfN1beta2"         :{'binning':[50,0,0.5], 'tex':"ecfN1beta2"},
    "genJet_ecfN2beta2"         :{'binning':[50,0,0.5], 'tex':"ecfN2beta2"},
    "genJet_ecfN3beta2"         :{'binning':[50,0,5], 'tex':"ecfN3beta2"},
    "genJet_ecfU1"              :{'binning':[50,0,0.5], 'tex':"ecfU1"},
    "genJet_ecfU2"              :{'binning':[50,0,0.04], 'tex':"ecfU2"},
    "genJet_ecfU3"              :{'binning':[50,0,0.004], 'tex':"ecfU3"},
    "genJet_ecfU1beta2"         :{'binning':[50,0,0.5], 'tex':"ecfU1beta2"},
    "genJet_ecfU2beta2"         :{'binning':[50,0,0.04], 'tex':"ecfU2beta2"},
    "genJet_ecfU3beta2"         :{'binning':[50,0,0.004], 'tex':"ecfU3beta2"},
}
feature_names   = list(plot_options.keys())

if __name__=="__main__":
   
    # load some events and their weights 
    x, w = getEvents(data_generator[0])

