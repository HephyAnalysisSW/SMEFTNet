import pickle
import random
import ROOT
import os

if __name__=="__main__":
    import sys
    sys.path.append('..')
    sys.path.append('../..')

from tools.DataGenerator import DataGenerator
from tools.WeightInfo    import WeightInfo
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

selection = lambda ar: (ar.genJet_pt>500) & (ar.genJet_SDmass>0) & (abs(ar.dR_genJet_maxq1q2b)<0.6) & (ar.genJet_SDsubjet1_mass>=0)
# -> https://schoef.web.cern.ch/schoef/pytorch/choleskyNN/genTops/training_plots/choleskyNN_genTops_ctWRe_nTraining_519075/lin/epoch.gif
import tools.user as user

data_generator =  DataGenerator(
    #input_files = [os.path.join( user.data_directory, "v6_tsch/tschRefPointNoWidthRW/*_1.root")],
    input_files = [os.path.join( user.data_directory, "v6_tsch/tschRefPointNoWidthRW/*.root")],
        n_split = -1,
        splitting_strategy = "files",
        selection   = selection,
        branches = [
            "genJet_pt", "dR_genJet_maxq1q2b", "genJet_SDmass", "genJet_mass", "genJet_nConstituents", "genJet_SDsubjet1_mass", # "genJet_SDsubjet0_deltaEta", "genJet_SDsubjet0_deltaPhi", "genJet_SDsubjet0_deltaR", "genJet_SDsubjet0_mass", "genJet_SDsubjet1_deltaEta", "genJet_SDsubjet1_deltaPhi", "genJet_SDsubjet1_deltaR", "genJet_tau1", "genJet_tau2", "genJet_tau3", "genJet_tau4", "genJet_tau21", "genJet_tau32", "genJet_ecf1", "genJet_ecf2", "genJet_ecf3", "genJet_ecfC1", "genJet_ecfC2", "genJet_ecfC3", "genJet_ecfD", "genJet_ecfDbeta2", "genJet_ecfM1", "genJet_ecfM2", "genJet_ecfM3", "genJet_ecfM1beta2", "genJet_ecfM2beta2", "genJet_ecfM3beta2", "genJet_ecfN1", "genJet_ecfN2", "genJet_ecfN3", "genJet_ecfN1beta2", "genJet_ecfN2beta2", "genJet_ecfN3beta2", "genJet_ecfU1", "genJet_ecfU2", "genJet_ecfU3", "genJet_ecfU1beta2", "genJet_ecfU2beta2", "genJet_ecfU3beta2", 

        "ngen", "gen_pt", "gen_etarel", "gen_phirel", #"gen_eta", "gen_phi", "gen_pdgId", "gen_charge", "gen_type",
        "ctWRe_coeff"
         ]
    )

class genTopsModel:
    def __init__( self ):
        self.data_generator = data_generator
 
    @staticmethod
    def getEvents(data):
        padding = 100 
        ctwRe_coeff = DataGenerator.vector_branch( data, 'ctWRe_coeff',padding_target=3 )
        weights = ctwRe_coeff[:, 0]
        truth = ctwRe_coeff[:, 1]/ctwRe_coeff[:, 0]

        pts = DataGenerator.vector_branch( data, 'gen_pt',padding_target=padding )
        ptmask = torch.ones_like( torch.Tensor(pts) ).to(device) #  (pts > 5)
        detas = DataGenerator.vector_branch( data, 'gen_etarel',padding_target=padding ) # [ptmask]
        dphis = DataGenerator.vector_branch( data, 'gen_phirel',padding_target=padding ) # [ptmask
        detas = torch.Tensor(detas).to(device) * ptmask  # 0-pad the pt < 5
        pts   = torch.Tensor(pts).to(device)   * ptmask  # 0-pad the pt < 5
        dphis = torch.Tensor(dphis).to(device) * ptmask  # 0-pad the pt < 5
        return pts, torch.stack( (dphis, detas), axis=-1), torch.tensor(weights).to(device), torch.tensor(truth).to(device)

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

random_eft = make_eft(**{v:random.random() for v in weightInfo.variables} )
sm         = make_eft()

if __name__=="__main__":
    wilson_coefficients = weightInfo.variables 
else:
    wilson_coefficients = ['ctWRe']

vector_branches = ["gen_pt", "gen_etarel", "gen_phirel"]

feature_names = [   
    "genJet_pt", "dR_genJet_maxq1q2b", "genJet_mass", "genJet_nConstituents", "genJet_SDmass", "genJet_SDsubjet0_deltaEta", "genJet_SDsubjet0_deltaPhi", "genJet_SDsubjet0_deltaR", "genJet_SDsubjet0_mass", "genJet_SDsubjet1_deltaEta", "genJet_SDsubjet1_deltaPhi", "genJet_SDsubjet1_deltaR", "genJet_SDsubjet1_mass", "genJet_tau1", "genJet_tau2", "genJet_tau3", "genJet_tau4", "genJet_tau21", "genJet_tau32", "genJet_ecf1", "genJet_ecf2", "genJet_ecf3", "genJet_ecfC1", "genJet_ecfC2", "genJet_ecfC3", "genJet_ecfD", "genJet_ecfDbeta2", "genJet_ecfM1", "genJet_ecfM2", "genJet_ecfM3", "genJet_ecfM1beta2", "genJet_ecfM2beta2", "genJet_ecfM3beta2", "genJet_ecfN1", "genJet_ecfN2", "genJet_ecfN3", "genJet_ecfN1beta2", "genJet_ecfN2beta2", "genJet_ecfN3beta2", "genJet_ecfU1", "genJet_ecfU2", "genJet_ecfU3", "genJet_ecfU1beta2", "genJet_ecfU2beta2", "genJet_ecfU3beta2", 
                    ]

def make_combinations( coefficients ):
    combinations = []
    for comb in weightInfo.combinations:
        good = True
        for k in comb:
            if k not in coefficients:
                good = False
                break
        if good:
            combinations.append(comb)
    return combinations

def getEvents( data ):
    combinations = make_combinations( wilson_coefficients )
    coeffs = data_generator.vector_branch(data, 'p_C')
    return {key:data_generator.vector_branch( data, key ) for key in vector_branches}, {comb:coeffs[:,weightInfo.combinations.index(comb)] for comb in combinations}

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
    "genJet_mass"               :{'binning':[50,150,200], 'tex':'M(jet) unpruned'},
    "genJet_nConstituents"      :{'binning':[50,30,230], 'tex':'n-constituents'},
    "genJet_SDmass"             :{'binning':[50,150,200], 'tex':'M_{SD}(jet)'},
    "dR_genJet_maxq1q2b"        :{'binning':[50,0,1], 'tex':'max #Delta R(jet, q1/q2/b})'},
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

if __name__=="__main__":
   
    # load some events and their weights 
    x, w = getEvents(data_generator[0])

