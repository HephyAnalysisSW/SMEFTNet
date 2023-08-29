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

reweight_pkl = '/eos/vbc/group/cms/robert.schoefbeck/gridpacks/ParticleNet/TT01jDebug_reweight_card.pkl'
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

wilson_coefficients = ['ctGRe']

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

selection = lambda ar: (ar.nrecoLep>=1) & (ar.delphesJet_pt >= 500) & (ar.delphesJet_dR_hadTop_maxq1q2b < 0.8) & (ar.nrecoJet >= 4) 

import tools.user as user

branches = [   
    "nrecoLep", "delphesJet_pt", "delphesJet_dR_hadTop_maxq1q2b", "nrecoJet", 

    "parton_hadTop_pt", "parton_hadTop_eta", "parton_hadTop_phi", "parton_lepTop_pt", "parton_lepTop_eta", "parton_lepTop_phi", "parton_lepTop_lep_pt", "parton_lepTop_lep_eta", "parton_lepTop_lep_phi", "parton_lepTop_nu_pt", "parton_lepTop_nu_eta", "parton_lepTop_nu_phi", "parton_lepTop_b_pt", "parton_lepTop_b_eta", "parton_lepTop_b_phi", "parton_lepTop_W_pt", "parton_lepTop_W_eta", "parton_lepTop_W_phi", 

    "delphesJet_pt", "delphesJet_eta", "delphesJet_phi", "delphesJet_nConstituents", "delphesJet_SDmass", "delphesJet_SDsubjet0_deltaEta", "delphesJet_SDsubjet0_deltaPhi", "delphesJet_SDsubjet0_deltaR", "delphesJet_SDsubjet0_mass", "delphesJet_SDsubjet1_deltaEta", "delphesJet_SDsubjet1_deltaPhi", "delphesJet_SDsubjet1_deltaR", "delphesJet_SDsubjet1_mass", "delphesJet_tau1", "delphesJet_tau2", "delphesJet_tau3", "delphesJet_tau4", "delphesJet_tau21", "delphesJet_tau32", 

    "delphesJet_ecf1", "delphesJet_ecf2", "delphesJet_ecf3", "delphesJet_ecfC1", "delphesJet_ecfC2", "delphesJet_ecfC3", "delphesJet_ecfD", "delphesJet_ecfDbeta2", "delphesJet_ecfM1", "delphesJet_ecfM2", "delphesJet_ecfM3", "delphesJet_ecfM1beta2", "delphesJet_ecfM2beta2", "delphesJet_ecfM3beta2", "delphesJet_ecfN1", "delphesJet_ecfN2", "delphesJet_ecfN3", "delphesJet_ecfN1beta2", "delphesJet_ecfN2beta2", "delphesJet_ecfN3beta2", "delphesJet_ecfU1", "delphesJet_ecfU2", "delphesJet_ecfU3", "delphesJet_ecfU1beta2", "delphesJet_ecfU2beta2", "delphesJet_ecfU3beta2", 

    "parton_hadTop_decayAngle_theta", "parton_hadTop_decayAngle_phi", "parton_hadTop_q1_pt", "parton_hadTop_q1_eta", "parton_hadTop_q2_pt", "parton_hadTop_q2_eta", "parton_hadTop_b_pt", "parton_hadTop_b_eta", "parton_hadTop_W_pt", "parton_hadTop_W_eta", 

    "parton_cosThetaPlus_n", "parton_cosThetaMinus_n", "parton_cosThetaPlus_r", "parton_cosThetaMinus_r", "parton_cosThetaPlus_k", "parton_cosThetaMinus_k", "parton_cosThetaPlus_r_star", "parton_cosThetaMinus_r_star", "parton_cosThetaPlus_k_star", "parton_cosThetaMinus_k_star", "parton_xi_nn",

    "parton_xi_rr", "parton_xi_kk", "parton_xi_nr_plus", "parton_xi_nr_minus", "parton_xi_rk_plus", "parton_xi_rk_minus", "parton_xi_nk_plus", "parton_xi_nk_minus", "parton_cos_phi", "parton_cos_phi_lab", "parton_abs_delta_phi_ll_lab",

    "delphesJet_lep_cosTheta_n", "delphesJet_lep_cosTheta_r", "delphesJet_lep_cosTheta_k", 
                    ]

data_generator = DataGenerator(
            input_files = [os.path.join( user.data_directory, "v11/TT01j_HT800_ext_comb/TT01j_HT800_ext_comb_1*.root")],
                n_split = 1,
                splitting_strategy = "files",
                selection   = selection,
                branches = [
                    #"genJet_pt", "dR_genJet_maxq1q2b", "genJet_SDmass", "genJet_mass", "genJet_nConstituents", "genJet_SDsubjet1_mass", 
                    "neflow", "eflow_pt", "eflow_etarel", "eflow_phirel", "eflow_charge", 
                    "eflow_cosTheta_n", "eflow_cosTheta_r", "eflow_cosTheta_k", "eflow_cosTheta_r_star", "eflow_cosTheta_k_star", "p_C",
                    "ctGRe_coeff"
                 ] + branches
            )

class delphesTTbarModel:
    def __init__( self, min_pt = 0, padding=100, small=False, features = [], scalar_features = [], 
            truth_interval = None,
            train_with_truth = False,
            ):
        self.scalar_features= scalar_features
        self.features       = features
        self.padding        = padding
        self.min_pt         = min_pt
        self.truth_interval = truth_interval
        self.data_generator = data_generator 
        self.data_generator.branches += features 

        if scalar_features is not None and len(scalar_features)>0:
            self.scalar_features = scalar_features
            if not type(self.scalar_features)==type([]): raise RuntimeError ("Need a list of scalar features")
            for feature in scalar_features:
                if feature not in branches:
                    branches.append( feature)
        else:
            self.scalar_features = None

        self.train_with_truth=train_with_truth

    def set_truth_mask( self, truth):
        self.mask = torch.ones(len(truth),dtype=bool).to(device)
        if self.truth_interval is not None:
            if self.truth_interval[0] is not None:
                self.mask*=(truth>self.truth_interval[0])
            if self.truth_interval[1] is not None:
                self.mask*=(truth<=self.truth_interval[1])

    #@staticmethod
    def getEvents(self, data):
        ctgRe_coeff = DataGenerator.vector_branch( data, 'ctGRe_coeff', padding_target=3 )
        weights = ctgRe_coeff[:, 0]
        truth = torch.Tensor(ctgRe_coeff[:, 1]).to(device)
        self.set_truth_mask(truth)
        #truth = -0.15*torch.ones_like(truth)

        pts = DataGenerator.vector_branch( data, 'eflow_pt',padding_target=self.padding )
        ptmask =  torch.Tensor((pts >= self.min_pt)).to(device) #  (pts > 5)
        ctn = DataGenerator.vector_branch( data, 'eflow_cosTheta_n',padding_target=self.padding ) # [ptmask]
        ctr = DataGenerator.vector_branch( data, 'eflow_cosTheta_r',padding_target=self.padding ) # [ptmask
        pts  = torch.Tensor(pts).to(device) * ptmask  
        ctn  = torch.Tensor(ctn).to(device) * ptmask  
        ctr  = torch.Tensor(ctr).to(device) * ptmask  
        #return pts, torch.stack( (dphis, detas), axis=-1), torch.tensor(weights).to(device), torch.tensor(truth).to(device)
        
        features = None
        for feature in self.features: 
            branch_data = DataGenerator.vector_branch( data, feature, padding_target=self.padding ) 
            if features is None:
                features = (torch.Tensor(branch_data).to(device) * ptmask).view(-1,self.padding,1)
            else:
                features = torch.cat( (features, (torch.Tensor(branch_data).to(device)* ptmask).view(-1,self.padding,1)), dim=2)

        if self.scalar_features:
            scalar_features = torch.Tensor(DataGenerator.scalar_branches(data, self.scalar_features)).to(device)
        else:
            scalar_features = None

        # append truth to scalar features
        if self.train_with_truth:
            if scalar_features is None:
                scalar_features = truth.view(-1,1)
            else: 
                scalar_features = torch.cat( (scalar_features, truth.view(-1,1)), dim=1)

        return  pts[self.mask],\
                torch.stack( (ctn, ctr), axis=-1)[self.mask],\
                features[self.mask] if features is not None else None,\
                scalar_features,\
                torch.Tensor( weights ).to(device)[self.mask],\
                truth[self.mask]

    def getWeightDict( self, data ):
        ctgRe_coeff = DataGenerator.vector_branch( data, 'ctGRe_coeff',padding_target=3 )
        truth = torch.Tensor(ctgRe_coeff[:, 1]).to(device)
        self.set_truth_mask(truth)
        combinations = make_combinations( wilson_coefficients )
        coeffs = data_generator.vector_branch(data, 'p_C')
        return {comb:coeffs[:,weightInfo.combinations.index(comb)][self.mask] for comb in combinations}

    def getScalarFeatures( self, data, branches = None):
        ctgRe_coeff = DataGenerator.vector_branch( data, 'ctGRe_coeff',padding_target=3 )
        truth = torch.Tensor(ctgRe_coeff[:, 1]).to(device)
        self.set_truth_mask(truth)
        return DataGenerator.scalar_branches( data, self.scalar_features if branches is None else branches)[self.mask]

tex = {"ctGRe":"C_{tG}^{Re}", "ctGIm":"C_{tG}^{Im}", "ctBIm":"C_{tB}^{Im}", "ctBRe":"C_{tB}^{Re}", "cHt":"C_{Ht}", 'cHtbRe':'C_{Htb}^{Re}', 'cHtbIm':'C_{Htb}^{Im}', 'cHQ3':'C_{HQ}^{(3)}'}

#['ctGRe', 'ctBRe', 'cHQ3', 'cHt', 'cHtbRe', 'ctGIm', 'ctBIm', 'cHtbIm']
eft_plot_points = [
    {'color':ROOT.kBlack,       'eft':sm, 'tex':"SM"},
    {'color':ROOT.kMagenta-4,   'eft':make_eft(ctGRe=.3),   'tex':"c_{tG}^{Re}=.3",},
    {'color':ROOT.kMagenta+2,   'eft':make_eft(ctGRe=-.3),   'tex':"c_{tG}^{Re}=-.3",},
    {'color':ROOT.kGreen-4,     'eft':make_eft(ctGRe=.5),   'tex':"c_{tG}^{Re}=.5",},
    {'color':ROOT.kGreen+2,     'eft':make_eft(ctGRe=-.5),   'tex':"c_{tG}^{Re}=-.5",},
    {'color':ROOT.kBlue+2,      'eft':make_eft(ctGRe=1.),   'tex':"c_{tG}^{Re}=1.",},
    {'color':ROOT.kBlue-4,      'eft':make_eft(ctGRe=-1.),   'tex':"c_{tG}^{Re}=-1.",},
    ]

plot_options =  {
    "parton_hadTop_pt" :{'binning':[50,0,1500], 'tex':'p_{T}(t)'},
    "parton_hadTop_eta" :{'binning':[30,-3,3], 'tex':'#eta(t)'},
    "parton_hadTop_phi" :{'binning':[30,-pi,pi], 'tex':'#phi(t)'},
    "parton_lepTop_pt" :{'binning':[30,0,800], 'tex':'p_{T}(t lep)'},
    "parton_lepTop_eta" :{'binning':[30,-3,3], 'tex':'#eta(t lep)'},
    "parton_lepTop_phi" :{'binning':[30,-pi,pi], 'tex':'#phi(t lep)'},
    "parton_lepTop_lep_pt" :{'binning':[30,0,800], 'tex':'p_{T}(l (t lep))'},
    "parton_lepTop_lep_eta" :{'binning':[30,-3,3], 'tex':'#eta(l(t lep))'},
    "parton_lepTop_lep_phi" :{'binning':[30,-pi,pi], 'tex':'#phi(l(t lep))'},
    "parton_lepTop_nu_pt" :{'binning':[30,0,800], 'tex':'p_{T}(#nu (t lep))'},
    "parton_lepTop_nu_eta" :{'binning':[30,-3,3], 'tex':'#eta(#nu(t lep))'},
    "parton_lepTop_nu_phi" :{'binning':[30,-pi,pi], 'tex':'#phi(#nu(t lep))'},
    "parton_lepTop_b_pt" :{'binning':[50,0,800], 'tex':'p_{T}(b (t lep))'},
    "parton_lepTop_b_eta" :{'binning':[30,-3,3], 'tex':'#eta(b(t lep))'},
    "parton_lepTop_b_phi" :{'binning':[30,-pi,pi], 'tex':'#phi(b(t lep))'},
    "parton_lepTop_W_pt" :{'binning':[30,0,1000], 'tex':'p_{T}(W (t lep))'},
    "parton_lepTop_W_eta" :{'binning':[30,-3,3], 'tex':'#eta(W(t lep))'},
    "parton_lepTop_W_phi" :{'binning':[30,-pi,pi], 'tex':'#phi(W(t lep))'},
    "delphesJet_lep_cosTheta_n":{'binning':[30,-1,1], 'tex':'lep cos #theta_{n}'},
    "delphesJet_lep_cosTheta_r":{'binning':[30,-1,1], 'tex':'lep cos #theta_{r}'},
    "delphesJet_lep_cosTheta_k":{'binning':[30,-1,1], 'tex':'lep cos #theta_{k}'},
    "delphesJet_pt"                 :{'binning':[50,500,2000], 'tex':'p_{T}(jet)'},
    "delphesJet_eta"                :{'binning':[30,-3,3], 'tex':'#eta(jet)'},
    "delphesJet_phi"                :{'binning':[30,-pi,pi], 'tex':'#phi(jet)'},
    "delphesJet_nConstituents"      :{'binning':[30,30,230], 'tex':'n-constituents'},
    "delphesJet_SDmass"             :{'binning':[30,150,200], 'tex':'M_{SD}(jet)'},
    "delphesJet_SDsubjet0_deltaEta" :{'binning':[30,-0.6,0.6], 'tex':'#Delta#eta(jet,jet_{SD,0})'},
    "delphesJet_SDsubjet0_deltaPhi" :{'binning':[30,-0.6,0.6], 'tex':'#Delta#phi(jet,jet_{SD,0})'},
    "delphesJet_SDsubjet0_deltaR"   :{'binning':[30,0,0.7], 'tex':'#Delta R(jet,jet_{SD,0})'},
    "delphesJet_SDsubjet0_mass"     :{'binning':[30,0,200], 'tex':'M_{SD}(jet_{0})'},
    "delphesJet_SDsubjet1_deltaEta" :{'binning':[30,-0.6,0.6], 'tex':'#Delta#eta(jet,jet_{SD,1})'},
    "delphesJet_SDsubjet1_deltaPhi" :{'binning':[30,-0.6,0.6], 'tex':'#Delta#phi(jet,jet_{SD,1})'},
    "delphesJet_SDsubjet1_deltaR"   :{'binning':[30,0,0.7], 'tex':'#Delta R(jet,jet_{SD,1})'},
    "delphesJet_SDsubjet1_mass"     :{'binning':[30,0,200], 'tex':'M_{SD}(jet_{1})'},
    "delphesJet_tau1"               :{'binning':[30,0,1], 'tex':'#tau_{1}'},
    "delphesJet_tau2"               :{'binning':[30,0,.5],'tex':'#tau_{2}'},
    "delphesJet_tau3"               :{'binning':[30,0,.3],'tex':'#tau_{3}'},
    "delphesJet_tau4"               :{'binning':[30,0,.3],'tex':'#tau_{4}'},
    "delphesJet_tau21"              :{'binning':[30,0,1], 'tex':'#tau_{21}'},
    "delphesJet_tau32"              :{'binning':[30,0,1], 'tex':'#tau_{32}'},
#https://github.com/cms-sw/cmssw/blob/6d2f66057131baacc2fcbdd203588c41c885b42c/RecoJets/JetProducers/python/ECF_cff.py
    "delphesJet_ecf1"               :{'binning':[30,0,2000], 'tex':"ecf1"},
    "delphesJet_ecf2"               :{'binning':[30,0,400000], 'tex':"ecf2"},
    "delphesJet_ecf3"               :{'binning':[30,0,4000000], 'tex':"ecf3"},
    "delphesJet_ecfC1"              :{'binning':[30,0,.5], 'tex':"ecfC1"},
    "delphesJet_ecfC2"              :{'binning':[30,0,.5], 'tex':"ecfC2"},
    "delphesJet_ecfC3"              :{'binning':[30,0,.5], 'tex':"ecfC3"},
    "delphesJet_ecfD"               :{'binning':[30,0,8], 'tex':"ecfD"},
    "delphesJet_ecfDbeta2"          :{'binning':[30,0,20], 'tex':"ecfDbeta2"},
    "delphesJet_ecfM1"              :{'binning':[30,0,0.35], 'tex':"ecfM1"},
    "delphesJet_ecfM2"              :{'binning':[30,0,0.2], 'tex':"ecfM2"},
    "delphesJet_ecfM3"              :{'binning':[30,0,0.2], 'tex':"ecfM3"},
    "delphesJet_ecfM1beta2"         :{'binning':[30,0,0.35], 'tex':"ecfM1beta2"},
    "delphesJet_ecfM2beta2"         :{'binning':[30,0,0.2], 'tex':"ecfM2beta2"},
    "delphesJet_ecfM3beta2"         :{'binning':[30,0,0.2], 'tex':"ecfM3beta2"},
    "delphesJet_ecfN1"              :{'binning':[30,0,0.5], 'tex':"ecfN1"},
    "delphesJet_ecfN2"              :{'binning':[30,0,0.5], 'tex':"ecfN2"},
    "delphesJet_ecfN3"              :{'binning':[30,0,5], 'tex':"ecfN3"},
    "delphesJet_ecfN1beta2"         :{'binning':[30,0,0.5], 'tex':"ecfN1beta2"},
    "delphesJet_ecfN2beta2"         :{'binning':[30,0,0.5], 'tex':"ecfN2beta2"},
    "delphesJet_ecfN3beta2"         :{'binning':[30,0,5], 'tex':"ecfN3beta2"},
    "delphesJet_ecfU1"              :{'binning':[30,0,0.5], 'tex':"ecfU1"},
    "delphesJet_ecfU2"              :{'binning':[30,0,0.04], 'tex':"ecfU2"},
    "delphesJet_ecfU3"              :{'binning':[30,0,0.004], 'tex':"ecfU3"},
    "delphesJet_ecfU1beta2"         :{'binning':[30,0,0.5], 'tex':"ecfU1beta2"},
    "delphesJet_ecfU2beta2"         :{'binning':[30,0,0.04], 'tex':"ecfU2beta2"},
    "delphesJet_ecfU3beta2"         :{'binning':[30,0,0.004], 'tex':"ecfU3beta2"},

    "parton_hadTop_decayAngle_theta" :{'binning':[30,0,pi], 'tex':'#theta(t had)'},
    "parton_hadTop_decayAngle_phi"   :{'binning':[30,-pi,pi], 'tex':'#phi(t had)'},

    "parton_hadTop_q1_pt" :{'binning':[30,0,800], 'tex':'p_{T}(q_{1}(t had))'},
    "parton_hadTop_q1_eta" :{'binning':[30,-3,3], 'tex':'#eta(q_{1}(t had))'},
    "parton_hadTop_q2_pt" :{'binning':[30,0,800], 'tex':'p_{T}(q_{2}(t had))'},
    "parton_hadTop_q2_eta" :{'binning':[30,-3,3], 'tex':'#eta(q_{2}(t had))'},
    "parton_hadTop_b_pt" :{'binning':[30,0,800], 'tex':'p_{T}(b(t had))'},
    "parton_hadTop_b_eta" :{'binning':[30,-3,3], 'tex':'#eta(b(t had))'},
    "parton_hadTop_W_pt" :{'binning':[30,0,800], 'tex':'p_{T}(W(t had))'},
    "parton_hadTop_W_eta" :{'binning':[30,-3,3], 'tex':'#eta(W(t had))'},

    "parton_cosThetaPlus_n"     :{'binning':[30,-1,1], 'tex':'cos#theta^{+}_{n}'},
    "parton_cosThetaMinus_n"    :{'binning':[30,-1,1], 'tex':'cos#theta^{-}_{n}'},
    "parton_cosThetaPlus_r"     :{'binning':[30,-1,1], 'tex':'cos#theta^{+}_{r}'},
    "parton_cosThetaMinus_r"    :{'binning':[30,-1,1], 'tex':'cos#theta^{-}_{r}'},
    "parton_cosThetaPlus_k"     :{'binning':[30,-1,1], 'tex':'cos#theta^{+}_{k}'},
    "parton_cosThetaMinus_k"    :{'binning':[30,-1,1], 'tex':'cos#theta^{-}_{k}'},
    "parton_cosThetaPlus_r_star"    :{'binning':[30,-1,1], 'tex':'cos#theta^{+*}_{n}'},
    "parton_cosThetaMinus_r_star"   :{'binning':[30,-1,1], 'tex':'cos#theta^{-*}_{n}'},
    "parton_cosThetaPlus_k_star"    :{'binning':[30,-1,1], 'tex':'cos#theta^{+*}_{k}'},
    "parton_cosThetaMinus_k_star"   :{'binning':[30,-1,1], 'tex':'cos#theta^{-*}_{k}'},
    "parton_xi_nn"              :{'binning':[30,-1,1], 'tex':'#xi_{nn}'},
    "parton_xi_rr"              :{'binning':[30,-1,1], 'tex':'#xi_{rr}'},
    "parton_xi_kk"              :{'binning':[30,-1,1], 'tex':'#xi_{kk}'},
    "parton_xi_nr_plus"         :{'binning':[30,-1,1], 'tex':'#xi_{nr}^{+}'},
    "parton_xi_nr_minus"        :{'binning':[30,-1,1], 'tex':'#xi_{nr}^{-}'},
    "parton_xi_rk_plus"         :{'binning':[30,-1,1], 'tex':'#xi_{rk}^{+}'},
    "parton_xi_rk_minus"        :{'binning':[30,-1,1], 'tex':'#xi_{rk}^{-}'},
    "parton_xi_nk_plus"         :{'binning':[30,-1,1], 'tex':'#xi_{nk}^{+}'},
    "parton_xi_nk_minus"        :{'binning':[30,-1,1], 'tex':'#xi_{nk}^{-}'},
    "parton_cos_phi"            :{'binning':[30,-1,1], 'tex':'cos(#phi)'},
    "parton_cos_phi_lab"        :{'binning':[30,-1,1], 'tex':'cos(#phi lab)'},
    "parton_abs_delta_phi_ll_lab":{'binning':[30,0,pi], 'tex':'|#Delta(#phi(l,l))|'},
}

feature_names   = list(plot_options.keys())

