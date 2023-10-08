import pickle
import random
import ROOT
import os
import numpy as np 
import math

if __name__=="__main__":
    import sys
    #sys.path.append('/work/sesanche/SMEFTNet')
    sys.path.append('..')
    sys.path.append('../..')

from tools.DataGenerator import DataGenerator
from tools.WeightInfo    import WeightInfo
import torch
import tools.user as user
import functools 
import operator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sig_selection = lambda ar: (ar.genJet_pt>500) & (ar.dR_genJet_maxq1q2 < 0.6) & (ar.genJet_SDmass > 70) & (ar.genJet_SDmass < 110)
bkg_selection = lambda ar: (ar.genJet_pt>500) & (ar.genJet_SDmass > 70) & (ar.genJet_SDmass < 110)

reweight_pkl = os.path.join( user.pkl_directory, 'WZto2L_HT300_reweight_card.pkl' )
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

def angle( x, y):
    return torch.arctan2( y, x)
def dphi(phi1,phi2):
    dph=phi1-phi2
    return dph + 2*np.pi*(dph < -np.pi) - 2*np.pi*(dph > np.pi)


plot_options =  {

    "parton_hadV_pt":                {'binning':[50,500,2000], 'tex':'p_{T}(had V)'},
    "parton_hadV_eta":               {'binning':[50,-3,3], 'tex':'#eta(had V)'},
    "parton_hadV_mass":              {'binning':[50,50,150], 'tex':'M(had V)'},

    "parton_hadV_q1_pt":             {'binning':[50,500,2000], 'tex':'p_{T}(q_{1})'},
    "parton_hadV_q1_eta":            {'binning':[50,-5,5], 'tex':'#eta(q_{1})'},
    "parton_hadV_q1_mass":           {'binning':[50,0,10], 'tex':'M(q_{1})'},
    "parton_hadV_q1_pdgId":          {'binning':[12,-6,6], 'tex':'pdgId(q_{1})'},

    "parton_hadV_q2_pt":             {'binning':[50,500,2000], 'tex':'p_{T}(q_{2})'},
    "parton_hadV_q2_eta":            {'binning':[50,-5,5], 'tex':'#eta(q_{2})'},
    "parton_hadV_q2_mass":           {'binning':[50,0,10], 'tex':'M(q_{2})'},
    "parton_hadV_q2_pdgId":          {'binning':[12,-6,6], 'tex':'pdgId(q_{2})'},

    "parton_lepV_pt":                {'binning':[50,500,2000], 'tex':'p_{T}(lep V)'},
    "parton_lepV_eta":               {'binning':[50,-5,5], 'tex':'#eta(lep V)'},
    "parton_lepV_mass":              {'binning':[50,50,150], 'tex':'M(lep V)'},

    "parton_lepV_l1_pt":            {'binning':[50,500,2000], 'tex':'p_{T}(l_{1})'},
    "parton_lepV_l1_eta":           {'binning':[50,-5,5], 'tex':'#eta(l_{1})'},
   # "parton_lepV_l1_phi":           {'binning':[50,0,10], 'tex':'M(l_{1})'},
    "parton_lepV_l1_pdgId":         {'binning':[32,-16,16], 'tex':'pdgId(l_{1})'},

    "parton_lepV_l2_pt":            {'binning':[50,500,2000], 'tex':'p_{T}(l_{2})'},
    "parton_lepV_l2_eta":           {'binning':[50,-5,5], 'tex':'#eta(l_{2})'},
    #"parton_lepV_l2_phi":           {'binning':[50,0,10], 'tex':'M(l_{2})'},
    "parton_lepV_l2_pdgId":         {'binning':[32,-16,16], 'tex':'pdgId(l_{2})'},

    "parton_hadV_angle_theta":      {'binning':[50,0,math.pi], 'tex':'#theta'},
    "parton_hadV_angle_Theta":      {'binning':[50,0,math.pi], 'tex':'#Theta'},
    "parton_hadV_angle_phi":        {'binning':[50,-math.pi,math.pi], 'tex':'#phi'},
    #"parton_hadV_angle_absPhi":     {'binning':[50,0,math.pi], 'tex':'abs(#phi)'},

    "parton_WZ_mass":               {'binning':[50,0,1000], 'tex':'M(WZ)'},
    "parton_WZ_deltaPhi":           {'binning':[50,0,math.pi], 'tex':'#Delta#phi(WZ)'},
    "parton_WZ_pt":                 {'binning':[50,0,500], 'tex':'p_{T}(WZ)'},
    "parton_WZ_p":                  {'binning':[50,0,500], 'tex':'p(WZ)'},
    "parton_WZ_eta":                {'binning':[50,-4,4], 'tex':'#eta(WZ)'},
    "parton_WZ_phi":                {'binning':[50,-math.pi,math.pi], 'tex':'#phi(WZ)'},

    "genJet_pt"                     :{'binning':[50,0,2000], 'tex':'p_{T}(gen-jet)'},
    #"genJet_p"                      :{'binning':[50,0,2000], 'tex':'p(gen-jet)'},
    "genJet_eta"                    :{'binning':[50,-5,5],   'tex':'#eta(gen-jet)'},
    "genJet_y"                      :{'binning':[50,-5,5],   'tex':'y(gen-jet)'},
    "genJet_mass"                   :{'binning':[50,50,120], 'tex':'M(gen-jet)'},
    "genJet_nConstituents"          :{'binning':[50,0,100],  'tex':'gen-jet n-constituents'},
    "genJet_VV_y"                   :{'binning':[50,-5,5],   'tex':'y(gen-jet) VV'},
    "genJet_VV_p"                   :{'binning':[50,0,2000], 'tex':'p(gen-jet) VV'},
    "genJet_q1_VV_p"                :{'binning':[50,0,2000], 'tex':'p(q1) gen-jet VV '},
    "genJet_q1_VV_Dy"               :{'binning':[50,-2,2], 'tex':'#Delta y(q1, gen-jet) VV'},
    "genJet_q1_VV_Theta"            :{'binning':[50,0,math.pi], 'tex':'#Theta(q1, gen-jet) VV'},
    "genJet_q1_VV_Phi"              :{'binning':[50,-math.pi, math.pi], 'tex':'q1(#phi) gen-jet VV'},
    "genJet_q2_VV_p"                :{'binning':[50,0,2000], 'tex':'p(q2) gen-jet VV '},
    "genJet_q2_VV_Dy"               :{'binning':[50,-5,5], 'tex':'#Delta y(q2, gen-jet) VV'},
    "genJet_q2_VV_Theta"            :{'binning':[50,0,math.pi], 'tex':'#Theta(q2, gen-jet) VV'},
    "genJet_q2_VV_Phi"              :{'binning':[50,-math.pi,math.pi], 'tex':'q2(#phi) gen-jet VV'},

    "dR_genJet_q1"                  :{'binning':[50,0,5], 'tex':'#Delta R(q1, gen-jet)'},
    "dR_genJet_q2"                  :{'binning':[50,0,5], 'tex':'#Delta R(q2, gen-jet)'},
    "dR_genJet_maxq1q2"             :{'binning':[50,0,5], 'tex':'max #Delta R(q1/2, gen-jet)'},
    "gen_beam_VV_phi"               :{'binning':[50,-math.pi,math.pi], 'tex':'#phi beam VV'},
    "gen_beam_VV_theta"             :{'binning':[50,0,math.pi], 'tex':'#theta beam VV'},
#    "gen_beam_VV_Dy"                :{'binning':[50,-5,5], 'tex':'#Delta y(beam, gen-jet) VV'},


    "genJet_SDmass"             :{'binning':[50,50,120], 'tex':'M_{SD}(jet)'},
    "genJet_SDsubjet0_deltaEta" :{'binning':[50,-0.6,0.6], 'tex':'#Delta#eta(gen-jet,jet_{SD,0})'},
    "genJet_SDsubjet0_deltaPhi" :{'binning':[50,-0.6,0.6], 'tex':'#Delta#phi(gen-jet,jet_{SD,0})'},
    "genJet_SDsubjet0_deltaR"   :{'binning':[50,0,0.7], 'tex':'#Delta R(gen-jet,jet_{SD,0})'},
    "genJet_SDsubjet0_mass"     :{'binning':[50,0,100], 'tex':'M_{SD}(gen-jet_{0})'},
    "genJet_SDsubjet1_deltaEta" :{'binning':[50,-0.6,0.6], 'tex':'#Delta#eta(gen-jet,jet_{SD,1})'},
    "genJet_SDsubjet1_deltaPhi" :{'binning':[50,-0.6,0.6], 'tex':'#Delta#phi(gen-jet,jet_{SD,1})'},
    "genJet_SDsubjet1_deltaR"   :{'binning':[50,0,0.7], 'tex':'#Delta R(gen-jet,jet_{SD,1})'},
    "genJet_SDsubjet1_mass"     :{'binning':[50,0,100], 'tex':'M_{SD}(gen-jet_{1})'},
    "genJet_tau1"               :{'binning':[50,0,.5], 'tex':'gen-jet #tau_{1}'},
    "genJet_tau2"               :{'binning':[50,0,.15],'tex':'gen-jet #tau_{2}'},
    "genJet_tau3"               :{'binning':[50,0,.15],'tex':'gen-jet #tau_{3}'},
    "genJet_tau4"               :{'binning':[50,0,.15],'tex':'gen-jet #tau_{4}'},
    "genJet_tau21"              :{'binning':[50,0,1], 'tex':'gen-jet #tau_{21}'},
    "genJet_tau32"              :{'binning':[50,0,1], 'tex':'gen-jet #tau_{32}'},
#https://github.com/cms-sw/cmssw/blob/6d2f66057131baacc2fcbdd203588c41c885b42c/RecoJets/JetProducers/python/ECF_cff.py
    "genJet_ecf1"               :{'binning':[50,0,2000], 'tex':"gen-jet ecf1"},
    "genJet_ecf2"               :{'binning':[50,0,100000], 'tex':"gen-jet ecf2"},
    "genJet_ecf3"               :{'binning':[50,0,1000000], 'tex':"gen-jet ecf3"},
    "genJet_ecfC1"              :{'binning':[50,0,.5], 'tex':"gen-jet ecfC1"},
    "genJet_ecfC2"              :{'binning':[50,0,.5], 'tex':"gen-jet ecfC2"},
    "genJet_ecfC3"              :{'binning':[50,0,.5], 'tex':"gen-jet ecfC3"},
    "genJet_ecfD"               :{'binning':[50,0,8], 'tex':"gen-jet ecfD"},
    "genJet_ecfDbeta2"          :{'binning':[50,0,20], 'tex':"gen-jet ecfDbeta2"},
    "genJet_ecfM1"              :{'binning':[50,0,0.35], 'tex':"gen-jet ecfM1"},
    "genJet_ecfM2"              :{'binning':[50,0,0.2], 'tex':"gen-jet ecfM2"},
    "genJet_ecfM3"              :{'binning':[50,0,0.2], 'tex':"gen-jet ecfM3"},
    "genJet_ecfM1beta2"         :{'binning':[50,0,0.35], 'tex':"gen-jet ecfM1beta2"},
    "genJet_ecfM2beta2"         :{'binning':[50,0,0.2], 'tex':"gen-jet ecfM2beta2"},
    "genJet_ecfM3beta2"         :{'binning':[50,0,0.2], 'tex':"gen-jet ecfM3beta2"},
    "genJet_ecfN1"              :{'binning':[50,0,0.2], 'tex':"gen-jet ecfN1"},
    "genJet_ecfN2"              :{'binning':[50,0,0.5], 'tex':"gen-jet ecfN2"},
    "genJet_ecfN3"              :{'binning':[50,0,5], 'tex':"gen-jet ecfN3"},
    "genJet_ecfN1beta2"         :{'binning':[50,0,0.1], 'tex':"gen-jet ecfN1beta2"},
    "genJet_ecfN2beta2"         :{'binning':[50,0,0.5], 'tex':"gen-jet ecfN2beta2"},
    "genJet_ecfN3beta2"         :{'binning':[50,0,5], 'tex':"gen-jet ecfN3beta2"},
    "genJet_ecfU1"              :{'binning':[50,0,0.5], 'tex':"gen-jet ecfU1"},
    "genJet_ecfU2"              :{'binning':[50,0,0.04], 'tex':"gen-jet ecfU2"},
    "genJet_ecfU3"              :{'binning':[50,0,0.004], 'tex':"gen-jet ecfU3"},
    "genJet_ecfU1beta2"         :{'binning':[50,0,0.2], 'tex':"gen-jet ecfU1beta2"},
    "genJet_ecfU2beta2"         :{'binning':[50,0,0.01], 'tex':"gen-jet ecfU2beta2"},
    "genJet_ecfU3beta2"         :{'binning':[50,0,0.001], 'tex':"gen-jet ecfU3beta2"},

    "delphesJet_pt"                 :{'binning':[50,0,2000], 'tex':'p_{T}(jet)'},
    "delphesJet_eta"                :{'binning':[50,-5,5], 'tex':'#eta(jet)'},
    "delphesJet_y"                  :{'binning':[50,-5,5],   'tex':'y(jet)'},
    "delphesJet_mass"               :{'binning':[50,50,120], 'tex':'M(jet) unpruned'},
    "delphesJet_nConstituents"      :{'binning':[50,0,100], 'tex':'n-constituents'},

    "delphesJet_VV_y"                   :{'binning':[50,-5,5],   'tex':'y(jet) VV'},
    "delphesJet_VV_p"                   :{'binning':[50,0,2000], 'tex':'p(jet) VV'},
    "delphesJet_q1_VV_p"                :{'binning':[50,0,2000], 'tex':'p(q1) jet VV '},
    "delphesJet_q1_VV_Dy"               :{'binning':[50,-6,6], 'tex':'#Delta y(q1, jet) VV'},
    "delphesJet_q1_VV_Theta"            :{'binning':[50,0,math.pi], 'tex':'#Theta(q1, jet) VV'},
    "delphesJet_q1_VV_Phi"              :{'binning':[50,-math.pi, math.pi], 'tex':'q1(#phi) jet VV'},
    "delphesJet_q2_VV_p"                :{'binning':[50,0,2000], 'tex':'p(q2) jet VV '},
    "delphesJet_q2_VV_Dy"               :{'binning':[50,-5,5], 'tex':'#Delta y(q2, jet) VV'},
    "delphesJet_q2_VV_Theta"            :{'binning':[50,0,math.pi], 'tex':'#Theta(q2, jet) VV'},
    "delphesJet_q2_VV_Phi"              :{'binning':[50,-math.pi,math.pi], 'tex':'q2(#phi) jet VV'},

    "delphes_beam_VV_phi"               :{'binning':[50,-math.pi,math.pi], 'tex':'#phi beam VV'},
    "delphes_beam_VV_theta"             :{'binning':[50,0,math.pi], 'tex':'#theta beam VV'},
    "delphes_beam_VV_Dy"                :{'binning':[50,-5,5], 'tex':'#Delta y(beam, gen-jet) VV'},

    "delphesJet_dR_matched_hadV_parton": {'binning':[50,0,.5], 'tex':'#Delta R(jet, had V)'},
    "delphesJet_dR_lepV_parton":         {'binning':[50,0,6], 'tex':'#Delta R(jet, lep V)'},
    "delphesJet_dR_hadV_q1":             {'binning':[50,0,1.5], 'tex':'#Delta R(jet, q_{1})'},
    "delphesJet_dR_hadV_q2":             {'binning':[50,0,1.5], 'tex':'#Delta R(jet, q_{2})'},
    "delphesJet_dR_hadV_maxq1q2":        {'binning':[50,0,1.5], 'tex':'max #Delta R(jet, q_{1,2})'},

    "delphesJet_SDmass"             :{'binning':[50,50,120], 'tex':'M_{SD}(jet)'},
    "delphesJet_SDsubjet0_deltaEta" :{'binning':[50,-0.6,0.6], 'tex':'#Delta#eta(jet,jet_{SD,0})'},
    "delphesJet_SDsubjet0_deltaPhi" :{'binning':[50,-0.6,0.6], 'tex':'#Delta#phi(jet,jet_{SD,0})'},
    "delphesJet_SDsubjet0_deltaR"   :{'binning':[50,0,0.7], 'tex':'#Delta R(jet,jet_{SD,0})'},
    "delphesJet_SDsubjet0_mass"     :{'binning':[50,0,100], 'tex':'M_{SD}(jet_{0})'},
    "delphesJet_SDsubjet1_deltaEta" :{'binning':[50,-0.6,0.6], 'tex':'#Delta#eta(jet,jet_{SD,1})'},
    "delphesJet_SDsubjet1_deltaPhi" :{'binning':[50,-0.6,0.6], 'tex':'#Delta#phi(jet,jet_{SD,1})'},
    "delphesJet_SDsubjet1_deltaR"   :{'binning':[50,0,0.7], 'tex':'#Delta R(jet,jet_{SD,1})'},
    "delphesJet_SDsubjet1_mass"     :{'binning':[50,0,100], 'tex':'M_{SD}(jet_{1})'},
    "delphesJet_tau1"               :{'binning':[50,0,.5], 'tex':'#tau_{1}'},
    "delphesJet_tau2"               :{'binning':[50,0,.15],'tex':'#tau_{2}'},
    "delphesJet_tau3"               :{'binning':[50,0,.15],'tex':'#tau_{3}'},
    "delphesJet_tau4"               :{'binning':[50,0,.15],'tex':'#tau_{4}'},
    "delphesJet_tau21"              :{'binning':[50,0,1], 'tex':'#tau_{21}'},
    "delphesJet_tau32"              :{'binning':[50,0,1], 'tex':'#tau_{32}'},
#https://github.com/cms-sw/cmssw/blob/6d2f66057131baacc2fcbdd203588c41c885b42c/RecoJets/JetProducers/python/ECF_cff.py
    "delphesJet_ecf1"               :{'binning':[50,0,2000], 'tex':"ecf1"},
    "delphesJet_ecf2"               :{'binning':[50,0,100000], 'tex':"ecf2"},
    "delphesJet_ecf3"               :{'binning':[50,0,1000000], 'tex':"ecf3"},
    "delphesJet_ecfC1"              :{'binning':[50,0,.5], 'tex':"ecfC1"},
    "delphesJet_ecfC2"              :{'binning':[50,0,.5], 'tex':"ecfC2"},
    "delphesJet_ecfC3"              :{'binning':[50,0,.5], 'tex':"ecfC3"},
    "delphesJet_ecfD"               :{'binning':[50,0,8], 'tex':"ecfD"},
    "delphesJet_ecfDbeta2"          :{'binning':[50,0,20], 'tex':"ecfDbeta2"},
    "delphesJet_ecfM1"              :{'binning':[50,0,0.35], 'tex':"ecfM1"},
    "delphesJet_ecfM2"              :{'binning':[50,0,0.2], 'tex':"ecfM2"},
    "delphesJet_ecfM3"              :{'binning':[50,0,0.2], 'tex':"ecfM3"},
    "delphesJet_ecfM1beta2"         :{'binning':[50,0,0.35], 'tex':"ecfM1beta2"},
    "delphesJet_ecfM2beta2"         :{'binning':[50,0,0.2], 'tex':"ecfM2beta2"},
    "delphesJet_ecfM3beta2"         :{'binning':[50,0,0.2], 'tex':"ecfM3beta2"},
    "delphesJet_ecfN1"              :{'binning':[50,0,0.2], 'tex':"ecfN1"},
    "delphesJet_ecfN2"              :{'binning':[50,0,0.5], 'tex':"ecfN2"},
    "delphesJet_ecfN3"              :{'binning':[50,0,5], 'tex':"ecfN3"},
    "delphesJet_ecfN1beta2"         :{'binning':[50,0,0.1], 'tex':"ecfN1beta2"},
    "delphesJet_ecfN2beta2"         :{'binning':[50,0,0.5], 'tex':"ecfN2beta2"},
    "delphesJet_ecfN3beta2"         :{'binning':[50,0,5], 'tex':"ecfN3beta2"},
    "delphesJet_ecfU1"              :{'binning':[50,0,0.5], 'tex':"ecfU1"},
    "delphesJet_ecfU2"              :{'binning':[50,0,0.04], 'tex':"ecfU2"},
    "delphesJet_ecfU3"              :{'binning':[50,0,0.004], 'tex':"ecfU3"},
    "delphesJet_ecfU1beta2"         :{'binning':[50,0,0.2], 'tex':"ecfU1beta2"},
    "delphesJet_ecfU2beta2"         :{'binning':[50,0,0.01], 'tex':"ecfU2beta2"},
    "delphesJet_ecfU3beta2"         :{'binning':[50,0,0.001], 'tex':"ecfU3beta2"},
}


class WZandDYModel:
    def __init__( self, charged=False,  scalar_features = [], what='lab', operator='cW'):
        self.what = what
        self.operator = operator
        self.charged = charged
        if self.what == 'lab': 
            branches=[
                "genJet_pt", "genJet_SDmass",'dR_genJet_maxq1q2',
                "ngen", 'gen_pt_lab', "gen_Deta_lab", "gen_Dphi_lab",
                "parton_hadV_q2_phi", "parton_hadV_q2_eta",
                "parton_hadV_q1_phi", "parton_hadV_q1_eta",
            ]
        elif self.what == 'VV':
            branches=["genJet_pt", "genJet_SDmass",'dR_genJet_maxq1q2',
               "ngen", 'gen_pt_lab', 'gen_Theta_VV', 'gen_phi_VV',
               'parton_hadV_angle_phi',
               'p_C','parton_hadV_pt', 'parton_hadV_pt', 'parton_lepV_pt',
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

        self.signal_generator =  DataGenerator(
            input_files = [os.path.join( user.data_directory, "v6/WZto2L_HT300_Ref_comb/*.root" )],
            n_split             = 200,
            splitting_strategy  = "files",
            selection           = sig_selection,
            branches            = branches,
        )
        self.bkg_generator =  DataGenerator(
            input_files = [os.path.join( user.data_directory, "v6/DY_HT300/*.root" )],
            n_split             = 10,
            splitting_strategy  = "files",
            selection           = bkg_selection,
            branches            = branches,
        )

    def getEvents(self, data):
        padding = 40
        pts = DataGenerator.vector_branch( data, 'gen_pt_lab',padding_target=padding ) 
        ptmask = torch.ones_like( torch.Tensor(pts) ).to(device) #  (pts > 5)
        pts    = torch.Tensor(pts).to(device)   * ptmask  # 0-pad the pt < 5

        coeffs = DataGenerator.vector_branch(data, 'p_C', padding_target=len(weightInfo.combinations))

        weight_sm    = torch.Tensor(coeffs[:,0])
        kwargs_p={self.operator : 1}
        kwargs_m={self.operator : -1}
        weight_plus  = torch.Tensor(getWeights( make_eft( **kwargs_p ), coeffs))
        weight_minus = torch.Tensor(getWeights( make_eft( **kwargs_m ), coeffs))

        target = 0.5*(weight_plus - weight_minus)/weight_sm

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
            parton_hadV_pt = torch.Tensor(DataGenerator.scalar_branches(data, ['parton_hadV_pt'])).to(device)
            parton_lepV_pt = torch.Tensor(DataGenerator.scalar_branches(data, ['parton_lepV_pt'])).to(device)
            truth = torch.stack( [parton_hadV_q1_phi[:,0], parton_hadV_pt[:,0], parton_lepV_pt[:,0]], axis=1)
        mask = (pts.sum(axis=1) > 0)
        pts=pts[mask]
        angles=angles[mask]
        scalar_features=scalar_features[mask]
        truth=truth[mask]
        weight_sm=weight_sm[mask]
        target=target[mask]
        
        return pts, angles, features, scalar_features, torch.stack([weight_sm, target],axis=1), truth
            

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
     

