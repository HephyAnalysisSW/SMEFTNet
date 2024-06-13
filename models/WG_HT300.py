import pickle
import random
import ROOT
import math
import numpy as np
import functools
import operator
import os

if __name__=="__main__":
    import sys
    sys.path.append('..')
    sys.path.append('../..')

from tools.DataGenerator import DataGenerator
from tools.WeightInfo    import WeightInfo

selection = lambda ar: (ar.delphesJet_dR_hadV_maxq1q2<0.6) & (ar.delphesJet_dR_matched_hadV_parton<0.6) & (ar.parton_hadV_pdgId<0)

import tools.user as user

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

    "parton_G_pt":                {'binning':[50,500,2000], 'tex':'p_{T}(G)'},
    "parton_G_eta":               {'binning':[50,-5,5], 'tex':'#eta(G)'},

    "parton_hadV_angle_theta":      {'binning':[50,0,math.pi], 'tex':'#theta'},
    "parton_hadV_angle_Theta":      {'binning':[50,0,math.pi], 'tex':'#Theta'},
    "parton_hadV_angle_phi":        {'binning':[50,-math.pi,math.pi], 'tex':'#phi'},
    #"parton_hadV_angle_absPhi":     {'binning':[50,0,math.pi], 'tex':'abs(#phi)'},

    "parton_VG_mass":               {'binning':[50,0,1000], 'tex':'M(VG)'},
    "parton_VG_deltaPhi":           {'binning':[50,0,math.pi], 'tex':'#Delta#phi(VG)'},
    "parton_VG_pt":                 {'binning':[50,0,500], 'tex':'p_{T}(VG)'},
    "parton_VG_p":                  {'binning':[50,0,500], 'tex':'p(VG)'},
    "parton_VG_eta":                {'binning':[50,-4,4], 'tex':'#eta(VG)'},
    "parton_VG_phi":                {'binning':[50,-math.pi,math.pi], 'tex':'#phi(VG)'},

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
    "delphesJet_dR_G_parton":         {'binning':[50,0,6], 'tex':'#Delta R(jet, G)'},
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

data_generator = DataGenerator(
        input_files = [os.path.join( user.data_directory, "v6/WG_HT300/WG_HT300*.root")],
        n_split = -1,
        splitting_strategy = "files",
        selection   = selection,
        branches = list(plot_options.keys()) + ["p_C", "parton_hadV_pdgId"]
    )

reweight_pkl = '/eos/vbc/group/cms/robert.schoefbeck/gridpacks/ParticleNet/WG_HT300_reweight_card.pkl'
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

#if __name__=="__main__":
wilson_coefficients = weightInfo.variables 
#else:
#    wilson_coefficients = ['ctWRe']

vector_branches = [] #"gen_pt", "gen_etarel", "gen_phirel"]

feature_names = list(plot_options.keys())

def getEvents( nTraining ):
    data_generator._load(-1, small=nTraining )
    #combinations = make_combinations( wilson_coefficients )
    coeffs = data_generator.vector_branch('p_C')

    f = data_generator.scalar_branches( feature_names )
    #f = {key:f[:,i_key] for i_key, key in enumerate(feature_names)}
    v = {key:data_generator.vector_branch( key ) for key in vector_branches}
    #w = {comb:coeffs[:,weightInfo.combinations.index(comb)] for comb in combinations}

    return f, v, coeffs

def getEvents( data ):
    coeffs       = data_generator.vector_branch(     data, 'p_C', padding_target=len(weightInfo.combinations))
    features     = data_generator.scalar_branches(   data, feature_names )
    vectors      = {key:data_generator.vector_branch(data, key ) for key in vector_branches}

    return features, vectors, coeffs

def getWeights( eft, coeffs, lin=False):

    if lin:
        combs = list(filter( lambda c:len(c)<2, weightInfo.combinations))
    else:
        combs = weightInfo.combinations
    fac = np.array( [ functools.reduce( operator.mul, [ (float(eft[v]) - weightInfo.ref_point_coordinates[v]) for v in comb ], 1 ) for comb in combs], dtype='float')
    #print (fac)
    return np.matmul(coeffs[:,:len(combs)], fac)

tex = { 'cHj1':'C_{Hj}^{(1)}','cHj3':'C_{Hj}^{(3)}', 'cHu':'C_{Hu}','cHd':'C_{Hd}', 'cHQ3':'C_{HQ}^{(3)}', 'cHb':'C_{Hb}','cHudRe':'C_{Hud}^{Re}','cuWRe':'C_{uW}^{Re}', 'cuBRe':'C_{uB}^{Re}', 'cuHRe':'C_{uH}^{Re}', 'cHDD':'C_{HDD}', 'cHbox':'C_{H#box}', 'cH':'C_{H}', 'cW':'C_{W}', 'cWtil':'C_{Wtil}', 'cHW':'C_{HW}', 'cHWtil':'C_{HWtil}', 'cHWB':'C_{HWB}', 'cHB':'C_{HB}', 'cHWBtil':'C_{HWBtil}', 'cHBtil':'C_{HBtil}'}

eft_plot_points = [
    {'color':ROOT.kBlack,       'eft':sm, 'tex':"SM"},
    {'color':ROOT.kBlue-4,     'eft':make_eft(cW=1),     'tex':"C_{W}=1" },
    {'color':ROOT.kBlue+2,     'eft':make_eft(cW=-1),  'tex':"C_{W}=-1" },
    {'color':ROOT.kGreen-4,     'eft':make_eft(cWtil=1),     'tex':"C_{Wtil}=1" },
    {'color':ROOT.kGreen+2,     'eft':make_eft(cWtil=-1),  'tex':"C_{Wtil}=-1" },
    ]

#cHj1 1 cHj3 1 cHu 1 cHd 1 cHQ1 1 cHQ3 1 cHb 1 cHudRe 1 cuWRe 1 cuBRe 1 cuHRe 1 cW 1 cWtil 1 cH 1 cHbox 1 cHDD 1 cHW 1 cHB 1 cHWB 1 cHWtil 1 cHBtil 1 cHWBtil 1
# Command line arguments: ../../../../make_reweight_card.py --couplings 2 cHj1 1 cHj3 1 cHu 1 cHd 1 cHQ1 1 cHQ3 1 cHb 1 cHudRe 1 cuWRe 1 cuBRe 1 cuHRe 1 cW 1 cWtil 1 cH 1 cHbox 1 cHDD 1 cHW 1 cHB 1 cHWB 1 cHWtil 1 cHBtil 1 cHWBtil 1 --referencepoint cHDD 1 cHbox 1 cW 1 cWtil 1 cHW 1 cHWtil 1 cHWB 1 cHB 1 cHWBtil 1 --filename WhadZlepJJ_EWK_LO_SM_mjj100_pTj10_reweight_card.dat --overwrite

if __name__=="__main__":
   
    # load some events and their weights 
    x, _, w = getEvents(data_generator[0])
    eft_weights = getWeights( make_eft(**weightInfo.ref_point), w)
