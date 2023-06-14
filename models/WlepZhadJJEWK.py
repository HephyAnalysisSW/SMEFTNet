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

selection = lambda ar: (ar.delphesJet_dR_hadV_maxq1q2<0.6) & (ar.delphesJet_dR_matched_hadV_parton<0.6)

import config

data_generator =  DataGenerator(
    input_files = [os.path.join( config.data_directory, "WlepZhadJJ/WlepZhadJJ_*.root")],
        n_split = 1,
        splitting_strategy = "files",
        selection   = selection,
        branches = [

                    "parton_hadV_pt", "parton_hadV_eta", "parton_hadV_angle_phi", "parton_hadV_mass", "parton_hadV_pdgId", "parton_hadV_angle_theta", "parton_hadV_angle_Theta", "parton_hadV_angle_phi", 
                    "parton_hadV_q1_pt", "parton_hadV_q1_eta", "parton_hadV_q1_phi", "parton_hadV_q1_mass", "parton_hadV_q1_pdgId", 
                    "parton_hadV_q2_pt", "parton_hadV_q2_eta", "parton_hadV_q2_phi", "parton_hadV_q2_mass", "parton_hadV_q2_pdgId", 
                    "parton_lepV_pt", "parton_lepV_eta", "parton_lepV_phi", "parton_lepV_mass", "parton_lepV_pdgId", 
                    "parton_lepV_l1_pt", "parton_lepV_l1_eta", "parton_lepV_l1_phi", "parton_lepV_l1_mass", "parton_lepV_l1_pdgId", 
                    "parton_lepV_l2_pt", "parton_lepV_l2_eta", "parton_lepV_l2_phi", "parton_lepV_l2_mass", "parton_lepV_l2_pdgId", 
                    "parton_hadV_angle_theta", "parton_hadV_angle_Theta", "parton_hadV_angle_phi", 

                    "delphesJet_pt", "delphesJet_eta", "delphesJet_phi", "delphesJet_mass", "delphesJet_nConstituents",

                    "delphesJet_SDmass", "delphesJet_SDsubjet0_eta", "delphesJet_SDsubjet0_deltaEta", "delphesJet_SDsubjet0_phi", "delphesJet_SDsubjet0_deltaPhi", "delphesJet_SDsubjet0_deltaR", 
                    "delphesJet_SDsubjet0_mass", "delphesJet_SDsubjet1_eta", "delphesJet_SDsubjet1_deltaEta", "delphesJet_SDsubjet1_phi", "delphesJet_SDsubjet1_deltaPhi", 
                    "delphesJet_SDsubjet1_deltaR", "delphesJet_SDsubjet1_mass", 
                    "delphesJet_tau1", "delphesJet_tau2", "delphesJet_tau3", "delphesJet_tau4", "delphesJet_tau21", "delphesJet_tau32", 
                    "delphesJet_ecf1", "delphesJet_ecf2", "delphesJet_ecf3", "delphesJet_ecfC1", "delphesJet_ecfC2", "delphesJet_ecfC3", "delphesJet_ecfD", "delphesJet_ecfDbeta2", "delphesJet_ecfM1", "delphesJet_ecfM2", "delphesJet_ecfM3", "delphesJet_ecfM1beta2", "delphesJet_ecfM2beta2", "delphesJet_ecfM3beta2", "delphesJet_ecfN1", "delphesJet_ecfN2", "delphesJet_ecfN3", "delphesJet_ecfN1beta2", "delphesJet_ecfN2beta2", "delphesJet_ecfN3beta2", "delphesJet_ecfU1", "delphesJet_ecfU2", "delphesJet_ecfU3", "delphesJet_ecfU1beta2", "delphesJet_ecfU2beta2", "delphesJet_ecfU3beta2", 

                    "delphesJet_dR_matched_hadV_parton", "delphesJet_dR_lepV_parton", "delphesJet_dR_hadV_q1", "delphesJet_dR_hadV_q2", "delphesJet_dR_hadV_maxq1q2",

                    "p_C"]
    )

reweight_pkl = '/eos/vbc/group/cms/robert.schoefbeck/gridpacks/ParticleNet/WlepZhadJJNoRef_reweight_card.pkl'
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

feature_names = [   
    "parton_hadV_pt",     "parton_hadV_eta",     "parton_hadV_mass", 
    "parton_hadV_angle_theta",     "parton_hadV_angle_Theta",     "parton_hadV_angle_phi", 
    "parton_hadV_q1_pt",     "parton_hadV_q1_eta",     "parton_hadV_q1_mass",     "parton_hadV_q1_pdgId", 
    "parton_hadV_q2_pt",     "parton_hadV_q2_eta",     "parton_hadV_q2_mass",     "parton_hadV_q2_pdgId", 
    "parton_lepV_pt",     "parton_lepV_eta",     "parton_lepV_mass", 
    "parton_lepV_l1_pt",     "parton_lepV_l1_eta",     "parton_lepV_l1_phi",     "parton_lepV_l1_pdgId", 
    "parton_lepV_l2_pt",     "parton_lepV_l2_eta",     "parton_lepV_l2_phi",     "parton_lepV_l2_pdgId", 

    "delphesJet_pt", "delphesJet_eta", "delphesJet_mass", "delphesJet_nConstituents",

    "delphesJet_SDmass", "delphesJet_SDsubjet0_deltaEta","delphesJet_SDsubjet0_deltaPhi", "delphesJet_SDsubjet0_deltaR", 
    "delphesJet_SDsubjet0_mass",  "delphesJet_SDsubjet1_deltaEta", "delphesJet_SDsubjet1_deltaPhi", 
    "delphesJet_SDsubjet1_deltaR", "delphesJet_SDsubjet1_mass", 
    "delphesJet_tau1", "delphesJet_tau2", "delphesJet_tau3", "delphesJet_tau4", "delphesJet_tau21", "delphesJet_tau32", 
    "delphesJet_ecf1", "delphesJet_ecf2", "delphesJet_ecf3", "delphesJet_ecfC1", "delphesJet_ecfC2", "delphesJet_ecfC3", "delphesJet_ecfD", "delphesJet_ecfDbeta2", 
    "delphesJet_ecfM1", "delphesJet_ecfM2", "delphesJet_ecfM3", "delphesJet_ecfM1beta2", "delphesJet_ecfM2beta2", "delphesJet_ecfM3beta2", 
    "delphesJet_ecfN1", "delphesJet_ecfN2", "delphesJet_ecfN3", "delphesJet_ecfN1beta2", "delphesJet_ecfN2beta2", "delphesJet_ecfN3beta2", 
    "delphesJet_ecfU1", "delphesJet_ecfU2", "delphesJet_ecfU3", "delphesJet_ecfU1beta2", "delphesJet_ecfU2beta2", "delphesJet_ecfU3beta2", 

    "delphesJet_dR_matched_hadV_parton", "delphesJet_dR_lepV_parton", "delphesJet_dR_hadV_q1", "delphesJet_dR_hadV_q2", "delphesJet_dR_hadV_maxq1q2",

                    ]

def getEvents( nTraining ):
    data_generator.load(-1, small=nTraining )
    #combinations = make_combinations( wilson_coefficients )
    coeffs = data_generator.vector_branch('p_C')

    f = data_generator.scalar_branches( feature_names )
    #f = {key:f[:,i_key] for i_key, key in enumerate(feature_names)}
    v = {key:data_generator.vector_branch( key ) for key in vector_branches}
    #w = {comb:coeffs[:,weightInfo.combinations.index(comb)] for comb in combinations}

    return f, v, coeffs

def getWeights( eft, coeffs, lin=False):

    if lin:
        combs = list(filter( lambda c:len(c)<2, weightInfo.combinations))
    else:
        combs = weightInfo.combinations
    fac = np.array( [ functools.reduce( operator.mul, [ (float(eft[v]) - weightInfo.ref_point_coordinates[v]) for v in comb ], 1 ) for comb in combs], dtype='float')
    #print (fac)
    return np.matmul(coeffs[:,:len(combs)], fac)

tex = { 'cHj1':'C_{Hj}^{(1)}','cHj3':'C_{Hj}^{(3)}', 'cHu':'C_{Hu}','cHd':'C_{Hd}', 'cHQ1':'C_{HQ}^{(1)}', 'cHQ3':'C_{HQ}^{(3)}', 'cHb':'C_{Hb}','cHudRe':'C_{Hud}^{Re}','cuWRe':'C_{uW}^{Re}', 'cuBRe':'C_{uB}^{Re}', 'cuHRe':'C_{uH}^{Re}', 'cHDD':'C_{HDD}', 'cHbox':'C_{H#box}', 'cH':'C_{H}', 'cW':'C_{W}', 'cWtil':'C_{Wtil}', 'cHW':'C_{HW}', 'cHWtil':'C_{HWtil}', 'cHWB':'C_{HWB}', 'cHB':'C_{HB}', 'cHWBtil':'C_{HWBtil}', 'cHBtil':'C_{HBtil}'}

#cHj1_0p_cHj3_0p_cHQ1_0p_cHQ3_0p_cW_0p_cWtil_0p_cHW_0p_cHWtil_0p

eft_plot_points = [
    {'color':ROOT.kBlack,       'eft':sm, 'tex':"SM"},
    {'color':ROOT.kMagenta-4,   'eft':make_eft(cHj1=1),   'tex':"C_{Hj}^{(1)}=1" },
    {'color':ROOT.kMagenta+2,   'eft':make_eft(cHj3=.1),  'tex':"C_{Hj}^{(3)}=.1" },
#    {'color':ROOT.kGreen-4,     'eft':make_eft(cHu=1),     'tex':"C_{Hu}=1" },
#    {'color':ROOT.kGreen+2,     'eft':make_eft(cHd=1),  'tex':"C_{Hd}=1" },
    {'color':ROOT.kBlue+2,      'eft':make_eft(cHQ1=1),    'tex':"C_{HQ}^{(1)}=1" },
    {'color':ROOT.kBlue-4,      'eft':make_eft(cHQ3=1),    'tex':"C_{HQ}^{(3)}=1" },
#    {'color':ROOT.kBlue-4,      'eft':make_eft(cHb=1),    'tex':"C_{Hb}=1" },
#    {'color':ROOT.kCyan+2,      'eft':make_eft(cHudRe=1),    'tex':"C_{Hud}^{(Re)}=1" },
#    {'color':ROOT.kCyan-4,      'eft':make_eft(cuWRe=1),    'tex':"C_{uW}^{(Re)}=1" },
#    {'color':ROOT.kOrange+2,    'eft':make_eft(cuBRe=1),    'tex':"C_{HuB}^{(Re)}=1" },
#    {'color':ROOT.kOrange-4,    'eft':make_eft(cuHRe=1),    'tex':"C_{Hu}^{(Re)}=1" },
#    {'color':ROOT.kMagenta+2,      'eft':make_eft(cHW=1),    'tex':"C_{HW}=1" },
#    {'color':ROOT.kMagenta-4,      'eft':make_eft(cHWtil=1),    'tex':"C_{HWtil}=1" },
    {'color':ROOT.kGreen-4,     'eft':make_eft(cW=1),     'tex':"C_{W}=1" },
    {'color':ROOT.kGreen+2,     'eft':make_eft(cWtil=1),  'tex':"C_{Wtil}=1" },
#    {'color':ROOT.kBlue-4,   'eft':make_eft(cHDD=1),   'tex':"C_{HDD}=1" },
#    {'color':ROOT.kBlue+2,   'eft':make_eft(cHbox=1),  'tex':"C_{H#box}=1" },
    {'color':ROOT.kBlue+2,   'eft':make_eft(cHW=1),  'tex':"C_{HW}=1" },
    {'color':ROOT.kCyan+2,      'eft':make_eft(cHWtil=1),    'tex':"C_{HWtil}=1" },
#    {'color':ROOT.kCyan+2,      'eft':make_eft(cHWB=1),    'tex':"C_{HWB}=1" },
#    {'color':ROOT.kCyan-4,      'eft':make_eft(cHB=1),    'tex':"C_{HB}=1" },
#    {'color':ROOT.kOrange+2,    'eft':make_eft(cHWBtil=1),    'tex':"C_{HWBtil}=1" },
#    {'color':ROOT.kOrange-4,    'eft':make_eft(cHBtil=1),    'tex':"C_{HBtil}=1" },
    ]

from plot_options import *

#cHj1 1 cHj3 1 cHu 1 cHd 1 cHQ1 1 cHQ3 1 cHb 1 cHudRe 1 cuWRe 1 cuBRe 1 cuHRe 1 cW 1 cWtil 1 cH 1 cHbox 1 cHDD 1 cHW 1 cHB 1 cHWB 1 cHWtil 1 cHBtil 1 cHWBtil 1
# Command line arguments: ../../../../make_reweight_card.py --couplings 2 cHj1 1 cHj3 1 cHu 1 cHd 1 cHQ1 1 cHQ3 1 cHb 1 cHudRe 1 cuWRe 1 cuBRe 1 cuHRe 1 cW 1 cWtil 1 cH 1 cHbox 1 cHDD 1 cHW 1 cHB 1 cHWB 1 cHWtil 1 cHBtil 1 cHWBtil 1 --referencepoint cHDD 1 cHbox 1 cW 1 cWtil 1 cHW 1 cHWtil 1 cHWB 1 cHB 1 cHWBtil 1 --filename WhadZlepJJ_EWK_LO_SM_mjj100_pTj10_reweight_card.dat --overwrite

if __name__=="__main__":
   
    # load some events and their weights 
    x, w = getEvents(1000)

    # x are a list of feature-vectors such that x[0] are the features of the first event. Their branch-names are stored in feature_names.
    # w are a dictionary with the weight-coefficients. The key tuple(), i.e., the empty n-tuple, is the constant term. The key ('ctWRe', ), i.e., the coefficient 
    # that is an tuple of length one is the linear derivative (in this case wrt to ctWRe). The quadratic derivatives are stored in the keys ('ctWRe', 'ctWRe') etc.
    # The list of Wilson coefficients is in: weightInfo.variables
    # The list of all derivatives (i.e., the list of all combiations of all variables up to length 2) is weightInfo.combinations. It includes the zeroth derivative, i.e., the constant term.

    # Let us scale all the weights to reasonable numbers. They come out of MG very small because the cross section of the process I used to genereate the top quarks is so small: s-channel single-top
    # Let us add up the constant terms of all events and normalize the sample to the number of events. (Arbitrary choice)
    const = (len(w[()])/w[()].sum())
    for k in w.keys():
        w[k] *= const 

    ##let's remove the most extreme weight derivatives ... cosmetics for the propaganda plots
    #from   tools import helpers 
    #len_before = len(x)
    #auto_clip = 0.001
    #x, w = helpers.clip_quantile(x, auto_clip, weights = w )
    #print ("Auto clip efficiency (training) %4.3f is %4.3f"%( auto_clip, len(x)/len_before ) )

    print ("Wilson coefficients:", weightInfo.variables )
    #print ("Features of the first event:\n" + "\n".join( ["%25s = %4.3f"%(name, value) for name, value in zip(feature_names, x[0])] ) )
    prstr = {0:'constant', 1:'linear', 2:'quadratic'}
    print ("Weight coefficients(!) of the first event:\n"+"\n".join( ["%30s = %4.3E"%( prstr[len(comb)] + " " +",".join(comb), w[comb][0]) for comb in weightInfo.combinations] ) )

    # Let us compute the quadratic weight for ctWRe=1:
    import copy
    eft_sm  = make_eft()
    eft_bsm = make_eft(ctWRe=1)
    # constant term
    reweight = copy.deepcopy(w[()])
    # linear term
    for param1 in wilson_coefficients:
        reweight += (eft_bsm[param1]-eft_sm[param1])*w[(param1,)] 
    # quadratic term
    for param1 in wilson_coefficients:
        if eft_bsm[param1]-eft_sm[param1] ==0: continue
        for param2 in wilson_coefficients:
            if eft_bsm[param2]-eft_sm[param2] ==0: continue
            reweight += .5*(eft_bsm[param1]-eft_sm[param1])*(eft_bsm[param2]-eft_sm[param2])*w[tuple(sorted((param1,param2)))]

    print ("w(ctWRe=1) for the first event is: ", reweight[0])

    # Let us compute the weight ratio we can use for the training:
    target_ctWRe        = w[('ctWRe',)]/w[()]
    target_ctWRe_ctWRe  = w[('ctWRe','ctWRe')]/w[()]

    # NOTE!! These "target" branches are already written to the training data! No need to compute
