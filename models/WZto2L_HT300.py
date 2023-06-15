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

import tools.user as user
from plot_options import *
                
data_generator = DataGenerator(
        input_files = [os.path.join( user.data_directory, "v6/WZto2L_HT300/WZto2L_HT300_*.root")],
        n_split = -1,
        splitting_strategy = "files",
        selection   = selection,
        branches = list(plot_options.keys()) + ["p_C"]
    )

reweight_pkl = '/eos/vbc/group/cms/robert.schoefbeck/gridpacks/ParticleNet/WZto2LNoRef_HT300_reweight_card.pkl'
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
    {'color':ROOT.kMagenta-4,   'eft':make_eft(cHj1=1),   'tex':"C_{Hj}^{(1)}=1" },
    {'color':ROOT.kMagenta+2,   'eft':make_eft(cHj3=.1),  'tex':"C_{Hj}^{(3)}=.1" },
#    {'color':ROOT.kGreen-4,     'eft':make_eft(cHu=1),     'tex':"C_{Hu}=1" },
#    {'color':ROOT.kGreen+2,     'eft':make_eft(cHd=1),  'tex':"C_{Hd}=1" },
#    {'color':ROOT.kBlue+2,      'eft':make_eft(cHQ3=1),    'tex':"C_{HQ}^{(3)}=1" },
#    {'color':ROOT.kBlue-4,      'eft':make_eft(cHb=1),    'tex':"C_{Hb}=1" },
#    {'color':ROOT.kCyan+2,      'eft':make_eft(cHudRe=1),    'tex':"C_{Hud}^{(Re)}=1" },
#    {'color':ROOT.kCyan-4,      'eft':make_eft(cuWRe=1),    'tex':"C_{uW}^{(Re)}=1" },
#    {'color':ROOT.kOrange+2,    'eft':make_eft(cuBRe=1),    'tex':"C_{HuB}^{(Re)}=1" },
#    {'color':ROOT.kOrange-4,    'eft':make_eft(cuHRe=1),    'tex':"C_{Hu}^{(Re)}=1" },
#    {'color':ROOT.kMagenta+2,      'eft':make_eft(cHW=1),    'tex':"C_{HW}=1" },
#    {'color':ROOT.kMagenta-4,      'eft':make_eft(cHWtil=1),    'tex':"C_{HWtil}=1" },
    {'color':ROOT.kGreen-4,     'eft':make_eft(cW=1),     'tex':"C_{W}=1" },
    {'color':ROOT.kGreen+2,     'eft':make_eft(cWtil=1),  'tex':"C_{Wtil}=1" },
    {'color':ROOT.kBlue-4,   'eft':make_eft(cHDD=1),   'tex':"C_{HDD}=1" },
#    {'color':ROOT.kBlue+2,   'eft':make_eft(cHbox=1),  'tex':"C_{H#box}=1" },
    {'color':ROOT.kCyan+2,      'eft':make_eft(cHWB=1),    'tex':"C_{HWB}=1" },
#    {'color':ROOT.kCyan-4,      'eft':make_eft(cHB=1),    'tex':"C_{HB}=1" },
#    {'color':ROOT.kOrange+2,    'eft':make_eft(cHWBtil=1),    'tex':"C_{HWBtil}=1" },
#    {'color':ROOT.kOrange-4,    'eft':make_eft(cHBtil=1),    'tex':"C_{HBtil}=1" },
    ]

from plot_options import *

#cHj1 1 cHj3 1 cHu 1 cHd 1 cHQ1 1 cHQ3 1 cHb 1 cHudRe 1 cuWRe 1 cuBRe 1 cuHRe 1 cW 1 cWtil 1 cH 1 cHbox 1 cHDD 1 cHW 1 cHB 1 cHWB 1 cHWtil 1 cHBtil 1 cHWBtil 1
# Command line arguments: ../../../../make_reweight_card.py --couplings 2 cHj1 1 cHj3 1 cHu 1 cHd 1 cHQ1 1 cHQ3 1 cHb 1 cHudRe 1 cuWRe 1 cuBRe 1 cuHRe 1 cW 1 cWtil 1 cH 1 cHbox 1 cHDD 1 cHW 1 cHB 1 cHWB 1 cHWtil 1 cHBtil 1 cHWBtil 1 --referencepoint cHDD 1 cHbox 1 cW 1 cWtil 1 cHW 1 cHWtil 1 cHWB 1 cHB 1 cHWBtil 1 --filename WhadZlepJJ_EWK_LO_SM_mjj100_pTj10_reweight_card.dat --overwrite

if __name__=="__main__":
   
    # load some events and their weights 
    x, _, w = getEvents(data_generator[0])
    eft_weights = getWeights( make_eft(**weightInfo.ref_point), w)
