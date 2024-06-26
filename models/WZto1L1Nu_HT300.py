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
        input_files = [os.path.join( user.data_directory, "v6/WZto1L1Nu_HT300/WZto1L1Nu_HT300_*.root")],
        n_split = -1,
        splitting_strategy = "files",
        selection   = selection,
        branches = list(plot_options.keys()) + ["p_C"]
    )

reweight_pkl = '/eos/vbc/group/cms/robert.schoefbeck/gridpacks/ParticleNet/WZto1L1NuNoRef_HT300_reweight_card.pkl'
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
#('cHj1',) -0.19147025574594592
#('cHj3',) 11.008226990736716
#('cHu',) -3.047507555296214e-11
#('cHd',) 4.925264735832264e-12
#('cHQ1',) -2.955158841499359e-11
#('cHQ3',) -4.063343407061618e-11
#('cHb',) -4.063343407061618e-11
#('cHudRe',) -3.90942888406686e-11
#('cuWRe',) -3.047507555296214e-11
#('cuBRe',) -3.693948551874198e-11
#('cuHRe',) -3.324553696686778e-11
#('cW',) 0.36759169575135575
#('cWtil',) -19.85648104998451
#('cH',) -3.5862083857778676e-11
#('cHbox',) -4.3711724530511346e-11
#('cHDD',) 0.09253571120487394
#('cHW',) -1.662276848343389e-11
#('cHB',) -1.5699281345465343e-11
#('cHWB',) -0.5666321612548608
#('cHWtil',) -3.663165647275247e-11
#('cHBtil',) -3.632382742676295e-11
#('cHWBtil',) -0.15464831714701896


if __name__=="__main__":
    # load some events and their weights 
    x, _, w = getEvents(data_generator[0])
    eft_weights = getWeights( make_eft(**weightInfo.ref_point), w)
