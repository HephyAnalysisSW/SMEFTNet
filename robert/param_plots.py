#!/usr/bin/env python

# Standard imports
import ROOT
import torch
import glob
import pickle
import numpy as np
import os, sys, copy
sys.path.insert(0, '../..')
sys.path.insert(0, '..')
from math import sqrt

dir_path = os.path.dirname(os.path.realpath(__file__))
ROOT.gROOT.LoadMacro(os.path.join( dir_path, "../tools/scripts/tdrstyle.C"))
ROOT.setTDRStyle()

from   tools import helpers
import tools.syncer as syncer
import tools.helpers as helpers

# User
import tools.user as user

# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument("--plot_directory",     action="store",      default="SMEFTNet",                 help="plot sub-directory")
argParser.add_argument("--config",              action="store",      default="regress_genTops_lin_ctWRe_pt5_features_charge_pt",                  help="Which config?")
argParser.add_argument("--prefix",              action="store",      default='closure',                  help="prefix?")
argParser.add_argument("--training",           action="store",      default="v4_0p8_2020_2020",              help="Which training?")
argParser.add_argument('--clip',  action='store', type=float,   default=None)
#argParser.add_argument("--input_files",        action="store",      default="/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/predictions/ctGIm/TT01j_HT800_ext_comb/output_*.root", type=str,  help="input files")

args = argParser.parse_args()

exec("import configs.%s as config"%args.config)

# directory for plots
plot_directory = os.path.join( user.plot_directory, args.plot_directory, args.config, (args.prefix + '_' if args.prefix is not None else '') + args.training)
os.makedirs( plot_directory, exist_ok=True)

ROOT.gStyle.SetPalette(ROOT.kBird)
ROOT.gROOT.LoadMacro(os.path.join( dir_path, "../tools/scripts/tdrstyle.C"))
ROOT.setTDRStyle()

torch.autograd.set_grad_enabled(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#/groups/hephy/cms/robert.schoefbeck/NN/models/SMEFTNet/regress_genTops_lin_ctWRe/v3
model_directory = "/groups/hephy/cms/robert.schoefbeck/NN/models/SMEFTNet/%s/%s/"%(args.config,args.training)
files = glob.glob( os.path.join( model_directory, 'epoch-*_state.pt') )

# GIF animation
tex = ROOT.TLatex()
tex.SetNDC()
tex.SetTextSize(0.06)

states = []
for i_filename, filename in enumerate(files):
    stuff = []
    epoch = int(filename.split('-')[-1].split('_')[0])
    print('At %s' % filename)
    if i_filename==0:
        cfg_dict = pickle.load( open( os.path.join( os.path.dirname(filename), 'best_cfg_dict.pkl'), 'rb') )
        model = config.get_model(
            dRN=cfg_dict['dRN'],
            conv_params=cfg_dict['conv_params'],
            readout_params=cfg_dict['readout_params'],
            )
        model.cfg_dict = cfg_dict

    states.append( torch.load(filename, map_location=device) )

tgraphs = {}
for key in states[0].keys():
    tgraphs[key] = []
    for param in range(len( states[0][key].flatten() )):
        tgraphs[key].append( ROOT.TGraph() )
        #tgraphs[key].append( [] ) #ROOT.TGraph(len(files)) )

for i_state, state in enumerate(states):
    for key in state.keys():
        for i_param, param in enumerate(state[key].flatten()):
            tgraphs[key][i_param].AddPoint( float(i_state), param.item() )
            #assert False, ""
            #tgraphs[key][i_param].append( (float(i_state), float(param.item())) )
            #if key == 'mlp.norms.0.module.weight' and i_param==0:
            #    print (key, i_param, i_state, param.item() )

#assert False, "" 
#model.load_state_dict(model_state)

plot_directory_ = os.path.join( plot_directory, "param_plots" )
os.makedirs( plot_directory_, exist_ok=True)

c1 = ROOT.TCanvas()
for key in tgraphs.keys():
    filename = key

    max_ = max(map( max, [ list(t.GetY()) for t in tgraphs[key] ] ))
    min_ = min(map( min, [ list(t.GetY()) for t in tgraphs[key] ] ))

    max_ = 1.2*max_ if max_>0 else 0.8*max_
    min_ = 0.8*min_ if min_>0 else 1.2*min_

    Nparam=100
    for i_t, t in enumerate(tgraphs[key]):
        i_plot = i_t//Nparam
        name = key.lower()+('.part%i'%(i_plot-1) if i_plot>1 else '')
        if i_t%Nparam==0:
            t.Draw("AL")
            t.GetYaxis().SetRangeUser( min_, max_ )
            t.GetYaxis().SetTitle( name )
            t.GetXaxis().SetTitle( "epoch" )
        else:
            t.Draw("L")

        if (i_t%50==0 or i_t==len(tgraphs[key])-1) and i_t>0:
            c1.Print( os.path.join( plot_directory_, name+'.png' ) )
    
helpers.copyIndexPHP( plot_directory_ )
syncer.sync()

#c1.Print( os.path.join( plot_directory_, "epoch%s_%05i.png"%(postfix, epoch) ) )

#    syncer.makeRemoteGif(plot_directory_, pattern="epoch%s_*.png"%postfix, name="epoch%s"%postfix )
#    syncer.sync()
