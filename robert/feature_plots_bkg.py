#!/usr/bin/env python

# Standard imports
import ROOT
import numpy as np
import math
import array
import sys, os, copy
import functools, operator

sys.path.insert(0, '..')
import tools.syncer as syncer
import tools.user as user
import tools.helpers as helpers


#ROOT.gStyle.SetPalette(ROOT.kBird)
dir_path = os.path.dirname(os.path.realpath(__file__))
ROOT.gROOT.LoadMacro(os.path.join( dir_path, "../tools/scripts/tdrstyle.C"))
ROOT.setTDRStyle()

# Parser
import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument("--plot_directory",     action="store",      default="SMEFTNet",     help="plot sub-directory")
argParser.add_argument("--prefix",             action="store",      default="WZandDYModel_v2", type=str,  help="prefix")

args = argParser.parse_args()

# directory for plots
plot_directory = os.path.join( user.plot_directory, args.plot_directory, args.prefix)
os.makedirs( plot_directory, exist_ok=True)

import models.WZandDYModel as model

feature_names = list(model.plot_options.keys())

data_model = model.WZandDYModel(scalar_features = feature_names)

#def getEvents( data ):
#    features     = model.data_generator.scalar_branches(   data, model.feature_names )
#    return features

features_sig  = data_model.signal_generator.scalar_branches(   data_model.signal_generator[-1], feature_names ) 
features_bkg  = data_model.bkg_generator.scalar_branches(   data_model.bkg_generator[0], feature_names ) 

# Text on the plots
def drawObjects( offset=0 ):
    tex1 = ROOT.TLatex()
    tex1.SetNDC()
    tex1.SetTextSize(0.05)
    tex1.SetTextAlign(11) # align right

    tex2 = ROOT.TLatex()
    tex2.SetNDC()
    tex2.SetTextSize(0.04)
    tex2.SetTextAlign(11) # align right

    line1 = ( 0.15+offset, 0.95, "Boosted Info Trees" )
    return [ tex1.DrawLatex(*line1) ]#, tex2.DrawLatex(*line2) ]

###############
## Plot Model #
###############

stuff = []
h    = {}

for name, features, color in [("sig", features_sig, ROOT.kRed), ("bkg", features_bkg, ROOT.kBlue)]:

    h[name]     = {}

    for i_feature, feature in enumerate(feature_names):
        h[name][feature]        = ROOT.TH1F(name+'_'+feature, name+'_'+feature, *model.plot_options[feature]['binning'] )

    for i_feature, feature in enumerate(feature_names):
        binning = model.plot_options[feature]['binning']

        h[name][feature] = helpers.make_TH1F( np.histogram(features[:,i_feature], np.linspace(binning[1], binning[2], binning[0]+1)) )

        h[name][feature].SetLineWidth(2)
        h[name][feature].SetLineColor( color )
        h[name][feature].SetMarkerStyle(0)
        h[name][feature].SetMarkerColor(color)
        h[name][feature].legendText = name

for i_feature, feature in enumerate(feature_names):

    for name in ["sig", "bkg"]:
        for i_histo, (name, histo) in enumerate(h[name].items()):
            norm = histo.Integral()
            if norm>0:
                histo.Scale(1./norm) 

    for logY in [True, False]:

        c1 = ROOT.TCanvas("c1");
        l = ROOT.TLegend(0.2,0.83,0.9,0.91)
        l.SetNColumns(2)
        l.SetFillStyle(0)
        l.SetShadowColor(ROOT.kWhite)
        l.SetBorderSize(0)

        h["sig"][feature].GetXaxis().SetTitle(model.plot_options[feature]['tex'])
        h["sig"][feature].GetYaxis().SetTitle("1/#sigma_{SM}d#sigma/d%s"%model.plot_options[feature]['tex'])
        h["sig"][feature].Draw('hist')
        #max_ = max( h["sig"][feature].GetMaximum(), h["bkg"][feature].GetMinimum() )
        if logY:
            h["sig"][feature].GetYaxis().SetRangeUser(0.001, 1.1)
        else:
            h["sig"][feature].GetYaxis().SetRangeUser(0., 0.9)
        h["bkg"][feature].Draw('histsame')

        l.AddEntry(h["sig"][feature], h["sig"][feature].legendText)
        l.AddEntry(h["bkg"][feature], h["bkg"][feature].legendText)
        c1.SetLogy(logY)
        l.Draw()

        plot_directory_ = os.path.join( plot_directory, "feature_plots", "log" if logY else "lin" )
        helpers.copyIndexPHP( plot_directory_ )
        c1.Print( os.path.join( plot_directory_, feature+'.png' ))

print ("Done with plots")
syncer.sync()

