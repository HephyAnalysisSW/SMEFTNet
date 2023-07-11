#!/usr/bin/env python

# Standard imports
import ROOT
import numpy as np
import math
import array
import sys, os, copy

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
argParser.add_argument("--model",              action="store",      default="WlepZhadJJ",   help="Which model?")
argParser.add_argument("--WC",                 action="store",      default="cW",   help="Which Wilson coefficient?")
argParser.add_argument("--prefix",             action="store",      default="v5", type=str,  help="prefix")

args = argParser.parse_args()

exec('import models.%s as model'%(args.model))

# directory for plots
plot_directory = os.path.join( user.plot_directory, args.plot_directory, args.prefix, args.model )
os.makedirs( plot_directory, exist_ok=True)

features, _, coeffs = model.getEvents(model.data_generator[-1])

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

#stuff = []
h    = {}
#h_lin= {}

sm = model.make_eft() 

h     = {}
weights      = model.getWeights( sm, coeffs)
scores       = (model.getWeights( model.make_eft(**{args.WC:1}), coeffs) - model.getWeights( model.make_eft(**{args.WC:-1}), coeffs))/(2*model.getWeights( sm, coeffs))
score_bins   = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]
thresholds   = helpers.weighted_quantile( scores, score_bins, weights=weights)
score_binned = np.digitize( scores, thresholds)

color = ROOT.kAzure

for i_quantile in range(len(score_bins)):
    h[i_quantile] = {} 
    mask = score_binned==i_quantile+1
    for i_feature, feature in enumerate(model.feature_names):
        binning = model.plot_options[feature]['binning']

        h[i_quantile][feature] = helpers.make_TH1F( np.histogram(features[:,i_feature][mask], np.linspace(binning[1], binning[2], binning[0]+1), weights=weights[mask]) )

        h[i_quantile][feature].SetLineWidth(0)
        h[i_quantile][feature].SetLineColor( color + i_quantile )
        h[i_quantile][feature].SetFillColor( color + i_quantile )
        h[i_quantile][feature].SetMarkerStyle(0)
        h[i_quantile][feature].SetMarkerColor(color + i_quantile)
        h[i_quantile][feature].legendText = "(%3.2f, %3.2f)"%(thresholds[i_quantile], thresholds[i_quantile+1])

for i_feature, feature in enumerate(model.feature_names):
    #    norm = _h[model.eft_plot_points[0]['eft']['name']][feature].Integral()
    #    if norm>0:
    #        for eft_plot_point in model.eft_plot_points:
    #            _h[eft_plot_point['eft']['name']][feature].Scale(1./norm) 

    histos = [h[i_quantile][feature] for i_quantile in range(len(score_bins)-1)]
    
    # stack
    for i_h, h_ in enumerate(histos):
        if i_h==0: continue
        h_.Add(histos[i_h-1])
    max_   = histos[-1].GetMaximum() 

    for logY in [True, False]:

        c1 = ROOT.TCanvas("c1");
        l = ROOT.TLegend(0.2,0.75,0.9,0.91)
        l.SetNColumns(2)
        l.SetFillStyle(0)
        l.SetShadowColor(ROOT.kWhite)
        l.SetBorderSize(0)
        for i_histo, histo in enumerate(reversed(histos)):
            histo.GetXaxis().SetTitle(model.plot_options[feature]['tex'])
            histo.GetYaxis().SetTitle("1/#sigma_{SM}d#sigma/d%s"%model.plot_options[feature]['tex'])
            if i_histo == 0:
                histo.Draw('hist')
                histo.GetYaxis().SetRangeUser( (0.001 if logY else 0), (10*max_ if logY else 1.3*max_))

                histo.Draw('hist')
            else:
                histo.Draw('histsame')
            l.AddEntry(histo, histo.legendText)
            c1.SetLogy(logY)
        l.Draw()

        plot_directory_ = os.path.join( plot_directory, "score_plots", "log" if logY else "lin" )
        helpers.copyIndexPHP( plot_directory_ )
        c1.Print( os.path.join( plot_directory_, feature+'.png' ))

print ("Done with plots")
syncer.sync()

