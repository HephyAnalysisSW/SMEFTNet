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
argParser.add_argument("--model",              action="store",      default="WZto2L_HT300",   help="Which model?")
argParser.add_argument("--WCs",                action="store",      nargs="*", default="WZto2L_HT300",   help="Which wilson coefficients?")
argParser.add_argument("--prefix",             action="store",      default="v6", type=str,  help="prefix")

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

stuff = []
h    = {}

sm = model.make_eft()
weights     = model.getWeights( sm, coeffs)

for i_WC, WC in enumerate(args.WCs):

    h[WC]     = {}

    for i_feature, feature in enumerate(model.feature_names):
        h[WC][feature]      = ROOT.TH1F(WC+'_'+feature,WC+'_'+feature+'', *model.plot_options[feature]['binning'] )

    scores = .5*(model.getWeights( model.make_eft(**{WC:1}), coeffs) - model.getWeights( model.make_eft(**{WC:-1}), coeffs))/weights

    for i_feature, feature in enumerate(model.feature_names):
        binning = model.plot_options[feature]['binning']

        h[WC][feature] = helpers.make_TH1F( np.histogram(features[:,i_feature], np.linspace(binning[1], binning[2], binning[0]+1), weights=weights*scores) )
        norm           = helpers.make_TH1F( np.histogram(features[:,i_feature], np.linspace(binning[1], binning[2], binning[0]+1), weights=weights) )
        h[WC][feature].Divide(norm)
        h[WC][feature].SetLineWidth(2)
        h[WC][feature].SetLineColor( ROOT.kBlack+i_WC )
        h[WC][feature].SetMarkerStyle(0)
        h[WC][feature].SetMarkerColor(ROOT.kBlack+i_WC)
        h[WC][feature].legendText = "t(%s)"%WC 

for i_feature, feature in enumerate(model.feature_names):

    histos = [h[WC][feature] for WC in args.WCs]
    max_   = max( map( lambda h__:h__.GetMaximum(), histos ))
    min_   = min( map( lambda h__:h__.GetMinimum(), histos ))
    if min_<0 and abs(min_)>max_:
        max_=abs(min_)

    for logY in [True, False]:

        c1 = ROOT.TCanvas("c1");
        l = ROOT.TLegend(0.2,0.75,0.9,0.91)
        l.SetNColumns(2)
        l.SetFillStyle(0)
        l.SetShadowColor(ROOT.kWhite)
        l.SetBorderSize(0)
        for i_histo, histo in enumerate(reversed(histos)):
            histo.GetXaxis().SetTitle(model.plot_options[feature]['tex'])
            histo.GetYaxis().SetTitle("score")
            if i_histo == 0:
                histo.Draw('hist')
                histo.GetYaxis().SetRangeUser( (0.001 if logY else -1.3*max_), (10*max_ if logY else 2*1.3*max_))
                #FIXME histo.GetYaxis().SetRangeUser( (0.001 if logY else 0), (10*max_ if logY else 1.3*max_))

                histo.Draw('hist')
            else:
                histo.Draw('histsame')
            l.AddEntry(histo, histo.legendText)
            c1.SetLogy(logY)
        l.Draw()

        plot_directory_ = os.path.join( plot_directory, "expected_score_plots", "log" if logY else "lin" )
        helpers.copyIndexPHP( plot_directory_ )
        c1.Print( os.path.join( plot_directory_, feature+'.png' ))

print ("Done with plots")
syncer.sync()

