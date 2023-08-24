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
argParser.add_argument("--WC",                 action="store",      default="ctWRe", type=str,  help="Which WC?")
argParser.add_argument("--epochs",             action="store",      nargs="*", type=int,  help="Which epochs to plot?")
argParser.add_argument('--clip',  action='store', type=float,   default=None)
#argParser.add_argument("--input_files",        action="store",      default="/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/predictions/ctGIm/TT01j_HT800_ext_comb/output_*.root", type=str,  help="input files")

args = argParser.parse_args()

exec("import configs.%s as config"%args.config)

config.data_model.data_generator.reduceFiles(to=10)

# directory for plots
plot_directory = os.path.join( user.plot_directory, args.plot_directory, args.config, (args.prefix + '_' if args.prefix is not None else '') + args.training)
os.makedirs( plot_directory, exist_ok=True)

# linear term
derivatives = [(args.WC, ) ]

ROOT.gStyle.SetPalette(ROOT.kBird)
ROOT.gROOT.LoadMacro(os.path.join( dir_path, "../tools/scripts/tdrstyle.C"))
ROOT.setTDRStyle()

torch.autograd.set_grad_enabled(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#/groups/hephy/cms/robert.schoefbeck/NN/models/SMEFTNet/regress_genTops_lin_ctWRe/v3
model_directory = "/groups/hephy/cms/robert.schoefbeck/NN/models/SMEFTNet/%s/%s/"%(args.config,args.training)
files = glob.glob( os.path.join( model_directory, 'epoch-*_state.pt') )

data = config.data_model.data_generator[-1]
pt, angles, features, scalar_features, _, truth = config.data_model.getEvents(data)
weights                        = config.data_model.getWeightDict(data)
observers = list(config.model.plot_options.keys())

observer_features = config.data_model.getScalarFeatures( data, observers ) 

if args.clip is not None:
    len_before = len(pt)
    #selection = helpers.clip_quantile( config.data_model.getScalarFeatures(data).to(device), args.clip, return_selection = True )
    selection = helpers.clip_quantile( truth.view(-1,1), args.clip, return_selection = True )
    pt          = pt[selection]
    angles      = angles[selection]
    features    = features[selection] if features is not None else None
    scalar_features  = scalar_features[selection] if scalar_features is not None else None
    observer_features = observer_features[selection] if observer_features is not None else None
    weights     = {k:v[selection] for k,v in weights.items()}
    truth       = truth[selection]
    print ("Auto clip efficiency (training) %4.3f is %4.3f"%( args.clip, len(pt)/len_before) )

# GIF animation
tex = ROOT.TLatex()
tex.SetNDC()
tex.SetTextSize(0.06)

# colors
color = {}
i_lin, i_diag, i_mixed = 0,0,0
for i_der, der in enumerate(derivatives):
    if len(der)==1:
        color[der] = ROOT.kAzure + i_lin
        i_lin+=1
    elif len(der)==2 and len(set(der))==1:
        color[der] = ROOT.kRed + i_diag
        i_diag+=1
    elif len(der)==2 and len(set(der))==2:
        color[der] = ROOT.kGreen + i_mixed
        i_mixed+=1

every = 5

for i_filename, filename in enumerate(files[0::every]):
    stuff = []
    epoch = int(filename.split('-')[-1].split('_')[0])
    print('At %s' % filename)
    if i_filename==0:
        cfg_dict = pickle.load( open( os.path.join( os.path.dirname(filename), 'best_cfg_dict.pkl'), 'rb') )
        model = config.get_model(
            #dRN=0.4,
            #conv_params=( (0.0, [20, 20]), ),
            #readout_params=(0.0, [32,32]),
            dRN=cfg_dict['dRN'],
            conv_params=cfg_dict['conv_params'],
            readout_params=cfg_dict['readout_params'],
            )
        model.cfg_dict = cfg_dict

    model_state = torch.load(filename, map_location=device)
    model.load_state_dict(model_state)

    predictions = model( pt, angles, features=features, scalar_features=scalar_features)

    # drop angles
    if len(model.EC)>0:
        predictions = predictions[:,:-2].numpy()
    else:
        predictions = predictions.numpy()

    if predictions.ndim==1:
        predictions=predictions.reshape(-1,1) 

    w0 = weights[()]

    # 2D plots for convergence + animation
    th2d = {}
    th1d_pred = {}
    th1d_truth= {}
    for i_der, der in enumerate( derivatives ):
        truth_ratio = weights[der]/w0
        quantiles = np.quantile(np.concatenate( (truth_ratio, predictions[:,i_der] ) ), q=(0.01,1-0.01))

        if len(der)==2: #quadratic
            binning = np.linspace( min([0, quantiles[0]]), quantiles[1], 21 )
        else:
            binning = np.linspace( quantiles[0], quantiles[1], 21 )

        th2d[der]      = helpers.make_TH2F( np.histogram2d( truth_ratio, predictions[:,i_der], bins = [binning, binning], weights=w0) )
        th1d_truth[der]= helpers.make_TH1F( np.histogram( truth_ratio, bins = binning, weights=w0) )
        th1d_pred[der] = helpers.make_TH1F( np.histogram( predictions[:,i_der], bins = binning, weights=w0) )
        tex_name = "%s"%(",".join( der ))
        th2d[der].GetXaxis().SetTitle( tex_name + " truth" )
        th2d[der].GetYaxis().SetTitle( tex_name + " prediction" )
        th1d_pred[der].GetXaxis().SetTitle( tex_name + " prediction" )
        th1d_pred[der].GetYaxis().SetTitle( "Number of Events" )
        th1d_truth[der].GetXaxis().SetTitle( tex_name + " truth" )
        th1d_truth[der].GetYaxis().SetTitle( "Number of Events" )

        th1d_truth[der].SetLineColor(ROOT.kBlack)
        th1d_truth[der].SetMarkerColor(ROOT.kBlack)
        th1d_truth[der].SetMarkerStyle(0)
        th1d_truth[der].SetLineWidth(2)
        th1d_truth[der].SetLineStyle(ROOT.kDashed)
        th1d_pred[der].SetLineColor(ROOT.kBlack)
        th1d_pred[der].SetMarkerColor(ROOT.kBlack)
        th1d_pred[der].SetMarkerStyle(0)
        th1d_pred[der].SetLineWidth(2)

    n_pads = len(derivatives)
    n_col  = len(derivatives) 
    n_rows = 2
    #for logZ in [False, True]:
    #    c1 = ROOT.TCanvas("c1","multipads",500*n_col,500*n_rows);
    #    c1.Divide(n_col,n_rows)

    #    for i_der, der in enumerate(derivatives):

    #        c1.cd(i_der+1)
    #        ROOT.gStyle.SetOptStat(0)
    #        th2d[der].Draw("COLZ")
    #        ROOT.gPad.SetLogz(logZ)

    #    lines = [ (0.29, 0.9, 'N_{B} =%5i'%( epoch )) ]
    #    drawObjects = [ tex.DrawLatex(*line) for line in lines ]
    #    for o in drawObjects:
    #        o.Draw()

    #    for i_der, der in enumerate(derivatives):
    #        c1.cd(i_der+1+len(derivatives))
    #        l = ROOT.TLegend(0.6,0.75,0.9,0.9)
    #        stuff.append(l)
    #        l.SetNColumns(1)
    #        l.SetFillStyle(0)
    #        l.SetShadowColor(ROOT.kWhite)
    #        l.SetBorderSize(0)
    #        l.AddEntry( th1d_truth[der], "R("+tex_name+")")
    #        l.AddEntry( th1d_pred[der],  "#hat{R}("+tex_name+")")
    #        ROOT.gStyle.SetOptStat(0)
    #        th1d_pred[der].Draw("hist")
    #        th1d_truth[der].Draw("histsame")
    #        ROOT.gPad.SetLogy(logZ)
    #        l.Draw()


    #    plot_directory_ = os.path.join( plot_directory, "training_plots", "log" if logZ else "lin" )
    #    os.makedirs( plot_directory_, exist_ok=True)
    #    helpers.copyIndexPHP( plot_directory_ )
    #    c1.Print( os.path.join( plot_directory_, "training_2D_epoch_%05i.png"%(epoch) ) )
    #    syncer.makeRemoteGif(plot_directory_, pattern="training_2D_epoch_*.png", name="training_2D_epoch" )

    for observables, observable_features, postfix in [
        #( model.observers if hasattr(model, "observers") else [], observers, "_observers"),
        ( observers, observer_features, ""),
            ]:
        if len(observables)==0: continue
        h_w0, h_ratio_prediction, h_ratio_truth, lin_binning = {}, {}, {}, {}
        wp_pred = np.multiply(w0[:,np.newaxis], predictions)
        for i_feature, feature in enumerate(observables):
            # root style binning
            binning     = config.model.plot_options[feature]['binning']
            # linspace binning
            lin_binning[feature] = np.linspace(binning[1], binning[2], binning[0]+1)
            #digitize feature
            binned      = np.digitize(observable_features[:,i_feature], lin_binning[feature] )
            # for each digit, create a mask to select the corresponding event in the bin (e.g. test_features[mask[0]] selects features in the first bin
            mask        = np.transpose( binned.reshape(-1,1)==range(1,len(lin_binning[feature])) )
            h_w0[feature]           = np.array([  w0[m].sum() for m in mask])
            h_derivative_prediction = np.array([ wp_pred[m].sum(axis=0) for m in mask])
            h_derivative_truth      = np.array([ (np.transpose(np.array([(weights[der] if der in weights else weights[tuple(reversed(der))]) for der in derivatives])))[m].sum(axis=0) for m in mask])
            h_ratio_prediction[feature] = h_derivative_prediction/(h_w0[feature].reshape(-1,1))
            h_ratio_truth[feature]      = h_derivative_truth/(h_w0[feature].reshape(-1,1))
        del wp_pred

        # 1D feature plot animation
        n_pads = len(observables)+1
        n_col  = int(sqrt(n_pads))
        n_rows = n_pads//n_col
        if n_rows*n_col<n_pads: n_rows+=1

        for logY in [False, True]:
            c1 = ROOT.TCanvas("c1","multipads",500*n_col,500*n_rows);
            c1.Divide(n_col,n_rows)

            l = ROOT.TLegend(0.2,0.1,0.9,0.85)
            stuff.append(l)
            l.SetNColumns(2)
            l.SetFillStyle(0)
            l.SetShadowColor(ROOT.kWhite)
            l.SetBorderSize(0)

            for i_feature, feature in enumerate(observables):

                th1d_yield       = helpers.make_TH1F( (h_w0[feature], lin_binning[feature]) )
                c1.cd(i_feature+1)
                ROOT.gStyle.SetOptStat(0)
                th1d_ratio_pred  = { der: helpers.make_TH1F( (h_ratio_prediction[feature][:,i_der], lin_binning[feature])) for i_der, der in enumerate( derivatives ) }
                th1d_ratio_truth = { der: helpers.make_TH1F( (h_ratio_truth[feature][:,i_der], lin_binning[feature])) for i_der, der in enumerate( derivatives ) }

                stuff.append(th1d_yield)
                stuff.append(th1d_ratio_truth)
                stuff.append(th1d_ratio_pred)

                th1d_yield.SetLineColor(ROOT.kGray+2)
                th1d_yield.SetMarkerColor(ROOT.kGray+2)
                th1d_yield.SetMarkerStyle(0)
                th1d_yield.GetXaxis().SetTitle(config.model.plot_options[feature]['tex'])
                th1d_yield.SetTitle("")

                th1d_yield.Draw("hist")

                for i_der, der in enumerate(derivatives):
                    th1d_ratio_truth[der].SetTitle("")
                    th1d_ratio_truth[der].SetLineColor(color[der])
                    th1d_ratio_truth[der].SetMarkerColor(color[der])
                    th1d_ratio_truth[der].SetMarkerStyle(0)
                    th1d_ratio_truth[der].SetLineWidth(2)
                    th1d_ratio_truth[der].SetLineStyle(ROOT.kDashed)
                    th1d_ratio_truth[der].GetXaxis().SetTitle(config.model.plot_options[feature]['tex'])

                    th1d_ratio_pred[der].SetTitle("")
                    th1d_ratio_pred[der].SetLineColor(color[der])
                    th1d_ratio_pred[der].SetMarkerColor(color[der])
                    th1d_ratio_pred[der].SetMarkerStyle(0)
                    th1d_ratio_pred[der].SetLineWidth(2)
                    th1d_ratio_pred[der].GetXaxis().SetTitle(config.model.plot_options[feature]['tex'])

                    tex_name = "%s"%(",".join( der ))

                    if i_feature==0:
                        l.AddEntry( th1d_ratio_truth[der], "R("+tex_name+")")
                        l.AddEntry( th1d_ratio_pred[der],  "#hat{R}("+tex_name+")")

                if i_feature==0:
                    l.AddEntry( th1d_yield, "yield (SM)")

                max_ = max( map( lambda h:h.GetMaximum(), list(th1d_ratio_truth.values())+list(th1d_ratio_pred.values()) ))
                max_ = 10**(1.5)*max_ if logY else 1.5*max_
                min_ = min( map( lambda h:h.GetMinimum(), list(th1d_ratio_truth.values())+list(th1d_ratio_pred.values()) ))
                min_ = 0.1 if logY else (1.5*min_ if min_<0 else 0.75*min_)

                th1d_yield_min = th1d_yield.GetMinimum()
                th1d_yield_max = th1d_yield.GetMaximum()
                for bin_ in range(1, th1d_yield.GetNbinsX() ):
                    th1d_yield.SetBinContent( bin_, (th1d_yield.GetBinContent( bin_ ) - th1d_yield_min)/th1d_yield_max*(max_-min_)*0.95 + min_  )

                #th1d_yield.Scale(max_/th1d_yield.GetMaximum())
                th1d_yield   .Draw("hist")
                ROOT.gPad.SetLogy(logY)
                th1d_yield   .GetYaxis().SetRangeUser(min_, max_)
                th1d_yield   .Draw("hist")
                for h in list(th1d_ratio_truth.values()) + list(th1d_ratio_pred.values()):
                    h .Draw("hsame")

            c1.cd(len(observables)+1)
            l.Draw()

            lines = [ (0.29, 0.9, 'N_{B} =%5i'%( epoch )) ]
            drawObjects = [ tex.DrawLatex(*line) for line in lines ]
            for o in drawObjects:
                o.Draw()

            plot_directory_ = os.path.join( plot_directory, "training_plots", "log" if logY else "lin" )
            os.makedirs( plot_directory_, exist_ok=True)
            helpers.copyIndexPHP( plot_directory_ )
            c1.Print( os.path.join( plot_directory_, "epoch%s_%05i.png"%(postfix, epoch) ) )
            syncer.makeRemoteGif(plot_directory_, pattern="epoch%s_*.png"%postfix, name="epoch%s"%postfix )
        syncer.sync()
