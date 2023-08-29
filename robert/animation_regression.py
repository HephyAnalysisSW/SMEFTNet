import torch
import numpy as np
import math 
import array
import sys, os
sys.path.insert(0, '..')
import glob
import tools.syncer as syncer
import tools.user as user
import tools.helpers as helpers
import pickle

import ROOT
dir_path = os.path.dirname(os.path.realpath(__file__))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', action='store', default='regress_genTops_lin_ctWRe_pt5')
parser.add_argument('--training', action='store', default='v4_0p4_2020_2020')
parser.add_argument('--xmin', action='store', default=-.5, type=float)
parser.add_argument('--xmax', action='store', default=+.5, type=float)
parser.add_argument('--nBins', action='store', default=20, type=int)
parser.add_argument('--varName', action='store', default='C_{tG}^{Re}', help="Which prefix?")

args = parser.parse_args()

delay  = 50/5

#config_name    = "regress_genTops_lin_ctWRe_pt5"
#training_name  = "v4_0p4_2020_2020"
#xmin, xmax = -.5, .5
#varName = "C_{tW}^{Re}"
index_truth = 0
index_out   = 0
every = 5
#logZ=False
def func( out ):
    return out[:, 0] 

exec("import configs.%s as config"%args.config)

config.data_model.data_generator.reduceFiles(to=10)

pt, angles, features, scalar_features, weights, truth = config.data_model.getEvents(config.data_model.data_generator[-1])

#scalar_features = config.data_model.data_generator.scalar_branches( config.data_model.data_generator[-1], model.feature_names )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plot_directory_ = os.path.join( user.plot_directory, 'SMEFTNet', "regression", args.config, args.training)
os.makedirs( plot_directory_, exist_ok=True)

ROOT.gStyle.SetPalette(ROOT.kBird)
ROOT.gROOT.LoadMacro(os.path.join( dir_path, "../tools/scripts/tdrstyle.C"))
ROOT.setTDRStyle()

torch.autograd.set_grad_enabled(False)

#/groups/hephy/cms/robert.schoefbeck/NN/models/SMEFTNet/regress_genTops_lin_ctWRe/v3
model_directory = "/groups/hephy/cms/robert.schoefbeck/NN/models/SMEFTNet/%s/%s/"%(args.config,args.training)
files = glob.glob( os.path.join( model_directory, 'epoch-*_state.pt') )

# Text on the plots
tex = ROOT.TLatex()
tex.SetNDC()
tex.SetTextSize(0.04)

c1 = ROOT.TCanvas()
for i_filename,  filename in enumerate(files[0::every]):
    load_epoch = int(filename.split('-')[-1].split('_')[0])
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
    out = model( pt, angles, features=features, scalar_features=scalar_features)
    h = ROOT.TH2F( "R", "R", args.nBins,args.xmin, args.xmax,args.nBins, args.xmin, args.xmax)
    for R_pred, R_true in zip(func(out).numpy(), truth.numpy()):
        h.Fill( R_true, R_pred)

    for logZ in [True, False]:
        h.Scale(1./h.Integral())
        h.Draw("COLZ")
        #h.GetZaxis().SetRangeUser(0,100)
        h.GetYaxis().SetTitle(args.varName+" (pred)")
        h.GetXaxis().SetTitle(args.varName+" (truth)")
        line1 = ( 0.16, 0.96, "Epoch %i"%load_epoch )
        o = tex.DrawLatex(*line1) 
        o.Draw()
        c1.SetRightMargin(0.15)
        c1.SetLogz(logZ)
        c1.Print(os.path.join( plot_directory_, "log" if logZ else "lin", args.training+'_%s.png'%(str(load_epoch).zfill(5))) )
        syncer.makeRemoteGif(os.path.join( plot_directory_, "log" if logZ else "lin"), pattern=args.training+"_*.png", name=args.training, delay=delay)

syncer.sync()
helpers.copyIndexPHP( plot_directory_ )
