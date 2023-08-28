import warnings
warnings.filterwarnings("ignore")

import os
import math
import numpy as np
import torch
import pickle
import glob
import tqdm 
import argparse
import matplotlib.pyplot as plt 
parser = argparse.ArgumentParser()
parser.add_argument('--overwrite', action='store_true', default=False, help="restart training?")
parser.add_argument('--prefix',    action='store', default='v1', help="Prefix for training?")
parser.add_argument('--config',    action='store', default='regress_wz_v0', help="Which config?")
parser.add_argument('--learning_rate', '--lr',    action='store', default=0.01, help="Learning rate")
parser.add_argument('--epochs', action='store', default=100, type=int, help="Number of epochs.")
parser.add_argument('--nSplit', action='store', default=1000, type=int, help="Number of epochs.")

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Hola")
    torch.set_num_threads(16)

import sys
sys.path.insert(0, '..')
import tools.user as user
from collections import defaultdict 
exec("import configs.%s as config"%args.config)

# reproducibility
torch.manual_seed(0)
import numpy as np
np.random.seed(0)

###################### Micro MC Toy Data #########################
import MicroMC
########################## directories ###########################
model_directory = os.path.join( user.model_directory, 'SMEFTNet',  args.config, args.prefix)
os.makedirs( model_directory, exist_ok=True)
print ("Using model directory", model_directory)

################### model, scheduler, loss #######################
optimizer = torch.optim.Adam(config.model.parameters(), lr=float(args.learning_rate))
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=1./20)

#################### Loading previous state #####################
config.model.cfg_dict = {'best_loss_test':float('inf')}
config.model.cfg_dict.update( {key:getattr(args, key) for key in ['prefix', 'learning_rate', 'epochs', 'nSplit' ]} )

epoch_min = 0
if not args.overwrite:
    files = glob.glob( os.path.join( model_directory, 'epoch-*_state.pt') )
    if len(files)>0:
        load_file_name = max( files, key = lambda f: int(f.split('-')[-1].split('_')[0]))
        load_epoch = int(load_file_name.split('-')[-1].split('_')[0])
    else:
        load_epoch = None
    if load_epoch is not None:
        print('Resume training from %s' % load_file_name)
        model_state = torch.load(load_file_name, map_location=device)
        config.model.load_state_dict(model_state)
        opt_state_file = load_file_name.replace('_state.pt', '_optimizer.pt') 
        if os.path.exists(opt_state_file):
            opt_state = torch.load(opt_state_file, map_location=device)
            optimizer.load_state_dict(opt_state)
        else:
            print('Optimizer state file %s NOT found!' % opt_state_file)
        epoch_min=load_epoch+1
        config.model.cfg_dict = pickle.load( open( load_file_name.replace('_state.pt', '_cfg_dict.pkl'), 'rb') )
        # FIXME should add warning if keys change

########################  Training loop ##########################

# read all data

for epoch in range(epoch_min, args.epochs):

    #if epoch%10==0 or epoch==epoch_min:
    #    train_mask = torch.FloatTensor(args.nTraining).uniform_() < 0.8
    #    print ("New training and test dataset.")
    pt_test=None
    pbar=tqdm.tqdm(config.data_model.data_generator)
    test_elements=defaultdict(list)
    for data in pbar:
        pt, angles, features, scalar_features, weights, truth = config.data_model.getEvents(data)
        batch_size = pt.shape[0]
        train_size = int(batch_size*0.8)
        train_mask = (torch.cat([torch.ones(train_size), torch.zeros(batch_size-train_size)],axis=0) == 1)

        optimizer.zero_grad()

        out  = config.model(
            pt=pt[train_mask], 
            angles=angles[train_mask], 
            features=features[train_mask] if features is not None else None,
            scalar_features=scalar_features[train_mask] if scalar_features is not None else None,)

        loss = config.loss(out, truth[train_mask], weights[train_mask] if weights is not None else None)
        pbar.set_description(f'Loss: {loss.item()}')

        n_samples = len(out)
        loss.backward()
        optimizer.step()

        test_elements['pt'].append(pt[~train_mask])
        test_elements['angles'].append(angles[~train_mask])
        test_elements['scalar_features'].append(scalar_features[~train_mask] if scalar_features is not None else None)
        test_elements['features'].append(features[~train_mask] if features is not None else None)
        test_elements['weights'].append(weights[~train_mask])
        test_elements['truth'].append(truth[~train_mask])
    
    for el in test_elements:
        test_elements[el]=torch.cat( test_elements[el],axis=0) if test_elements[el][0] is not None else None
            
    with torch.no_grad():
        out_test  =  config.model(
            pt=test_elements['pt'], 
            angles=test_elements['angles'], 
            features=test_elements['features'],
            scalar_features=test_elements['scalar_features'])
        loss_test = config.loss( out_test, test_elements['truth'], test_elements['weights'])
        print(f"Test loss is {loss_test}")
        config.plot( out_test, test_elements['truth'], test_elements['weights'], model_directory, epoch)
        if not "test_losses" in config.model.cfg_dict:
            config.model.cfg_dict["train_losses"] = []
            config.model.cfg_dict["test_losses"] = []
        config.model.cfg_dict["train_losses"].append( loss.item() )
        config.model.cfg_dict["test_losses"].append(  loss_test.item() )

    print(f'Epoch {epoch:03d}: Loss(train): {loss:.4f} Loss(test): {loss_test:.4f}')

    config.model.cfg_dict['epoch']       = epoch
    if args.prefix:
        if loss_test.item()<config.model.cfg_dict['best_loss_test']:
            config.model.cfg_dict['best_loss_test'] = loss_test.item()
            torch.save(  config.model.state_dict(),     os.path.join( model_directory, 'best_state.pt'))
            torch.save(  optimizer.state_dict(), os.path.join( model_directory, 'best_optimizer.pt'))
            pickle.dump( config.model.cfg_dict,          open(os.path.join( model_directory, 'best_cfg_dict.pkl'),'wb'))
            
        torch.save(  config.model.state_dict(),     os.path.join( model_directory, 'epoch-%d_state.pt' % epoch))
        torch.save(  optimizer.state_dict(), os.path.join( model_directory, 'epoch-%d_optimizer.pt' % epoch))
        pickle.dump( config.model.cfg_dict,          open(os.path.join( model_directory, 'epoch-%d_cfg_dict.pkl' % epoch),'wb'))
