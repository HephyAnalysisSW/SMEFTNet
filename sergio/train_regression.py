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
parser.add_argument('--config',    action='store', default='regressJet', help="Which config?")
parser.add_argument('--learning_rate', '--lr',    action='store', default=0.01, help="Learning rate")
parser.add_argument('--epochs', action='store', default=100, type=int, help="Number of epochs.")
parser.add_argument('--nSplit', action='store', default=1000, type=int, help="Number of epochs.")

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_num_threads(8)

import sys
sys.path.insert(0, '..')
import tools.user as user

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
config.model.train()
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
    for data in pbar:
        pt, angles, features, weights, truth = config.data_model.getEvents(data)
        train_mask = torch.FloatTensor(pt.shape[0]).uniform_() < 0.8

        optimizer.zero_grad()

        out  = config.model(pt=pt[train_mask], angles=angles[train_mask], features=features[train_mask])
        loss = config.loss(out, truth[train_mask], weights[train_mask] if weights is not None else None)
        pbar.set_description(f'Loss: {loss.item()}')

        n_samples = len(out)
        loss.backward()
        optimizer.step()
        if pt_test is None:
            pt_test=pt[~train_mask]
            angles_test=angles[~train_mask]
            features_test=features[~train_mask]
            weights_test=None
            truth_test=truth[~train_mask]
        else:
            pt_test=torch.cat((pt_test,pt[~train_mask]),axis=0)
            angles_test=torch.cat((angles_test,angles[~train_mask]),axis=0)
            features_test=torch.cat((features_test,features[~train_mask]),axis=0)
            weights_test=None
            truth_test=torch.cat((truth_test,truth[~train_mask]),axis=0)
            
    with torch.no_grad():
        out_test  =  config.model(pt=pt_test, angles=angles_test, features=features_test)
        loss_test = config.loss( out_test, truth_test, weights_test)
        print(f"Test loss is {loss_test}")
        plt.hist2d(  out_test[:,0].numpy(),  truth_test.numpy() , bins=30)
        plt.savefig( os.path.join( model_directory, f'test_cov_{epoch}.png'))
        plt.clf()
 
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
