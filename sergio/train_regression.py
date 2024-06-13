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

    pt_test=None
    pbar=tqdm.tqdm(config.data_model_train.data_generator)
    count=0
    for data in pbar:
        if count > 250: break
        count=count+1
        pt, angles, features, scalar_features, weights, truth = config.data_model_train.getEvents(data)

        optimizer.zero_grad()

        out  = config.model(
            pt=pt, 
            angles=angles, 
            features=features if features is not None else None,
            scalar_features=scalar_features if scalar_features is not None else None,)
        mask_nan = torch.isnan( out[:,0] ) 
        loss = config.loss(out, truth, weights if weights is not None else None)

        pbar.set_description(f'Loss: {loss.item()}')

        n_samples = len(out)
        loss.backward()
        optimizer.step()




    if args.prefix:
        torch.save(  config.model.state_dict(),     os.path.join( model_directory, 'epoch-%d_state.pt' % epoch))
        torch.save(  optimizer.state_dict(), os.path.join( model_directory, 'epoch-%d_optimizer.pt' % epoch))
        pickle.dump( config.model.cfg_dict,          open(os.path.join( model_directory, 'epoch-%d_cfg_dict.pkl' % epoch),'wb'))
    
            
    with torch.no_grad():
        pbar=tqdm.tqdm(enumerate(config.data_model_test.data_generator))
        chunks=[]
        count=0
        for chunk_number,data in pbar:
            count = count+1
            pt, angles, features, scalar_features, weights, truth = config.data_model_test.getEvents(data)

            test_out  =  config.model(
                pt=pt, 
                angles=angles, 
                features=features,
                scalar_features=scalar_features)

            chunks.append( config.plot_chunk( test_out, truth, weights,  scalar_features, chunk_number ))

        merged_chunks=config.add_chunks( chunks ) 
        config.save( merged_chunks, model_directory, epoch ) 
    print(f'Epoch {epoch:03d}: Loss(train): {loss:.4f} ')

            
