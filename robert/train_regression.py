import warnings
warnings.filterwarnings("ignore")

import os
import math
import numpy as np
import torch
import pickle
import glob

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--overwrite', action='store_true', default=False, help="restart training?")
parser.add_argument('--prefix',    action='store', default='v1', help="Prefix for training?")
parser.add_argument('--config',    action='store', default='regressJet', help="Which config?")
parser.add_argument('--learning_rate', '--lr',    action='store', default=0.001, help="Learning rate")
parser.add_argument('--epochs', action='store', default=100, type=int, help="Number of epochs.")
#parser.add_argument('--load_every', action='store', default=5, type=int, help="Load new chunk of data every this number of epochs.")
parser.add_argument('--dRN',  action='store', type=float,   default=0.4)
parser.add_argument('--conv_params',  action='store',       default="( (0.0, [20, 20]), )", help="Conv params")
parser.add_argument('--readout_params',  action='store',    default="(  0.0, [32, 32])", help="Conv params")

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
################### make model ###################################

model = config.get_model(
    dRN=args.dRN,
    conv_params=eval(args.conv_params),
    readout_params=eval(args.readout_params),
    )

################### model, scheduler, loss #######################
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=1./20)

#################### Loading previous state #####################
model.cfg_dict = {'best_loss_test':float('inf')}
model.cfg_dict.update( {key:getattr(args, key) for key in ['prefix', 'learning_rate', 'epochs' ]} )

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
        model.load_state_dict(model_state)
        opt_state_file = load_file_name.replace('_state.pt', '_optimizer.pt') 
        if os.path.exists(opt_state_file):
            opt_state = torch.load(opt_state_file, map_location=device)
            optimizer.load_state_dict(opt_state)
        else:
            print('Optimizer state file %s NOT found!' % opt_state_file)
        epoch_min=load_epoch+1
        model.cfg_dict = pickle.load( open( load_file_name.replace('_state.pt', '_cfg_dict.pkl'), 'rb') )
        # FIXME should add warning if keys change

########################  Training loop ##########################

# read all data
#pt, angles, weights, truth = config.data_model.getEvents(config.data_model.data_generator[0])
#train_mask = torch.FloatTensor(pt.shape[0]).uniform_() < 0.8

#assert False, ""

#data_counter=0
for epoch in range(epoch_min, args.epochs):

    #if epoch%args.load_every==0 or epoch==epoch_min:
    #    pt, angles, weights, truth = config.data_model.getEvents(config.data_model.data_generator[data_counter])
    #    data_counter+=1
    #    if data_counter == len(config.data_model.data_generator): data_counter=0
 
    #    train_mask = torch.FloatTensor(pt.shape[0]).uniform_() < 0.8 
    #    print ("New training and test dataset.")
    n_samples = 0
    for i_data, data in enumerate(config.data_model.data_generator):
        pt, angles, weights, truth = config.data_model.getEvents(data)
 
        train_mask = torch.FloatTensor(pt.shape[0]).uniform_() < 0.8 
        #print ("Training data set %i/%i" % (i_data, len(config.data_model.data_generator)))

        out  = model(pt=pt[train_mask], angles=angles[train_mask])
        loss = config.loss(out, truth[train_mask], weights[train_mask] if weights is not None else None)
        n_samples += len(out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():

            out_test  =  model(pt=pt[~train_mask], angles=angles[~train_mask])
            scale_test_train = len(out)/(len(out_test))
            loss_test = config.loss( out_test, truth[~train_mask], weights[~train_mask] if weights is not None else None)
            loss_test*=scale_test_train
     
            if not "test_losses" in model.cfg_dict:
                model.cfg_dict["train_losses"] = []
                model.cfg_dict["test_losses"] = []

            if i_data == 0:
                model.cfg_dict["train_losses"].append( loss.item() )
                model.cfg_dict["test_losses" ].append( loss_test.item() )
            else:
                model.cfg_dict["train_losses"] [-1] += loss.item()
                model.cfg_dict["test_losses" ] [-1] += loss_test.item()
            print ("Data %i/%i Loss train/test = %4.3f/%4.3f (Total: %4.3f/%4.3f)" % (
                    i_data, len(config.data_model.data_generator),
                    loss.item(), loss_test.item(), 
                    model.cfg_dict["train_losses"] [-1], model.cfg_dict["test_losses" ][-1]), 
                    end="\r")

    print ("")

    print(f'Epoch {epoch:03d} with N={n_samples:03d}, Loss(train): {loss:.4f} Loss(test, scaled to nTrain): {loss_test:.4f}')

    model.cfg_dict['epoch'] = epoch
    if args.prefix:
        if loss_test.item()<model.cfg_dict['best_loss_test']:
            model.cfg_dict['best_loss_test'] = loss_test.item()
            torch.save(  model.state_dict(),     os.path.join( model_directory, 'best_state.pt'))
            torch.save(  optimizer.state_dict(), os.path.join( model_directory, 'best_optimizer.pt'))
            pickle.dump( model.cfg_dict,          open(os.path.join( model_directory, 'best_cfg_dict.pkl'),'wb'))
            
        torch.save(  model.state_dict(),     os.path.join( model_directory, 'epoch-%d_state.pt' % epoch))
        torch.save(  optimizer.state_dict(), os.path.join( model_directory, 'epoch-%d_optimizer.pt' % epoch))
        pickle.dump( model.cfg_dict,          open(os.path.join( model_directory, 'epoch-%d_cfg_dict.pkl' % epoch),'wb'))
