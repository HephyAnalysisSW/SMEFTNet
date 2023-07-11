import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import RootDataset 
from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt 

torch.set_num_threads(8)

model = nn.Sequential(
    nn.Linear(1,5),
    nn.ReLU(),
    nn.Linear(5,5),
    nn.ReLU(),
    nn.Linear(5,1),
)

dataset=RootDataset("/pnfs/psi.ch/cms/trivcat/store/user/sesanche/SMEFTNet/v6/WZto2L_HT300/*1.root", ['parton_hadV_angle_phi', 'p_C'])
train,test = torch.utils.data.random_split( dataset, [0.8,0.2], generator=torch.Generator().manual_seed(42))
train_loader     = DataLoader(  train  , batch_size=len(train))
test_loader     = DataLoader(  test  , batch_size=len(test))
train_loader_eval     = DataLoader(  train  , batch_size=len(train))


optimizer = optim.RMSprop(model.parameters(), lr=0.5, momentum=0)

def cost( estimate, weight):
    return torch.mean((estimate-weight.view(-1,1))**2)

costs_train=[]
costs_test=[]
# nsteps=100
# for phi, weight in train_loader_eval:
#     mean = torch.mean(weight).item()
#     print("mean is", mean, 'and std is', torch.std(weight).item())
#     rng=torch.linspace(-mean, 3*mean, steps=nsteps)
#     WW,RR=torch.meshgrid( weight, rng)
#     cost=torch.mean( (WW-RR)**2, axis=0)
#     print(cost.shape)
#     plt.plot( rng.numpy(),  cost.numpy())
#     plt.savefig('cost_scan.png')
#     plt.clf()
#     print(kk)


for epoch in range(50):
    for phi, weight in tqdm( train_loader ):
        optimizer.zero_grad()
        thecost = cost(model(phi.view(-1,1)),weight)
        thecost.backward()
        optimizer.step()
    with torch.no_grad():
        for phi, weight in test_loader:
            #print('mean test', torch.mean(weight))
            estimate = model(phi.view(-1,1))
            costs_test .append( cost(estimate, weight).item())
            hist,bins,_=plt.hist( estimate.numpy(), bins=500, label='Test')
        for phi, weight in train_loader_eval:

            estimate = model(phi.view(-1,1))
            costs_train.append( cost(estimate, weight).item())
            print(estimate.shape, weight.shape)
            print((estimate.view(-1)-weight).shape)

            plt.hist(((estimate.view(-1)-weight)**2 ).numpy(), bins=200)
            plt.yscale('log')
            plt.savefig(f'cost_per_event_{epoch}.png')
            plt.clf() 

            print('neural network cost on train data', costs_train[-1])
            print('cost on train data if we replace neural network by mean', cost(torch.ones_like(estimate)*torch.mean(weight), weight).item())
            print('mean estimate and mean weight', torch.mean(estimate).item(), torch.mean(weight) )
            
            hist,bins,_=plt.hist( estimate.numpy(), bins=bins, label='Train')
        plt.legend()
        plt.savefig(f"hist_epoch_{epoch}.png")
        plt.clf()

plt.plot(costs_train[1:], label='train')
plt.plot(costs_test[1:] , label='test')
plt.savefig('training.png')
plt.clf()
        
    
