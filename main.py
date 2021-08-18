#2021.08.03
#Hyo-Jun Lee

##########Set Config##########
config = {
'betas' : (6,9,10),  # smoothing parameter for (whole, positive, negative) networks
'save_TOM' : True,   # save TOM or not
'save_embed' : True, # save embedding features or not 
'n_cluster' : 4,     # number of cluster (k)
'epochs' : 5000,     # trainning epochs
'lr' : 1e-3,         # trainning learning rate
'mp_layers' : 1,     # number of message passing layers
'CEPR_features' : 8, # CEPR_embedding demesions
'lambda' : 2.2,      # balancing hyper-parameter 
'Lo_thr' : 0.6,      # orthogonal threshold
'tune_epoch' : 100,  # first tunning epochs, which prevent the empty modules 
'tune_lr' : 1e-2,    # learning rate for first tunning
'device' : 0         # used GPU device. if Not use GPU, then write False 
}
##############################

# DO NOT change the code below unless you're familiar with python and tensorflow

#Load library and etc..
import os
if config['device']:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=config['device']

import argparse
import numpy as np 
from gmcNet import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--expr", help="PATH of expression data.")
parser.add_argument("--TOM", help="if you input TOM, code will not compute TOM (this will save time).", default='')
parser.add_argument("--out", help="PATH for saving results")
args = parser.parse_args()

#Load data
##Load expression data
print('Load Expression data......')
X, genes = load_expr(args.expr)
print('Data includes {} genes with {} expression samples.'.format(len(genes),len(X[0])))

##Load or Calculate TOM
if args.TOM:
    print('Load TOM...')
    T, Tp, Tn = load_TOM(args.TOM)
else:
    print('Calculate TOM...')
    T, Tp, Tn = cal_TOM(X,config['betas'])

##save TOM
if config['save_TOM']:
    print('Save TOMs at {}/TOMs'.format(args.out))
    save_TOM(args.out, [T,Tp,Tn])

#Clustering
##Run gmcNet...
print('Rum gmcNet...')
M, Xe = Clustering(X, [T,Tp,Tn], config['n_cluster'],
                mp_layers = config['mp_layers'], CEPR_features = config['CEPR_features'],
                epochs = config['epochs'], lr = config['lr'], lamb = config['lambda'],
                Lo_thr = config['Lo_thr'], tune_epoch = config['tune_epoch'], tune_lr = config['tune_lr'])
print('Done..')

##Save CEPR_embedding features
if config['save_embed']:
    print('Save CEPR_embdding feature at {}/CEPR_embedding.txt'.format(args.out))
    print('Save CEPR_embdding Plot at {}/CEPR_embedding.png'.format(args.out))
    np.savetxt(os.path.join(args.out,'CEPR_embedding.txt'),Xe, delimiter = '\t')        
    save_plot(args.out, Xe, M)

##Save clustering results
print('Save clustering labels at {}/labels.txt'.format(args.out))
save_labels(args.out, M, genes, config)