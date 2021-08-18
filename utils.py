import numpy as np
import os
from copy import copy
from collections import Counter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def load_expr(path):

    expr = np.loadtxt(path,dtype=str)
    genes = expr[1:,0]

    return expr[1:,1:].astype(np.float32), genes

def load_TOM(path):

    T = np.loadtxt(os.path.join(path,'whole.txt'),dtype=np.float32)
    pos_T = np.loadtxt(os.path.join(path,'positive.txt'),dtype=np.float32)
    neg_T = np.loadtxt(os.path.join(path,'negative.txt'),dtype=np.float32)

    return T, pos_T, neg_T

def cal_TOM(X,betas):
   
    corr = np.corrcoef(X)
    p_corr = copy(corr) 
    n_corr = copy(corr)
  
    p_corr[corr<0] = 0
    n_corr[corr>=0] = 0

    Ts = []
    for b, c in enumerate([corr,p_corr,n_corr]):
        
        A = c ** betas[b]
        L = A @ A
        K = A.sum(axis=1)

        T = np.zeros_like(A)
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                num = L[i,j] + A[i,j]
                den = min(K[i],K[j]) + 1 - A[i,j]
                T[i,j] = num/den

        T += T.T
        T[range(len(T)),range(len(T))] = 1

        Ts.append(T.astype(np.float32))

    return Ts

def save_TOM(path,TOMs):
    T, pos_T, neg_T = TOMs
    
    os.system('mkdir {}'.format(os.path.join(path,'TOMs')))
    
    np.savetxt(os.path.join(path,'TOMs','whole.txt'),T, delimiter = '\t')
    np.savetxt(os.path.join(path,'TOMs','positive.txt'),pos_T, delimiter = '\t')
    np.savetxt(os.path.join(path,'TOMs','negative.txt'),neg_T, delimiter = '\t')

def save_plot(path, Xe, M):
    pca = PCA(n_components=2).fit_transform(Xe)
    labels = np.array([str(i+1) for i in np.argmax(M, axis=1)]) 
    
    colors = sns.color_palette('colorblind', len(set(labels)))
    
    fig = plt.figure(figsize = (4, 4))
    for label in set(labels):
        indices = labels == str(label)
        plt.scatter(pca[indices][:,0],
                    pca[indices][:,1],
                    color = colors[int(label)-1], s = 10)
    
    plt.yticks([])
    plt.xticks([])
    plt.title('CEPR_embedding features')
    plt.tight_layout()

    plt.savefig(os.path.join(path,'CEPR_embedding.png'), transparent=True)

def save_labels(path, M, genes,config):
    
    labels = np.array([str(i+1) for i in np.argmax(M, axis=1)]) 
    
    print('-'*50)
    for k,v in sorted(Counter(labels).items()): print('K{}: {}'.format(k,v))
    print('-'*50)
    
    with open(os.path.join(path, 'labels.txt'),'w') as save:
        for gene, label in zip(genes, labels): save.write(gene+'\t'+label+'\n')

    with open(os.path.join(path, 'config.txt'),'w') as save:
        for k, v in config.items():
            save.write(k+' : '+str(v)+'\n')
    