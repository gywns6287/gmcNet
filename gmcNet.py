from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import tensorflow as tf
import math
import numpy as np

def MinCut_loss(w):
    
    def func(M,T):
    
        n, k = M.shape
        D = tf.linalg.diag(tf.reduce_sum(T,axis=1))
    
        Lc = -(tf.linalg.trace(tf.matmul(tf.matmul(tf.transpose(M),T),M))/
               tf.linalg.trace(tf.matmul(tf.matmul(tf.transpose(M),D),M)))

        SS = tf.matmul(tf.transpose(M),M)
        Ik = tf.eye(k, dtype = SS.dtype)
        Lo = tf.norm(SS/tf.norm(SS) - Ik/tf.norm(Ik))
    
        return w*Lc + Lo , (Lc.numpy(),Lo.numpy())
    
    return func

def matrix_normalization(T):
    T[range(len(T)),range(len(T))] = 0
    D = np.diag(np.sum(T,axis=0) ** (-1/2))
    return np.matmul(np.matmul(D,T),D)

class CEPR(Layer):

    def __init__(self, out_feats, activation = None):
        
        super(CEPR, self).__init__()
        
        self.out_feats = out_feats
        self.activation = tf.keras.activations.get(activation)
    
    def build(self, input_shape):
        
        self.self_layer = tf.keras.layers.Dense(self.out_feats)
        self.neg_layer = tf.keras.layers.Dense(self.out_feats)
        self.pos_layer = tf.keras.layers.Dense(self.out_feats)

    def call(self, inputs):
        
        X0, X, P, N = inputs
        
        pos_X = tf.matmul(P,X)
        neg_X = tf.matmul(N,X)
        
        self_F = self.self_layer(X0)
        pos_F = self.pos_layer(pos_X)
        neg_F = self.neg_layer(neg_X)
        
        return self.activation(self_F+pos_F+neg_F) 

class gmcNet(tf.keras.Model):
    
    def __init__(self, n_clusters, mp_layers = 1, CEPR_features = 8):
        
        super(gmcNet, self).__init__()
        
        self.n_clusters = n_clusters
        self.mp_layers = mp_layers
        self.CEPR_features = CEPR_features
        
    def build(self, input_shape):
        
        self.CEPR_layers = []
        for l in range(self.mp_layers): 
            self.CEPR_layers.append(CEPR(self.CEPR_features,'relu'))
            
        self.mlp =  Dense(self.n_clusters, activation='softmax')
        
    def call(self, inputs):
        
        X0, P, N = inputs
        
        for i, layer in enumerate(self.CEPR_layers):
            if i == 0:
                X = layer([X0,X0,P,N])
            else:
                X = layer([X0,X,P,N])
        
        M = self.mlp(X)
        
        return M, X

def Clustering(X, Ts, n_clusters, mp_layers = 1, CEPR_features = 8,
            epochs = 100, lr = 1e-4, lamb = 1, Lo_thr = 0.6, tune_epoch=100, tune_lr = 1e-3):
        
    model = gmcNet(n_clusters, mp_layers = mp_layers, CEPR_features = CEPR_features)
    opt = tf.keras.optimizers.Adam(lr = tune_lr)
    loss_fn = MinCut_loss(0)
    T, Tp, Tn = Ts
    
    nT = matrix_normalization(T)
    nTp = matrix_normalization(Tp)
    nTn = matrix_normalization(Tn)

    for epoch in range(1,epochs+1):
        
        if epoch%10 == 0:
            print('Epoch {}/{}....'.format(epoch,epochs),end='') 

        if epoch == tune_epoch:
            opt.lr = lr
            loss_fn = MinCut_loss(lamb)          
        
        with tf.GradientTape() as tape:    
            M, Xe = model([X,nTp,nTn],training=True)
            loss, monitor = loss_fn(M,T)
            
        if epoch > tune_epoch and monitor[1] > Lo_thr:
            print('\n Training is early stop on Epoch '+str(epoch))
            return M, Xe
        
        gradients = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))
        
        if epoch%10 == 0:
            if epoch%100 == 0:
                print('{}..{}'.format(loss.numpy(),monitor)+' '*10)
            else:
                print('{}..{}'.format(loss.numpy(),monitor)+' '*10,end='\r')
        
    return M, Xe
