from __future__ import division
import numpy as np
import logging,os
import copy
import logging,os,sys
import pickle
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
from data_loader import load_data
from resnet_keras import train
from img_datagen import  DataGenerator
from scipy import stats
from random import shuffle
from collections import Counter
from functions_keras import generate_workers, generate_labels_weight, majority_voting, post_prob_DS
#from resnet import train, max_val_epoch
n, n1, k = 50000, 10000, 10
m, gamma, class_wise = 100, 0.2, 0
repeat, samples =3, 50000

(train_x, train_y), (test_x, test_y) = load_data()
conf = generate_workers(m,k,gamma,class_wise) 
print ("\nnumber of training examples: " + str(samples) + "\t redundancy: " + str(repeat))
# calling the main function
resp_org, workers_train_label_org, workers_this_example, var, temp= generate_labels_weight(train_y,n,repeat,conf)

_,_,_,_, temp1 = generate_labels_weight(train_y,n1,repeat,conf)

########geting the vote count for classes #########################################################
def varian(temp,q,l):
    e = []
    g = np.zeros((q,l))
    for i in range(q):
        v =Counter(temp[i])
        e = list(v.keys())
        for w in range(len(e)):
                g[i][w] = e[w]
        if len(e) < 10:
            kp = np.arange(w+1,10)
            g[i][kp] = -1
    ###############################################converting into custom representation##
    z = np.zeros((q,l))
    for i in range(q):
        for j in range(l):
            if g[i][j] != -1:
                a  = copy.deepcopy(g[i][j])
                a = int(a)
                z[i][a] = 1
    return z
########################################################################################

z = varian(temp,n,k)
z1 = varian(temp1,n1,k)

print ("Algorithm: majority vote:\t\t"),
pred_mv = majority_voting(resp_org)   
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# t_score, tes_score, _ = train(x_train, pred_mv, y_dummy,z, x_test, y_test)
# print ("train error:  " + str(t_score))
# print ("Test error:  " + str(tes_score))
##########################################################################################
print ("Algorithm: weighted majority vote:\t"), 
naive_agg = np.zeros((n,k))
for r in range(repeat):
    naive_agg = naive_agg + (1/repeat)*copy.deepcopy(workers_train_label_org['softmax'+ str(r) +'_label']) 
# calling the "call_train" function which besides printing the generalization error 
# returns model prediction on the training examples, which is being stored in the variable "naive_pred".
t_score, tes_score, naive_pred = train(x_train, naive_agg, x_test, y_test, z, z1)
print ("train error:  " + str(t_score))
print ("Test error:  " + str(tes_score))
#########################################################################################
print ("Algorithm: MBEM:\t\t\t"),    
probs_est_labels = post_prob_DS(resp_org, naive_pred, workers_this_example)      
algo_agg = np.zeros((n,k))    
algo_agg = probs_est_labels
# calling the "call_train" function with aggregated labels being the posterior probability distribution of the 
# examples given the model prediction obtained using the "weighted majority vote" algorithm.
t_score, tes_score, _ = train(x_train, naive_pred, x_test, y_test, z1)   
print ("train error:  " + str(t_score))
print ("Test error:  " + str(tes_score))
##########################################################################################






