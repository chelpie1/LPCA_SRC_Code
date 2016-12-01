# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 14:51:45 2016

@author: Chelsea Weaver

Mon Feb 15 04:36:59 2016 written by Chelsea Weaver

LPCA_SRC_function performs classification using the local PCA
modification of Sparse Representation-based Classification (Wright, et
al. 2009) as proposed in the paper 
    "Improving sparse representation using principal component analysis''
    by C. Weaver and N. Saito



Inputs: X_train: Matrix of training data with rows corresponding to 
            features and columns corresponding to samples
        lab: Vector of training labels
        X_test: Matrix of test data
        ulab_GT: Ground truth label for test data
        epsilon: error/sparsity tradeoff parameter in l1-minimization
            Default is epsilon = 0.001
        occ_on: Specify whether to use or not use occlusion 
            version of algorithm. 0 ~ no occlusion, 1 ~ occlusion
            Default is occ_on = 0
        n: Number of neighbors in local PCA. Must be no more than min class 
            size - 2
            Default is maximum value
        d_vect: Manual specification of intrinsic class
            dimension for each class (Lx1 vector for L = # of classes)
            Default is vector of ones
                      
Outputs: ulab_LPCA_SRC: Vector of class assignments
         t: runtime
         Avg_LEN: Average number of columns in dictionary
         Accuracy: Accuracy of classification (given ground truth test labels)
         
Required scripts/functions:

    Normalize.py

    Local_PCA.py
    
    l1-Minimization function:
       L1L2Py: "Python package to perform variable selection by means of l1l2 
           regularization with double optimization." Available at 
           https://pypi.python.org/pypi/L1L2Py

    

"""
def LPCA_SRC_func( X_train, lab, X_test, ulab_GT, epsilon=None, occ_on=None, n=None, d_vect=None ):

    import numpy as np
    
    if occ_on == None:
        occ_on = 0
    if epsilon == None:
        epsilon = 1e-3

    # Extract information from training data:

    m, n_train = np.shape(X_train)
    L = len(np.unique(np.asarray(lab))) # of classes

    quant_train = np.zeros((L,1)) # number of training points in each class
    for l in range(L):
        quant_train[l] = list(lab).count(l) 
        
    if n == None:
        n = min(quant_train) - 2
    if d_vect == None:
        d_vect = np.ones((L,1))

    # START OFFLINE PHASE ##############################################
    # Start timer
    import time
    start_time = time.time()
    
    # Normalize training data:
    from Normalize import Normalize
    X_train_norm = Normalize(X_train)

    # Stack training data by class:
    n_max_l = int(max(quant_train)) # max class size
    X_train_stacked = np.zeros((m,n_max_l,L))
    for l in range(L):
        n_l = int(quant_train[l])
        class_l_index = list(np.where(lab==l)[0])
        X_train_stacked[:,0:n_l,l] = X_train_norm[:,class_l_index]

    # Compute tangent vectors and neighborhood radius parameter r:
    from Local_PCA import Local_PCA
    DICT_full_stacked, r_1 = Local_PCA( X_train_stacked, quant_train, d_vect, n )
    
    # Write DICT_full_stacked as a 2D matrix and compute label vector as well
    # as an index vector of training points in DICT_full:
    DICT_full = np.zeros((m,int(np.dot(quant_train.T,(d_vect+1)))))
    lab_DICT_full = np.zeros((1,int(np.dot(quant_train.T,(d_vect+1)))))
    train_pt_ind = np.zeros((n_train,1))
    # Note that lab_DICT_full(train_pt_ind(i)) will retrieve the class of the
    # ith training point in X_train.

    count_1 = 0
    count_2 = 0 
    for l in range(L):
        n_l = int(quant_train[l])
        d_l = int(d_vect[l])
        DICT_full[:,count_1:count_1+n_l*(d_l+1)] = DICT_full_stacked[:,0:n_l*(d_l+1),l]
        lab_DICT_full[:,count_1:count_1+n_l*(d_l+1)]= l*np.ones((1,n_l*(d_l+1)))
  
        train_pt_ind[count_2:count_2+n_l,0] = range(count_1+d_l,count_1+(d_l+1)*n_l,d_l+1)
        
        count_1 = count_1 + n_l*(d_l+1)
        count_2 = count_2 + n_l
    
    ## Start Online Phase ###################################################

    # Extract information from test data:

    _, n_test = np.shape(X_test)

    # Normalize test data:
    X_test_norm = Normalize(X_test)
    
    # Compute average number of columns in dictionary after r constraint:
    len_DICT = np.zeros((n_test,1))

    # Initialize test label vector:
    ulab = np.zeros((n_test,1))

    # Begin classification:

    for j in range(n_test):
        y = X_test_norm[:,j]
    
        DICT_y = np.zeros((np.shape(DICT_full)))
        _, col_num = np.shape(DICT_y)
        lab_DICT_y = np.zeros((col_num,1)) # will contain labels

        # Compute distances between the test point and each training point:
        dist_vects_pos = np.tile(y,(n_train,1)).T-X_train_norm
        DIST_pos = np.sqrt(sum(dist_vects_pos**2))

        dist_vects_neg = np.tile(y,(n_train,1)).T+X_train_norm
        DIST_neg = np.sqrt(sum(dist_vects_neg**2))
    
        # Compute minimum distance from y to a class rep
        r_2 = min(min(DIST_pos),min(DIST_neg))
    
        # Set neighborhood radius parameter:
        r = max(r_1,r_2)
    
        # Amend dictionary to include only training points (and their 
        # corresponding tangent basis vectors) that are within r of the 
        # test point y:
        ind_pos = np.asarray(np.where(DIST_pos <= r))
        ind_neg = np.asarray(np.where(DIST_neg <= r))
        if ind_pos.size and ind_neg.size: # if both nonempty
            ind = np.unique(np.concatenate(ind_pos, ind_neg))  
        elif ind_pos.size and not ind_neg.size:
            ind = ind_pos
        elif ind_neg.size and not ind_pos.size:
            ind = ind_neg
     
        _, LEN = np.shape(ind) # size of DICT_y
        
        count = 0
        close_train_pts = train_pt_ind[ind,0]  
        # index of nearby training points in DICT_full
        
        for i in range(LEN):
            class_l_index = int(lab_DICT_full[0,int(close_train_pts[0,i])])
            d_ind = int(d_vect[class_l_index])
            ctp_i = int(close_train_pts[0,i])
            DICT_y[:,count:count+d_ind+1] = DICT_full[:,ctp_i-d_ind:ctp_i+1]
            lab_DICT_y[count:count+d_ind+1] = class_l_index*np.ones((d_ind+1,1))
            count += d_ind+1
    
        # Remove empty columns of DICT_y and entries of lab_DICT_y:
        DICT_y = DICT_y[:,0: count-d_ind]
        lab_DICT_y = lab_DICT_y[0: count-d_ind]
    
        # Store number of columns in dictionary:
        _, len_DICT[j] = np.shape(DICT_y)
        
        # l1-Minimization:#####################################################
    
        # Normalize dictionary:
        DICT_y = Normalize(DICT_y)
    
        from algorithms import l1l2_regularization
        x = l1l2_regularization(DICT_y,y,0,epsilon)

        # Compute class error for each class:
        ERR_y = np.zeros((L,1))
        for l in range(L):
            coeff = np.zeros((int(len_DICT[j]),1))
            ind_l = np.where(np.asarray(lab_DICT_y)==l)[0]
            if ind_l.size: # if DICT_j contains vectors from class l
                coeff[ind_l] = x[ind_l]
                if occ_on == 1:
                    e_hat = x[len_DICT[j]+1:-1]
                    ERR_y[l] = np.linalg.norm(y-e_hat-np.matmul(DICT_y,coeff))
                else:
                    ERR_y[l] = np.linalg.norm(y-np.matmul(DICT_y,coeff))
            else:
                ERR_y[l] = np.inf
    ulab[j] = np.argmin(ERR_y)

    ## Compute accuracy:
    Num_Correct = 0
    for j in range(len(ulab_GT)):
        if ulab[j] == ulab_GT[j]:
            Num_Correct += 1
            
    Accuracy = Num_Correct/n_test


    ## Compute statistics:

    # Computational time:
    t = time.time() - start_time

    # Average number of columns in DICT_y:
    Avg_LEN = np.mean(len_DICT)

    
    return(ulab, t, Avg_LEN, Accuracy)#, Avg_Hom_Iter, Avg_Hom_Time)  