# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 16:42:27 2016

@author: Chelsea Weaver


This is a function to compute and store tangent basis vectors for use in
the LPCA-SRC classification algorithm. It uses the local PCA technique of 
Singer and Wu (2012) to compute the tangent vectors. 

Inputs: X_train_stacked: A 3D array of training data, with first
            dimension corresponding to feature, second dimension 
            corresponding to training sample, and third dimension 
            corresponding to class
       quant_train: Vector of length L (L = # of classes) with lth entry
            corresponding to the number of training points in the lth
            class. Entries must be integers.
       d_vect: Vector of length L with lth entry corresponding to the
               intrinsic dimension of the lth class (user-specified). Default 
               is d_vect = np.ones((L,1)). 
       n: number of neighbors for use in local PCA (user-specified). Default is
          min(quant_train)-2 (maximum allowed value).
       
 Outputs: DICT_full_stacked: (m x n_train x L) 3D array with
               DICT_full_stacked(:,:,l) corresponding to the decomposition 
               dictionary for the lth class. Columns mod (d+1) correspond
               to the original training vectors in the lth class with the
               previous d columns the corresponding scaled and shifted
               tangent vectors.
          r_1: Scalar to be used in computation of neighborhood radius r.

"""
import numpy as np
from sklearn.neighbors import NearestNeighbors

def Local_PCA( X_train_stacked, quant_train, d_vect=None, n=None ):
    
    if d_vect is None:
        d_vect=np.ones((len(quant_train),1))
        
    if n is None:
        n = min(quant_train)-2

    n = int(n)

    [m,_,L] = np.shape(X_train_stacked)

    # Initialize matrix that will store results:
    DICT_full_stacked = np.zeros((m, int((np.max(d_vect)+1)*np.max(quant_train)), L))

    # For computation of r_1:
    PREP_r_1 = np.zeros((int(np.max(quant_train)), L))

    # Compute median distance between each training point and its (n+1)st 
    # nearest neighbor in the same class:
    for l in range(L):
        N_l = int(quant_train[l])
        d_l = int(d_vect[l])
        
        X_train_l = X_train_stacked[:,0:N_l,l]

        nbrs = NearestNeighbors(n_neighbors=n+2).fit(X_train_l.T)
        D,_ = nbrs.kneighbors(X_train_l.T)
        for i in range(N_l):
            PREP_r_1[i,l] = D[i,-1]

    r_1 = np.median(PREP_r_1)
  
    # Compute tangent vectors:
    for l in range(L):     
        DICT_Class_l = np.zeros((m,N_l*(d_l+1)))
        X_train_l = X_train_stacked[:,0:N_l,l]
        nbrs = NearestNeighbors(n_neighbors=n+2).fit(X_train_l.T)
        for i in range(N_l):
            x_i = X_train_l[:,i]
            DICT_x_i = np.zeros((m,d_l+1))
            # Will contain x_i and the shifted 
            # and scaled tangent plane basis vectors at x_i.
           
            # Find the n-nearest neighbors of x_i in the same class:
            DIST,IDX = nbrs.kneighbors(X_train_l.T)
            Neighbors_i = X_train_l[:,IDX[i,:]]
            Neighbors_i = Neighbors_i[:,1:] # delete the first column since 
                                           # corresponds to x_i

            # Compute epsilon_pca and weight matrix D_i:
            # Set it to be the squared distance from its (n+1)st nearest
            ## neighbor
            R = np.tile(x_i,(n+1,1)).T
            X_i = Neighbors_i - R # neighbors centered around x_i
            
            epsilon_pca = (max(DIST[i,:]))**2
            
            # Delete the (n+1)st neighbor from set of neighbors:
            X_i = X_i[:,:n] # Excludes last column
            # Now the local covariance matrix is X_i*X_i'
            
            # Weight the neighbors of x_i according to their distances from
            # x_i:
            D_i = np.eye(n);
            # Use Epanechnikov function 
            # (1-u^2)*I[0,1] (indicator function on [0,1]):
            for j in range(n):
                val = 1-((DIST[i,j]**2)/epsilon_pca)
                if 0 <= val <=1:
                    K_pca = val
                else:
                    K_pca = 0

                D_i[j,j] = np.sqrt(K_pca)  
                
            B_i = np.matmul(X_i,D_i)
            # The weighted local covariance matrix is B_i*B_i'
            
            # Compute the first d eigenvectors of the weighted covariance
            # matrix using SVD:
            eig_vects,_,_ = np.linalg.svd(B_i);
            DICT_x_i[:,0:d_l] = eig_vects[:,0:d_l]

            # Add the training sample x_i to each vector in the basis after
            # scaling:
            c = np.random.rand()*r_1;
            DICT_x_i = c*DICT_x_i + np.tile(x_i,(d_l+1,1)).T 

            # Store the set of basis vectors for the approximate tangent plane 
            # at x_i on the manifold.
            DICT_Class_l[:,i*(d_l+1): (i+1)*(d_l+1)] = DICT_x_i
            
        DICT_full_stacked[:,0:N_l*(d_l+1),l] = DICT_Class_l 
    
    return DICT_full_stacked, r_1
    
