function [ DICT_full_stacked, r_1] = Local_PCA( X_tr_stacked, quant_train, d_vect, n )

% Mon Feb 15 04:36:59 2016 written by Chelsea Weaver
%
% This is a function to compute and store tangent basis vectors for use in
% the LPCA-SRC classification algorithm. It uses the local PCA technique of 
% Singer and Wu (2012) to compute the tangent vectors. 
%
% Inputs: X_tr_stacked: A 3D array of training data, with first
%             dimension corresponding to feature, second dimension 
%             corresponding to training sample, and third dimension 
%             corresponding to class
%        quant_train: Vector of length L (L = # of classes) with lth entry
%             corresponding to the number of training points in the lth
%             class
%        d_vect: Vector of length L with lth entry corresponding to the
%                intrinsic dimension of the lth class (user-specified)
%        n: number of neighbors for use in local PCA (user-specified)
%
% Outputs: DICT_full_stacked: (m x n_train x L) 3D array with
%               DICT_full_stacked(:,:,l) corresponding to the decomposition 
%               dictionary for the lth class. Columns mod (d+1) correspond
%               to the original training vectors in the lth class with the
%               previous d columns the corresponding scaled and shifted
%               tangent vectors.
%          r_1: Scalar to be used in computation of neighborhood radius r. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[m,~,L] = size(X_tr_stacked);

% Initialize matrix that will store results:
DICT_full_stacked = zeros(m, (max(d_vect)+1)*max(quant_train), L); 

% Compute r_1:
    PREP_r_1 = zeros(max(quant_train), L);
    % Compute median distance between each training point and its (n+1)st 
    % nearest neighbor:
    for l=1:L
        X_tr_l = X_tr_stacked(:,1:quant_train(l),l);
        atria = nn_prepare(X_tr_l');
        for i=1:quant_train(l)
            x_i = X_tr_stacked(:,i,l);
            [~,D] = nn_search(X_tr_l',atria,x_i',n+2);
            PREP_r_1(i,l) = D(end);
        end
    end
    r_1 = median(PREP_r_1(:));

% Compute tangent vectors:
for l=1:L     
    DICT_Class_l = zeros(m,quant_train(l)*(d_vect(l)+1));
    X_tr_l = X_tr_stacked(:,1:quant_train(l),l);
    atria = nn_prepare(X_tr_l');
    for i=1:quant_train(l); 
        x_i = X_tr_l(:,i);
        DICT_x_i = zeros(m,d_vect(l)+1); % Will contain x_i and the shifted 
                           % and scaled tangent plane basis vectors at x_i.
            
        % Find the n-nearest neighbors of x_i in the same class:
            [IDX,DIST] = nn_search(X_tr_l', atria, x_i', n+2);
            Neighbors_i = X_tr_l(:,IDX);
            Neighbors_i(:,1) = []; % delete x_i

        % Compute epsilon_pca and weight matrix D_i:
            % Sets it to be the squared distance from its (n+1)st nearest
            % neighbor
            R = repmat(x_i,1,n+1);
            X_i = Neighbors_i - R; % neighbors centered around x_i
            
            epsilon_pca = (max(DIST))^2;
            
            % Delete the (n+1)st neighbor from set of neighbors:
            X_i(:,end) = []; 
            % Now the local covariance matrix is X_i*X_i';
            
            % Weight the neighbors of x_i according to their distances from
            % x_i:
            D_i = eye(n,n);
            % Can use Epanechnikov function 
            % (1-u^2)*I[0,1] (indicator function on [0,1]):
            for j=1:n 
                val = 1-((DIST(j)^2)/epsilon_pca);
                if val >= 0 && val <=1; % Need 0 <= val <=1
                    K_pca = val;
                else
                    K_pca = 0;
                end
                D_i(j,j) = sqrt(K_pca);
            end

            % Or use Gaussian weighting with parameter signa:
%             for j=1:k
%                 D_i(j,j) = exp(-norm(x_i-Neighbors_i(:,j))^2/sigma^2);
%             end
            
            B_i = X_i*D_i;
            % The weighted local covariance matrix is B_i*B_i';
            
        % Compute the first d eigenvectors of the weighted covariance
        % matrix using SVD:
        [eig_vects,~,~] = svd(B_i);
        DICT_x_i(:,1:d_vect(l)) = eig_vects(:,1:d_vect(l));

        % Add the training sample x_i to each vector in the basis after
        % scaling:
        c = rand(1)*r_1;
        DICT_x_i = c*DICT_x_i + repmat(x_i,1,d_vect(l)+1); 
          
       % Store the set of basis vectors for the approximate tangent plane 
       % at x_i on the manifold.
        DICT_Class_l(:,(i-1)*(d_vect(l)+1)+1: i*(d_vect(l)+1)) = DICT_x_i;
            
    end
  
    DICT_full_stacked(:,1:quant_train(l)*(d_vect(l)+1),l) = DICT_Class_l; 
       
 end
 

end

