% 
% Mon Feb 15 05:56:52 2016 written by Chelsea Weaver
% This script runs the LPCA-SRC classification algorithm.
%
% Note that there is also a function version of this script.
%
% Required scripts/functions:
%
%   Normalize.m
%
%   l1-Minimization function:
%       HOMOTOPY: "L1 Homotopy: A MATLAB Toolbox for Homotopy Algorithms in 
%           L1 Norm Minimization Problems." Available at 
%           http://users.ece.gatech.edu/sasif/homotopy/
%       L1LS and OMP: L1-Benchmark package (Yang et al.). Available at 
%           http://www.eecs.berkeley.edu/~yang/software/l1benchmark/
%
%   Intrinsic dimension estimator (if specified):
%       DANCo: "Intrinsic Dimensionality Estimation Techniques" by Gabriele
%           Lombardi. Available on MATLAB File Exchange at 
%           http://www.mathworks.com/matlabcentral/fileexchange/40112-intrinsic-dimensionality-estimation-techniques

%%
% clear
% clc

%% USER INPUT:

% Load data. Must specify:
    % A_train (m x n_train matrix of training samples)
    % A_test (m x n_test matrix of test samples)
    % lab (training label vector)
    % ulab_GT (ground truth label of test samples)
    
% tot_num = 100; K = 8; noise_level = 0.01;
% run Data_Gen_Spherical_Sine % My synthetic data set
    lab = LAB;
    ulab_GT = ULAB_GT;
    
% Set parameters:

% Specify l1-minimization algorithm:
alg_opt = 1; % 1 ~ HOM, 2 ~ L1LS, 3 ~ OMP

if alg_opt == 1 || 2
    % Specify sparsity/accuracy trade-off in l1-minimization:
    lambda = 1e-3;
elseif alg_opt == 3
    % Specify number of nonzero coefficients:
    Te = 2;
end

% Specify whether or not data has occlusion:
occ_on = 0; % 0 ~ no occlusion, 1 ~ occlusion

% How is the intrinsic class parameter d set?
d_opt = 0; % 0 ~ manually, % 1 ~ per class using DANCo algorithm.

% if d_opt == 0
%     % Manually specify d_vect as a Kx1 dimensional vector of intrinsic
%     % class dimensions:
%     d_vect = ones(K,1);
% end

% Number of neighbors in Local PCA:
% number of neighbors must be >= d and <= min(quant_train)-2
%n = 3;

tic

%% START OFFLINE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extract information from training data:

[m,n_train] = size(A_train);

K = length(unique(lab));

quant_train = zeros(K,1);
for l=1:K
    quant_train(l) = nnz(lab==l);
end

% Normalize training data:
A_train_norm = Normalize(A_train);

% Set d_vect if needed:
if d_opt == 1; % if using an estimation algorithm:
    d_vect = zeros(K,1);
    for l=1:K
        class_l_index = lab==l;
        A_train_l = A_train(:,class_l_index);
        d_vect(l) = DANCo(A_train_l);
    end
end

% Stack training data by class:
A_train_stacked = zeros(m,max(quant_train),K);
for l=1:K
    class_l_index = lab==l; 
    A_train_stacked(:,1:quant_train(l),l) = A_train_norm(:,class_l_index);
end

% Compute tangent vectors and neighborhood radius parameter r:
[ DICT_full_stacked, r_1] = ...
    Local_PCA( A_train_stacked, quant_train, d_vect, n );

% Write DICT_full_stacked as a 2D matrix and compute label vector as well
% as an index vector of training points in DICT_full:
DICT_full = zeros(m,dot(quant_train,(d_vect+1)));
lab_DICT_full = zeros(1,dot(quant_train,(d_vect+1)));
train_pt_ind = zeros(n_train,1); 
% Note that lab_DICT_full(train_pt_ind(i)) will retrieve the class of the
% ith training point in A_train.

count_1 = 1;
count_2 = 1;
for l=1:K
    
    DICT_full(:,count_1:count_1+quant_train(l)*(d_vect(l)+1)-1) = ...
        DICT_full_stacked(:,1:quant_train(l)*(d_vect(l)+1),l);
    lab_DICT_full(count_1:count_1+quant_train(l)*(d_vect(l)+1)-1) = ...
        l*ones(1,quant_train(l)*(d_vect(l)+1));
    train_pt_ind(count_2:count_2+quant_train(l)-1) = ...
        count_1+d_vect(l):d_vect(l)+1:count_1+(d_vect(l)+1)*quant_train(l)-1;
    
    count_1 = count_1 + quant_train(l)*(d_vect(l)+1);
    count_2 = count_2 + quant_train(l);
end

DICT_full = Normalize(DICT_full);

%% Start Online Phase %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extract information from test data:

[~,n_test] = size(A_test);

% Normalize test data:
A_test_norm = Normalize(A_test);

% Compute average number of columns in dictionary after r constraint:
len_DICT = zeros(n_test,1);

% Store HOMOTOPY statistics:
if alg_opt == 1
    % Number of iterations in HOMOTOPY algorithm:
    ITER = zeros(n_test,1);
    % Computation time of HOMOTOPY algorithm:
    TIME = zeros(n_test,1);
end

% Store sparsity of coefficient vector:
SPAR = zeros(n_test,1);

% Check variance in classes represented in D_y:
VAR = zeros(n_test,1);

% Initialize test label vector:
ulab_LPCA_SRC = zeros(n_test,1);

% Begin classification:

for j=1:n_test
    y = A_test_norm(:,j);
    
    DICT_y = zeros(size(DICT_full));
    lab_DICT_y = zeros(length(lab_DICT_full));
    
    % Compute distances between the test point and each training point:
    dist_vects_pos = repmat(y,1,n_train)-A_train_norm;
    DIST_pos = sqrt(sum(dist_vects_pos.^2,1))';
    
    dist_vects_neg = repmat(y,1,n_train)+A_train_norm;
    DIST_neg = sqrt(sum(dist_vects_neg.^2,1))';
    
    % Compute minimum distance from y to a class rep
    r_2 = min(min(DIST_pos),min(DIST_neg)); 
    
    % Set neighborhood radius parameter:
    r = max(r_1,r_2);
    
    % Amend dictionary to include only training points (and their ...
    % corresponding tangent basis vectors) that are within r of the ...
    % test point y:
    
    ind_pos = find(DIST_pos <= r);
    ind_neg = find(DIST_neg <= r);
    ind = unique([ind_pos; ind_neg]);  
    ind = reshape(ind,length(ind),1);

    len = length(ind);    
    count = 1;
    close_train_pts = train_pt_ind(ind); 
    % index of nearby training points in DICT_full
    for i=1:len
        class_l_index = lab_DICT_full(close_train_pts(i));
        DICT_y(:,count:count+d_vect(class_l_index)) = ...
            DICT_full(:,close_train_pts(i)-d_vect(class_l_index):...
            close_train_pts(i));
        lab_DICT_y(count:count+d_vect(class_l_index)) = ...
            class_l_index*ones(d_vect(class_l_index)+1,1);
        count = count + d_vect(class_l_index)+1;
    end
    
    % Remove empty columns of DICT_y and entries of lab_DICT_y:
    DICT_y = DICT_y(:,1:count - 1);
    lab_DICT_y = lab_DICT_y(1:count - 1);
    
    % Store number of columns in dictionary:
    len_DICT(j) = size(DICT_y,2);
    VAR(j) = length(unique(lab_DICT_y));
    
    % l1-Minimization:%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if alg_opt == 1
        if occ_on == 1
            B = [DICT_y,eye(m)];
            % BPDN_homotopy_function_old (v2 of L1-Homotopy toolbox)
            in.tau = lambda; in.x_orig = zeros(m,1); in.record = 0;
            out = BPDN_homotopy_function_old(B,y,in);
            x = out.x_out; % output solution
            ITER(j) = out.iter; % number of homotopy iterations used
            TIME(j) = out.time; % computation time
            SPAR(j) = nnz(x);
        else
            % BPDN_homotopy_function_old (v2 of L1-Homotopy toolbox)
            in.tau = lambda; in.x_orig = zeros(m,1); in.record = 0;
            out = BPDN_homotopy_function_old(DICT_y,y,in);
            x = out.x_out; % output solution
            ITER(j) = out.iter; % number of homotopy iterations used
            TIME(j) = out.time; % computation time
            SPAR(j) = nnz(x);
        end
    elseif alg_opt == 2
        if occ_on == 1
            x = SolveL1LS([DICT_y,eye(m)],y, 'lambda', lambda);
            SPAR(j) = nnz(x);
        else
            x = SolveL1LS(DICT_y,y, 'lambda', lambda);
            SPAR(j) = nnz(x);            
        end
    elseif alg_opt == 3
        in.Te = Te;
        in.thresh = tol;
        out = OMP_function(y,DICT_y,in);
        x = out.x_out;
        SPAR(j) = nnz(x);        
    end
    
    % Compute class error for each class:
    ERR_y = zeros(1,K);
    for l=1:K
        coeff = zeros(len_DICT(j),1);
        ind_l = find(lab_DICT_y==l);
        if isempty(ind_l)==0 % if DICT_j contains vectors from class l
            coeff(ind_l) = x(ind_l);
            if occ_on == 1
                e_hat = x(len_DICT(j)+1:end);
                ERR_y(l) = norm(y-e_hat-DICT_y*coeff);
            else
                ERR_y(l) = norm(y-DICT_y*coeff);
            end
        else
            ERR_y(l) = inf;
        end
    end
    
    [error_LPCA_SRC,ulab_LPCA_SRC(j)] = min(ERR_y);
    
end

%% Compute accuracy:

ACC_LPCA_SRC = nnz(ulab_LPCA_SRC == ulab_GT)/n_test;

%% Compute statistics:

% Computational time:
t_LPCA_SRC = toc;

% Average number of columns in DICT_y:
Avg_LEN = mean(len_DICT);

% Average number of HOMOTOPY iterations:
Avg_Hom_Iter = mean(ITER);

% Average HOMOTOPY computational time:
Avg_Hom_Time = mean(TIME);

% Average sparsity of l1-minimized coefficient vector x:
Avg_Sparsity = mean(SPAR);

Var_in_DICT_y = mean(VAR);

%% Optional: Plot classification results:

%     DATA = A;
%     Y = DATA';
% 
%     Y_train = Y(1:LB,1:3);
%     Y_test = Y(LB+1:LB+ULB,1:3);
% 
%     % Plot correctly classified points (blue):
%     CORRECT = find(ulab_LPCA_SRC == ULAB_GT);
%     CORRECT_COORDS = Y_test(CORRECT,:);
%     x1 = CORRECT_COORDS(:,1); y1 = CORRECT_COORDS(:,2);
%     z1=CORRECT_COORDS(:,3);
% 
%     scatter3(x1,y1,z1,'k');
% 
%     % Plot incorrectly classified points (red):
%     INCORRECT = find(ulab_LPCA_SRC ~= ULAB_GT);
%     INCORRECT_COORDS = Y_test(INCORRECT,:);
%     x2 = INCORRECT_COORDS(:,1); y2 = INCORRECT_COORDS(:,2);
%     z2 = INCORRECT_COORDS(:,3);
% 
%     hold on
%     scatter3(x2,y2,z2,'k','filled');
%     
%     % Plot tangent vectors:
%     for l=1:K
%         class = find(lab_DICT_full==l);
%         DICT_full_class = DICT_full(:,class);
%         Tan_Vects_class = DICT_full_class(:,1:d_vect(1)+1:end);
%         hold on
%         scatter3(Tan_Vects_class(1,:),Tan_Vects_class(2,:),Tan_Vects_class(3,:),colors(l),'d','filled');
%     end
    
    
    
    
    
