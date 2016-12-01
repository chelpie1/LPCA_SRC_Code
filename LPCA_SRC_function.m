function [ ulab, t, Avg_LEN, Accuracy, Avg_Hom_Iter, Avg_Hom_Time ] ...
    = LPCA_SRC_function( X_train, lab, X_test, n, varargin )

% Mon Feb 15 04:36:59 2016 written by Chelsea Weaver
%
% LPCA_SRC_function performs classification using the local PCA
% modification of Sparse Representation-based Classification (Wright, et
% al. 2009).
%
% Note that there is also a script version of this function titled
% "LPCA_SRC.m"
%
% Inputs: X_tr: Matrix of training data with rows corresponding to 
%               features and columns corresponding to samples
%         X_te: Matrix of test data
%         lab: Vector of training labels
%         varargin: 'test_label': Ground truth label for test data
%                   'optimization_algorithm': Specify which
%                       l1-mininimization algorithm to use. 0 ~ HOMOTOPY,
%                       1 ~ L1LS, 2 ~ OMP
%                   'algorithm_parameter': error/sparsity tradeoff parameter
%                       lambda in HOMOTOPY and L1LS; # of nonzero 
%                       coefficients in OMP
%                   'occlusion': Specify whether to use or not use occlusion 
%                       version of algorithm. 0 ~ no occlusion, 1 ~ occlusion   
%                   'd_option': Specify how to set d. 0 ~ manually, 1 ~
%                       using Danco algorithm
%                   'd_vect': Manual specification of intrinsic class
%                       dimension for each class (Kx1 vector for K = # of classes)
%
% Outputs: ulab_LPCA_SRC: Vector of class assignments
%          t: runtime
%          Avg_LEN: Average number of columns in dictionary
%    OPTIONAL OUTPUTS:
%          Accuracy: Accuracy of classification given test labels
%          Avg_Hom_Iter: Average number of HOMOTOPY iterations           
%          Avg_Hom_Time: Average HOMOTOPY runtime
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


% Parse the optional inputs.
if (mod(length(varargin), 2) ~= 0 ),
    error(['Extra Parameters passed to the function ''' mfilename ''' must be passed in pairs.']);
end
parameterCount = length(varargin)/2;

for parameterIndex = 1:parameterCount,
    parameterName = varargin{parameterIndex*2 - 1};
    parameterValue = varargin{parameterIndex*2};
    switch lower(parameterName)
        case 'test_label'
            ulab_GT = parameterValue;
        case 'optimization_algorithm'
            alg_opt = parameterValue;
        case 'algorithm_parameter'
            alg_par = parameterValue;
        case 'occlusion'
            occ_on = parameterValue;
        case 'd_option'
            d_opt = parameterValue;
        case 'd_vect'
            d_vect = parameterValue;
        otherwise
            error(['The parameter ''' parameterName ''' is not recognized by the function ''' mfilename '''.']);
    end
end
clear varargin

if ~exist('alg_opt','var') 
    alg_opt = 1; % HOMOTOPY
end

if ~exist('alg_par','var')
    if alg_opt == 1 || 2
        alg_par = 1e-3; % error/sparsity tradeoff parameter in HOM/L1LS
    elseif alg_opt == 3
        alg_par = 10; % number of OMP iterations = # of nonzero coefficients
    else
        error('Not a valid algorithm parameter.\n')
    end
end

if ~exist('occ_on','var')
    occ_on = 0;
end

if ~exist('d_opt','var')
    d_opt = 0;
end

tic

%% START OFFLINE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extract information from training data:

[m,n_tr] = size(X_tr);

L = length(unique(lab)); % # of classes

quant_train = zeros(L,1); % # of training points in each class
for l=1:L
    quant_train(l) = nnz(lab==l);
end

if ~exist('d_vect','var')
    if d_opt == 0
        d_vect = ones(L,1); % Sets d_vect to be all 1's by default.
    elseif d_opt == 1
        warning('Ignoring user specification: Setting d_vect automatically')
    end
end

% Set d_vect if needed:
if d_opt == 1; % if using an estimation algorithm:
    d_vect = zeros(L,1);
    for l=1:K
        class_l_index = lab==l;
        X_tr_l = X_tr(:,class_l_index);
        d_vect(l) = DANCo(X_tr_l);
    end
end

% Normalize training data:
X_tr_norm = Normalize(X_tr);

% Stack training data by class:
X_tr_stacked = zeros(m,max(quant_train),L);
for l=1:L
    class_l_index = lab==l; 
    X_tr_stacked(:,1:quant_train(l),l) = X_tr_norm(:,class_l_index);
end

% Compute tangent vectors and neighborhood radius parameter r:
[ DICT_full_stacked, r_1] = ...
    Local_PCA( X_tr_stacked, quant_train, d_vect, n );

% Write DICT_full_stacked as a 2D matrix and compute label vector as well
% as an index vector of training points in DICT_full:
DICT_full = zeros(m,dot(quant_train,(d_vect+1)));
lab_DICT_full = zeros(1,dot(quant_train,(d_vect+1)));
train_pt_ind = zeros(n_tr,1); 
% Note that lab_DICT_full(train_pt_ind(i)) will retrieve the class of the
% ith training point in X_tr.

count_1 = 1;
count_2 = 1;
for l=1:L
    
    DICT_full(:,count_1:count_1+quant_train(l)*(d_vect(l)+1)-1) = ...
        DICT_full_stacked(:,1:quant_train(l)*(d_vect(l)+1),l);
    lab_DICT_full(count_1:count_1+quant_train(l)*(d_vect(l)+1)-1) = ...
        l*ones(1,quant_train(l)*(d_vect(l)+1));
    train_pt_ind(count_2:count_2+quant_train(l)-1) = ...
        count_1+d_vect(l):d_vect(l)+1:count_1+(d_vect(l)+1)*quant_train(l)-1;
    
    count_1 = count_1 + quant_train(l)*(d_vect(l)+1);
    count_2 = count_2 + quant_train(l);
end

%% Start Online Phase %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extract information from test data:

[~,n_te] = size(X_te);

% Normalize test data:
X_te_norm = Normalize(X_te);

% Compute average number of columns in dictionary after r constraint:
len_DICT = zeros(n_te,1);

% Store HOMOTOPY statistics:
if alg_opt == 1
    % Number of iterations in HOMOTOPY algorithm:
    ITER = zeros(n_te,1);
    % Computation time of HOMOTOPY algorithm:
    TIME = zeros(n_te,1);
end

% Initialize test label vector:
ulab = zeros(n_te,1);

% Begin classification:

for j=1:n_te
    y = X_te_norm(:,j);
    
    DICT_y = zeros(size(DICT_full));
    lab_DICT_y = zeros(length(lab_DICT_full));
    
    % Compute distances between the test point and each training point:
    dist_vects_pos = repmat(y,1,n_tr)-X_te;
    DIST_pos = sqrt(sum(dist_vects_pos.^2,1))';
    
    dist_vects_neg = repmat(y,1,n_tr)+X_te;
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
    
    % l1-Minimization:%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Normalize dictionary:
    DICT_y = Normalize(DICT_y);
    
    if alg_opt == 1
        if occ_on == 1
            B = [DICT_y,eye(m)];
            % BPDN_homotopy_function_old (v2 of L1-Homotopy toolbox)
            in.tau = alg_par; in.x_orig = zeros(m,1); in.record = 0;
            out = BPDN_homotopy_function_old(B,y,in);
            x = out.x_out; % output solution
            ITER(j) = out.iter; % number of homotopy iterations used
            TIME(j) = out.time; % computation time
        else
            % BPDN_homotopy_function_old (v2 of L1-Homotopy toolbox)
            in.tau = alg_par; in.x_orig = zeros(m,1); in.record = 0;
            out = BPDN_homotopy_function_old(DICT_y,y,in);
            x = out.x_out; % output solution
            ITER(j) = out.iter; % number of homotopy iterations used
            TIME(j) = out.time; % computation time
        end
    elseif alg_opt == 2
        if occ_on == 1
            x = SolveL1LS([DICT_y,eye(m)],y, 'lambda', alg_par);
        else
            x = SolveL1LS(DICT_y,y, 'lambda', alg_par);
        end
    elseif alg_opt == 3
        in.Te = alg_par;
        in.thresh = tol;
        out = OMP_function(y,DICT_y,in);
        x = out.x_out;
    end
    
    % Compute class error for each class:
    ERR_y = zeros(1,L);
    for l=1:L
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
    
    [~,ulab(j)] = min(ERR_y);
end

%% Compute accuracy:

if exist('ulab_GT','var')
    Accuracy = nnz(ulab == ulab_GT)/n_te;
end

%% Compute statistics:

% Computational time:
t = toc;

% Average number of columns in DICT_y:
Avg_LEN = mean(len_DICT);

if alg_opt == 1
    % Average number of HOMOTOPY iterations:
    Avg_Hom_Iter = mean(ITER);

    % Average HOMOTOPY computational time:
    Avg_Hom_Time = mean(TIME);
end


end

