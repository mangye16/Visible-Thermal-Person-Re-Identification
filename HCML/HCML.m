function [P, outV, outT, latent, eta, rankM, loss] = HCML(VisibleX, ThermalZ, Xlabels, Zlabels, options)

% This scipt is modified from MLAPG[1].
% [1] Shengcai Liao and Stan Z. Li, "Efficient PSD Constrained Asymmetric
% Metric Learning for Person Re-identification." In ICCV 2015.
%
% HCML Function: 
% Input:
%   <VisibleX>: features of gallery samples. Size: [n, d]
%   <ThermalZ>: features of probe samples. Size: [m, d]
%   <Xlabels>: class labels of the visiable samples
%   <Zlabels>: class labels of the thermal samples
%   [options]: optional parameters. A structure containing any of the
%   following fields:
%       maxIters: the maximal number of iterations. Default: 300.
%       tol: tolerance of convergence. Default: 1e-4.
%       L: initialization value of the Lipschitz constant, or 1/eta, where
%           eta is the step size. Default: 1 / 2^8.
%       gamma: a constant factor to adapt L or eta. Default: 2.
%       verbose: whether to print the learning details. Default: false
%
% Output:
%   P: the learned cross-modality projection matrix. M = P*P'. Size: [d,r]
%   V: the learned intra-modality projection matrix.  Size: [d,d]
%   T: the learned intra-modality projection matrix.  Size: [d,d]
%   latent: latent values of C_t.
%   eta: the working step size of each iteration.
%   rankM: the rank of M of each iteration.
%   loss: values of the loss function of each iteration.
% 

maxIters =300;
tol =1e-6;% 1e-4  ;
L = 1 / 2^8;
gamma = 2;
verbose = false;

if nargin >= 5 && ~isempty(options)
    if isfield(options,'maxIters') && ~isempty(options.maxIters) && isscalar(options.maxIters) && isnumeric(options.maxIters) && options.maxIters > 0
        maxIters = options.maxIters;
    end
    if isfield(options,'tol') && ~isempty(options.tol) && isscalar(options.tol) && isnumeric(options.tol) && options.tol > 0
        tol = options.tol;
    end
    if isfield(options,'L') && ~isempty(options.L) && isscalar(options.L) && isnumeric(options.L) && options.L > 0
        L = options.L;
    end
    if isfield(options,'gamma') && ~isempty(options.gamma) && isscalar(options.gamma) && isnumeric(options.gamma) && options.gamma > 1
        gamma = options.gamma;
    end
    if isfield(options,'verbose') && ~isempty(options.verbose) && isscalar(options.verbose) && islogical(options.verbose)
        verbose = options.verbose;
    end
end

if verbose == true
    fprintf('options.maxIters = %d.\n', maxIters);
    fprintf('options.tol = %g.\n', tol);
    fprintf('options.L = %g.\n', L);
    fprintf('options.gamma = %g.\n', gamma);
end

% [nGals, d] = size(unique()); % n
nGals = length(unique(Xlabels));
d = size(VisibleX,2);
nProbs = length(unique(Xlabels)); % m

if verbose == true
    fprintf('\nStart to learn HCML...\n\n');
end

t0 = tic;

% compute mean of each person 

[MgalX,   galLabels] = compute_mean_feature(VisibleX,Xlabels);
[MprobX, probLabels] = compute_mean_feature(ThermalZ,Zlabels);


exMgalX   = expand_mean(MgalX,  Xlabels) ; % for easy computation
exMprobX  = expand_mean(MprobX, Zlabels) ;


%% initilize preV and preT
% % initialize preV
% Qv = (VisibleX - exMgalX)'*(VisibleX - exMgalX);
% [EigenVv, Dv] = eig(Qv);
% latent = diag(Dv);
% [latent, index] = sort(latent);
% preV = EigenVv(:, index);
% 
% % initialize
% Qt = (ThermalZ - exMprobX)'*(ThermalZ - exMprobX);
% [EigenVt, Dt] = eig(Qt);
% latent = diag(Dt);
% [latent, index] = sort(latent);
% preT = EigenVt(:, index);
    
preV = eye(d);
preT = eye(d);

Y = bsxfun(@eq, galLabels(:), probLabels(:)');
Y = double(Y);
Y(Y == 0) = -1;
nPos = sum(Y(:) == 1);
nNeg = sum(Y(:) == -1);
W = zeros(nGals, nProbs);
W(Y == 1) = 1 / nPos;
W(Y == -1) = 1 / nNeg;
WY = W .* Y;

M = eye(d);
P = eye(d);
prevM = eye(d);
prevP2 = eye(d);
prevAlpha = 0;

eta = zeros(maxIters, 1);
rankM = zeros(maxIters, 1);
loss = zeros(maxIters, 1);

D = EuclidDist(MgalX * P, MprobX * P);
mu = mean(D(:));

D = D - mu;
D(Y == -1) = - D(Y == -1);
newF = Logistic(D); % log(1 + exp( D ));
newF = W(:)' * newF(:); % sum(sum( W .* log(1 + exp( Y .* (D - mu) )) ));

for iter = 1 : maxIters
    % update M
%     M = eye(d);
%     P = eye(d);
%     prevM = eye(d);
%     prevP2 = eye(d);
%    preV = eye(d); 
    
    newAlpha = (1 + sqrt(1 + 4 * prevAlpha^2)) / 2;
    S = M + (prevAlpha - 1) / newAlpha * (M - prevM);
    alpha = -(prevAlpha - 1) / newAlpha; % for prevP1
    beta = 1 + (prevAlpha - 1) / newAlpha; % for prevP2
    
    prevP1 = prevP2;
    prevP2 = P;
    prevM = M;
    prevF = newF;
    prevAlpha = newAlpha;
    
    galX  = MgalX  * preV;
    probX = MprobX * preT;
    
    D = alpha * EuclidDist(galX * prevP1, probX * prevP1) + beta * EuclidDist(galX * prevP2, probX * prevP2) - mu;
    D(Y == -1) = - D(Y == -1);
    T = WY ./ (1 + SafeExp( -D ));
    X = galX' * T * probX;
    gradF = galX' * bsxfun(@times, sum(T, 2), galX) - X - X' + bsxfun(@times, probX', sum(T, 1)) * probX;
    
    prevF_S = Logistic(D);
    prevF_S = W(:)' * prevF_S(:);
    
    while true
        [optFlag, M, P, latent, r, newF] = LineSearch(S, gradF, prevF_S, galX, probX, Y, W, L, mu);
        
        if ~optFlag
            L = gamma * L;
            if verbose == true
                fprintf('\tEta adapted to %g.\n', 1 / L);
            end
        else
            break;
        end
    end
    %% update V
    Dxx = EuclidDist(VisibleX * preV, exMgalX * preV) ; % for easy computation    
    Wv = 1 ./ (1 + SafeExp( -Dxx ));
    Wv = diag(Wv);      
    gradV1 = 1/length(Wv)* 2*(VisibleX' * diag(Wv) * VisibleX - VisibleX' * diag(Wv) * exMgalX - exMgalX' * diag(Wv) * VisibleX + exMgalX' * diag(Wv) * exMgalX) * preV;
    % two options
    gradV2 = (MgalX *S)' * bsxfun(@times, sum(T, 2), MgalX) * preV - (MgalX )' * T* (probX *S') - ((MgalX )' * T* (probX *S'))';
    
    % update V
    preV = preV - 0.1 * 0.95^iter * (0.2 * gradV1 + gradV2);
    
    %calculate the objective value
    Dxx = EuclidDist(VisibleX * preV, exMgalX * preV) ;
    prevF_V = sum(Logistic(diag(Dxx)))/length(Wv);
    
    %% update T
    Dzz = EuclidDist(ThermalZ * preT, exMprobX * preT) ; % for easy computation    
    Wt = 1 ./ (1 + SafeExp( -Dzz ));
    Wt = diag(Wt);      
    gradT1 = 1/length(Wt)* 2*(ThermalZ' * diag(Wt) * ThermalZ - ThermalZ' * diag(Wt) * exMprobX - exMprobX' * diag(Wt) * ThermalZ + exMprobX' * diag(Wt) * exMprobX) * preV;
    % two options
    gradT2 = (MprobX *S)' * bsxfun(@times, sum(T, 2), MprobX) * preT - (MprobX )' * T* (galX *S') - ((MprobX )' * T* (galX *S'))';
    
    % update V
    preT = preT - 0.1* 0.95^iter * (0.2 * gradT1 + gradT2);
    
    %calculate the objective value
    Dzz = EuclidDist(ThermalZ * preT, exMprobX * preT) ;
    prevF_T = sum(Logistic(diag(Dzz)))/length(Wt);
       
    if verbose == true
        fprintf('Iteration %d: Obj_All = %1.4f, Obj_M = %1.4f, Obj_V = %1.4f, Obj_T = %1.4f \n', iter, prevF_S + prevF_V*0.2 + prevF_T* 0.2,prevF_S, prevF_V, prevF_T);
    end
    
    eta(iter) = 1 / L;
    rankM(iter) = r;
    loss(iter) = prevF_S + prevF_V*0.2 + prevF_T* 0.2;
    
    if verbose == true && mod(iter, maxIters/10) == 0
        fprintf('Iteration %d: rankM = %d, lossF = %g. Elapsed time: %.3f seconds.\n', iter, rankM(iter), loss(iter), toc(t0));
    end
    
    outV = preV;
    outT = preT;
    if abs( (newF - prevF) / prevF ) < tol
        fprintf('Converged at iter %d. rankM = %d, loss = %g.\n', iter, rankM(iter), loss(iter));
        eta(iter+1 : end) = [];
        rankM(iter+1 : end) = [];
        loss(iter+1 : end) = [];
        
        break;
    end
end

if verbose == true
    fprintf('\nTraining time: %.3g seconds.\n', toc(t0));
    figure;
    subplot(2,2,1); plot(eta); title('Eta'); grid on;
    subplot(2,2,2); plot(rankM); title('Rank of M'); grid on;
    subplot(2,2,3); plot(loss); title('Loss'); grid on;
end

end


function [optFlag, M, P, latent, r, newF] = LineSearch(V, gradF, prevF_S, galX, probX, Y, W, L, mu)
    M = V - gradF / L;
    M =(M + M') / 2; % correct the rounding-off error to make sure M is symmetric
    [U, S] = eig(M);
    latent = max(0, diag(S));
    M = U * diag(latent) * U';
    r = sum(latent > 0);
    [latent, index] = sort(latent, 'descend');
    latent = latent(1:r);
    P = U(:, index(1:r)) * diag(sqrt( latent ));
    
    D = EuclidDist(galX * P, probX * P) - mu;
    D(Y == -1) = - D(Y == -1);
    newF = Logistic(D); % log(1 + exp( D ));
    newF = W(:)' * newF(:);
    
    diffM = M - V;
    optFlag = newF <= prevF_S + diffM(:)' * gradF(:) + L * norm(diffM, 'fro')^2 / 2;
end


function Y = SafeExp(X)
    Y = exp(X);
    Y(isinf(Y)) = realmax;
end


function Y = Logistic(X)
    Y = log(1 + exp(X));
    Y(isinf(Y)) = X(isinf(Y));
end

function [mean_feat, label] = compute_mean_feature(feat,labels)
    % feat: n * d,  d represents the feature dim
    % labels repsresent the label of each person
    % out: mean of each person n * d
    [n,d] = size(feat);
    
    unique_label = unique(labels);
    mean_feat = zeros(length(unique_label), d);
    for i = 1: length(unique_label)
        idx = find(labels==unique_label(i));
        mean_feat(i,:) = mean(feat(idx,:),1);      
    end
    label = unique_label;
end


function ex_mean = expand_mean(feat, labels)
 % feat: n*d
 % labels
 % output: len(labels) * d
 
 ex_mean = [];
 unique_label = unique(labels);
 for i = 1: length(unique_label)
     tmp_length = length(find(labels==unique_label(i)));
     
     tmp_feat = repmat(feat(i,:),tmp_length,1);
     ex_mean = [ex_mean ; tmp_feat];
 end
end