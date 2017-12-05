function [P, latent, eta, rankM, loss] = MLAPG(galX, probX, galLabels, probLabels, options)
%% function [P, latent, eta, rankM, loss] = MLAPG(galX, probX, galLabels, probLabels, options)
% Cross-view logistic metric learning by accelerated proximal gradient
%
% Input:
%   <galX>: features of gallery samples. Size: [n, d]
%   <probX>: features of probe samples. Size: [m, d]
%   <galLabels>: class labels of the gallery samples
%   <probLabels>: class labels of the probe samples
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
%   P: the learned projection matrix. M = P*P'. Size: [d,r]
%   latent: latent values of C_t.
%   eta: the working step size of each iteration.
%   rankM: the rank of M of each iteration.
%   loss: values of the loss function of each iteration.
% 
% Example:
%     Please see Demo.m.
%
% Reference:
%   Shengcai Liao and Stan Z. Li, "Efficient PSD Constrained Asymmetric Metric 
%   Learning for Person Re-identification." In IEEE International Conference 
%   on Computer Vision (ICCV 2015), December 11-18, Santiago, Chile, 2015.
% 
% Version: 1.0
% Date: 2015-12-07
%
% Author: Shengcai Liao
% Institute: National Laboratory of Pattern Recognition,
%   Institute of Automation, Chinese Academy of Sciences
% Email: scliao@nlpr.ia.ac.cn


% maxIters = 200;
maxIters = 300;
tol = 1e-4;
L = 1 / 2^8;
gamma = 2;
verbose = true;

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

[nGals, d] = size(galX); % n
nProbs = size(probX, 1); % m

if verbose == true
    fprintf('\nStart to learn MLAPG...\n\n');
end

t0 = tic;

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

D = EuclidDist(galX * P, probX * P);
mu = mean(D(:));

D = D - mu;
D(Y == -1) = - D(Y == -1);
newF = Logistic(D); % log(1 + exp( D ));
newF = W(:)' * newF(:); % sum(sum( W .* log(1 + exp( Y .* (D - mu) )) ));

for iter = 1 : maxIters
    newAlpha = (1 + sqrt(1 + 4 * prevAlpha^2)) / 2;
    V = M + (prevAlpha - 1) / newAlpha * (M - prevM);
    alpha = -(prevAlpha - 1) / newAlpha; % for prevP1
    beta = 1 + (prevAlpha - 1) / newAlpha; % for prevP2
    
    prevP1 = prevP2;
    prevP2 = P;
    prevM = M;
    prevF = newF;
    prevAlpha = newAlpha;
    
    D = alpha * EuclidDist(galX * prevP1, probX * prevP1) + beta * EuclidDist(galX * prevP2, probX * prevP2) - mu;
    D(Y == -1) = - D(Y == -1);
    T = WY ./ (1 + SafeExp( -D ));
    X = galX' * T * probX;
    gradF = galX' * bsxfun(@times, sum(T, 2), galX) - X - X' + bsxfun(@times, probX', sum(T, 1)) * probX;
    
    prevF_V = Logistic(D);
    prevF_V = W(:)' * prevF_V(:);
    
    while true
        [optFlag, M, P, latent, r, newF] = LineSearch(V, gradF, prevF_V, galX, probX, Y, W, L, mu);
        
        if ~optFlag
            L = gamma * L;
            if verbose == true
                fprintf('\tEta adapted to %g.\n', 1 / L);
            end
        else
            break;
        end
    end
%     fprintf('Iteration %d: Objective_M = %d \n', iter, prevF_V);
    eta(iter) = 1 / L;
    rankM(iter) = r;
    loss(iter) = newF;
    
    if verbose == true && mod(iter, maxIters/10) == 0
        fprintf('Iteration %d: rankM = %d, lossF = %g. Elapsed time: %.3f seconds.\n', iter, rankM(iter), loss(iter), toc(t0));
    end
    
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


function [optFlag, M, P, latent, r, newF] = LineSearch(V, gradF, prevF_V, galX, probX, Y, W, L, mu)
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
    optFlag = newF <= prevF_V + diffM(:)' * gradF(:) + L * norm(diffM, 'fro')^2 / 2;
end


function Y = SafeExp(X)
    Y = exp(X);
    Y(isinf(Y)) = realmax;
end


function Y = Logistic(X)
    Y = log(1 + exp(X));
    Y(isinf(Y)) = X(isinf(Y));
end
