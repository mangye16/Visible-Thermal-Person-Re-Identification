%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Demo code on the RegDB datasets for the following paper:
%%%
%%% Mang Ye, Xiangyuan Lan, Jiawei Li and Pong C Yuen. 
%%% "Hierarchical Discriminative Learning for Visible Thermal Person Re-Identification". 
%%% In AAAI 2018.
%%%
%%% Contact: mangye@comp.hkbu.edu.hk
%%% Last updated: 2017/11/20
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
% add path
addpath('lib\');

num_id = 412;
pcaDims = 600;

load ../data/RegDB_split.mat;
trial = 1 ;

feat_dir = 'feature/';

%% load features
path = [feat_dir 'train_color_iter_' num2str(trial) '.mat'];
tra_co_feature = h5read(path,'/feature');

path = [feat_dir 'train_thermal_iter_' num2str(trial) '.mat'];
tra_th_feature = h5read(path,'/feature');

path = [feat_dir 'test_color_iter_' num2str(trial) '.mat'];
tst_co_feature = h5read(path,'/feature');

path = [feat_dir 'test_thermal_iter_' num2str(trial) '.mat'];
tst_th_feature = h5read(path,'/feature');
    
%% Training HCML
% normalization is important !!
% for color images   
% sum_val_col = sqrt(sum(tra_co_feature.^2));
% for n = 1:size(tra_co_feature, 1)
%     tra_co_feature(n, :) = tra_co_feature(n, :)./sum_val_col;
% end
% % for thermal images
% sum_val_th = sqrt(sum(tra_th_feature.^2));
% for n = 1:size(tra_th_feature, 1)
%     tra_th_feature(n, :) = tra_th_feature(n, :)./sum_val_th;
% end

% generate label
label = repmat(1: num_id/2,[10 1]);
label = reshape(label,2060,1);

X = [tra_co_feature'; tra_th_feature']; % [n, d]
mu = mean(X);
W = PCA(X, pcaDims);
clear X

tra_co_feature = bsxfun(@minus,  tra_co_feature', mu) * W;
tra_th_feature = bsxfun(@minus, tra_th_feature', mu) * W; 

% Baseline
% [P, latent, eta, rankM, loss] = MLAPG(tra_co_feature, tra_th_feature, label, label);  
% T = eye(pcaDims);
% V = eye(pcaDims);

%% Main function train HCML 
[P, T, V, latent, eta, rankM, loss] = HCML(tra_th_feature, tra_co_feature, label, label);

%% Testing  
% sum_val_col = sqrt(sum(tst_co_feature.^2));
% for n = 1:size(tst_co_feature, 1)
%     tst_co_feature(n, :) = tst_co_feature(n, :)./sum_val_col;
% end
% sum_val_th = sqrt(sum(tst_th_feature.^2));
% for n = 1:size(tst_th_feature, 1)
%     tst_th_feature(n, :) = tst_th_feature(n, :)./sum_val_th;
% end
    
% evaluation    
tst_co_feature = bsxfun(@minus, tst_co_feature', mu) * W ; 
tst_th_feature = bsxfun(@minus, tst_th_feature', mu) * W ; 
  label = repmat(1: num_id/2,[10 1]);
label = reshape(label,2060,1);
fprintf('L2 distance: ......................\n')
dist = EuclidDist(tst_th_feature,tst_co_feature);
cmc = EvalCMC( -dist, label', label', 100 );
mAP = EvalMAP( -dist, label', label' );
fprintf(' Rank-1, Rank-5, Rank-10, Rank-15, Rank-20\n');
fprintf(' %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n', cmc([1,5,10,15,20]) * 100);
fprintf('MAP is %5.2f%%, \n\n', mAP * 100);   

fprintf('HCML: ......................\n')
tst_co_feature = tst_co_feature *  V *P;
tst_th_feature = tst_th_feature *  T *P;
dist = EuclidDist(tst_th_feature,tst_co_feature);
cmc = EvalCMC( -dist, label', label', 100 );
mAP = EvalMAP( -dist, label', label' );
fprintf(' Rank-1, Rank-5, Rank-10, Rank-15, Rank-20\n');
fprintf(' %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n', cmc([1,5,10,15,20]) * 100);
fprintf('    MAP is %5.2f%%, \n\n', mAP * 100);     


