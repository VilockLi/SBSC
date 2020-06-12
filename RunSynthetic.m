%% Load the synthetic data set and pre-processing
clear all
warning off
addpath ../data

% warning off
K = 20; % number of subspaces
D = 30; % ambient dimension
di = 5; % dimension of subspaces
lam_min = 1e-7;
n = 200; % Number of points in subsample
m = 10; % number of points to construct ridge regression for each subcluster
dmax = 9; % The dimension threshold
thre_vec = [20];
bag_num = 1; % The bagging number



%% parameters for first step in SBSC
method = 'tsc';

if strcmp(method, 'tsc')
    optionalPar = [];
elseif strcmp(method, 'ssc')
    optionalPar.lambda = [1e-7];
    optionalPar.tolerance = [1e-4];
    optionalPar.isNonnegative = false;
    optionalPar.maxIteration = 2000;
elseif strcmp(method, 'dsc')
    optionalPar.norm = 2;
    optionalPar.mu = 3;
    optionalPar.gamma = 0.03;
    optionalPar.itr = 100;
end


%% Synthetic data set.
Ni = 1000; % Number of points in each subspace.
N = Ni*K; % Number of total data set.

%% We run 10 experiments
nExper = 10;
% labelName = strcat('N_label',num2str(N));
% load(labelName)
N_label = zeros(N, 1);
for k = 1:K
    N_label(1 + (k-1)*Ni:k * Ni) = k;
end

for noise_level = 0.1:0.1:0.5
    for iExper = 1:nExper
        dataName = strcat('Yraw_', 'Noise', ...
            num2str(noise_level), '_N', num2str(N), 'rng', num2str(iExper),'.mat');
        load(dataName)
        Y_norm = normc(Y_raw);
        rng(iExper);
        tic
        [label_final, accr_sub] = SBSC(Y_norm, K, bag_num, n, dmax, m, thre_vec, lam_min, method, optionalPar, N_label);
        accr = evalAccuracy(N_label, label_final);
        nmi_val = nmi(N_label, label_final);
        time = toc;
        dataformat = '%d-th experiment: accr = %f, accrSub = %f, nmi = %f, time = %f\n';
        dataValue = [iExper, accr, accr_sub, nmi_val,time];
        fprintf(dataformat, dataValue);
    end
end

