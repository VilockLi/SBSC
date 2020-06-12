clear all
warning off

addpath ../data;
addpath utils;

%% global parameters
lam_min = 1e-7;
bag_num = 1; % The bagging number.
nExper = 10;
method = 'tsc';

%% Load the data set and pre-processing
dataName = 'ZIPCODE';
[Y_norm, N_label, K, D, N] = loadRealData(dataName);

if strcmp(dataName, 'YaleB')
    n = 500; % Number of points in subsample.
    m = 30; % number of points to construct ridge regression for each subcluster
    dmax = 19; % The dimension threshold
    thre_vec = 5; %thresholding vector
elseif strcmp(dataName, 'MNIST')
    n = 500; 
    thre_vec = floor([n/1/K,n/1.5/K,n/2/K,n/2.5/K,n/3/K]);
    dmax = 29; 
    m = 100;
elseif strcmp(dataName, 'ZIPCODE')
    n = 500; 
    thre_vec = floor([n/1/K,n/1.5/K,n/2/K,n/2.5/K,n/3/K]);
    m = 50;
    dmax = 19;
end


%% parameters for first stage in SBSC

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

%% Algorithm begins
accr_vec = zeros(1, nExper);
time_vec = zeros(1, nExper);
nmi_vec = zeros(1, nExper);
accr_sub_vec = zeros(1, nExper);

for iExper = 1:nExper  
    rng(iExper);
    tic
    [label_final, accr_sub] = SBSC(Y_norm, K, bag_num, n, dmax, m, thre_vec, lam_min, method, optionalPar, N_label);
    accr = evalAccuracy(N_label, label_final);
    accr_sub_vec(iExper) = accr_sub;
    accr_vec(iExper) = accr;
    nmi_val = nmi(N_label, label_final);
    nmi_vec(iExper) = nmi_val;
    time = toc;
    time_vec(iExper) = time;
    dataformat = '%d-th experiment: accr = %f, accrSub = %f, nmi = %f, time = %f\n';
    dataValue = [iExper, accr, accr_sub, nmi_val,time];
    fprintf(dataformat, dataValue);
    results(iExper, :) = dataValue;
end

dataValue = mean(results, 1);
fprintf('\nAverage: accr= %f, accrSub= %f, nmi = %f,  time = %f, std_accr = %f, std_accr_sub = %f, std_nmi = %f,std_time = %f\n', ...
    [dataValue(2:end), std(accr_vec), std(accr_sub_vec), std(nmi_vec), std(time_vec)]);