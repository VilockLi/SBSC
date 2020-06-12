function [Y_norm, N_label, K, D, N] = loadRealData(dataName, original)
if nargin < 2
  original = false;
end
    
if strcmp(dataName, 'YaleB')
    load YaleBCrop025.mat
    [~, ~, K] = size(Y);
    D = 500;
    selected_classes = 1:K;
    Y_raw = []; N_label = [] ;
    for m = 1:K       % forming the data
        Y_raw = [Y_raw Y(: , : , selected_classes(m))] ;   % X is the data
        N_label = [N_label m*ones(1,64)] ;         % Xlabels contains the true labels of the data
    end
    N = length(N_label);
    [P,~,~] = svd(Y_raw, 'econ') ;
    Y_norm = P(:,1:min(D, K * 64))' * Y_raw;  % In order to expedite the running time, the data is projected onto the span of dominant singular vectors.
    Y_norm = normc(Y_norm);
    if original
        Y_norm = normc(Y_raw);
        [D, ~] = size(Y_raw);
    end
    
elseif strcmp(dataName, 'MNIST')
    load MNIST_SC.mat
    label = MNIST_LABEL;
    Y_raw = MNIST_SC_DATA;
    clear MNIST_LABEL MNIST_SC_DATA
    
    K = length(unique(label)); % number of sub clusters.
    D = 500; % Dimension of ambient space.
    reduceDimension = @(data) dimReduction_PCA(data, D); % The PCA function.
    Y = [];
    for k = 0:K-1
        Y = [Y, Y_raw(:, label == k)];
    end
    Y_ori = Y;
    Y = reduceDimension(Y);
    [~, N] = size(Y);
    Y_norm = normc(Y);
    N_k = [];
    N_label = [];
    for k = 0:K-1
        N_k = [N_k,sum(label==k)];
        N_label = [N_label, ones(1, N_k(k + 1)) * k];
    end % N_label is the label of whole data set
    clear Y_raw label
    if original
        Y_norm = Y;
    end
    
elseif strcmp(dataName, 'ZIPCODE')
    data = feval('load','zip.test');
    N_label = data(:,1);
    Y_raw = data(:,2:end)';
    data = feval('load','zip.train');
    N_label = [N_label; data(:, 1)] + 1;
    Y_raw = [Y_raw, data(:, 2:end)'];
    clear data
    Y_norm = normc(Y_raw);
    [D, N] = size(Y_raw);
    K = length(unique(N_label)); % number of sub clusters.
    if original
        Y_norm = Y_raw;
    end
end