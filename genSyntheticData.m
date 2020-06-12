function [Y_raw, s, U_cell, X, noise_term] = genSyntheticData(D, K, Ni, di, noise_level)
% This function generates synthetic data set with noises
% Input:
%     D:  the dimension of ambient space
%     K:  # of subspaces
%     Ni: # of data points of each subspace, this is a vector of K
%     positive integers
%     di: the dimension of each subspace, this is a vector of K positive
%     integers
%     noise_level (optional): the noise level, between 0 and 1
% Output:
%     Y_raw: the data without normalization of columns.   
%     s: label of each data point, this is a vector of Ni * K positive
%     integers.
%     U_cell: the cell of orthogonal bases
%     X: the noiseless data 
%     noise_term: the noise matrix

if nargin < 5
  noise_level = 0;
end

if length(di) == 1
    di = repmat(di, 1, K);
end


if length(Ni) == 1
    Ni = repmat(Ni, 1, K);
end

X = zeros(D, sum(Ni)); s = zeros(1, sum(Ni)); idx = 0;
U_cell = cell(1);
for in = 1:K
    Xtmp = randn(D, D);
    [Utmp, ~, ~] = svds(Xtmp, di(in)); % generate a random subspace
    Vtmp = randn(di(in), Ni(in));
    U_cell{in} = Utmp;
    Xtmp = normc(Utmp * Vtmp); % generate random points in subspace and normalize the columns
    X(:, idx + 1 : idx + Ni(in)) = Xtmp;
    s(idx+1 : idx+Ni(in)) = in;
    idx = idx + Ni(in);
end

norm_matrix = normrnd(0, 1, [D, sum(Ni)]);
R = chi2rnd(di(1), 1, sum(Ni));
idx = [0, cumsum(Ni)];
for i = 1:length(di)
    R(1 + idx(i):idx(i + 1)) = R(1 + idx(i):idx(i + 1)) / di(i);
end
R = repmat(R,D,1);
sigma = sqrt(noise_level ./ (D * di ./ (di - 2)));
noise_term = norm_matrix ./ sqrt(R);
for i = 1:length(di)
    noise_term(:, 1 + idx(i):idx(i + 1)) = noise_term(:, 1 + idx(i):idx(i + 1)) * sigma(i);
end

Y_raw = X + noise_term; % The raw data.