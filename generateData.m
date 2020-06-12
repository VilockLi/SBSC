%% Generate the synthetic data
clear all
warning off
% warning off

K = 20; % number of subspaces
D = 30; % ambient dimension
di = 5; % dimension of subspaces
nExper = 10;

Ni_vec = [100];
for i = 1:9
    temp = Ni_vec(i) * 2;
    Ni_vec = [Ni_vec; temp];
end

for noise_level = 0.2
    for i = 1:length(Ni_vec)
        Ni = Ni_vec(i);
        N = Ni*K;
        for iExper = 1:nExper
            rng(iExper);
            [Y_raw, N_label, U_cell, Y_noiseless, noise_term] = genSyntheticData(D, K, Ni, di, noise_level);
            dataName = strcat('Yraw_', 'Noise', ...
                num2str(noise_level), '_N', num2str(N), 'rng', num2str(iExper),'.mat');
            cd ../data
            save(dataName,'Y_raw')
            labelName = strcat('N_label',num2str(N));
            save(labelName, 'N_label')
            cd ../SBSC
        end
    end
end
