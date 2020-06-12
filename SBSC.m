function [label_final, mean_sub_accr] = SBSC(Y_norm, K, bag_num, n, dmax, m, thre_vec, lam_min, method, optionalPar, N_label)
[D, N] = size(Y_norm);
label_test = zeros(N, bag_num);
label_final = [];
score_fit = zeros(bag_num,1);
groups_sub_vec = zeros(bag_num, 1);

if nargin < 8
  method = 'tsc';
end

for b_num = 1:bag_num
    active_set = 1:N;
    ind_sub = sort(randperm(length(active_set), n));
    Y_sub = Y_norm(:, ind_sub); % The subset.
    if strcmp(method, 'tsc')
        similarity_all = abs(Y_sub' * Y_norm); % Calculate the inner products.
        for i = 1:n
            similarity_all(i, ind_sub(i)) = 1.01;
        end
    end
    if strcmp(method, 'ssc')
        similarity_all = zeros(n, N);
        for i = 1:n
            del_idx = ind_sub(i);
            if del_idx == 1
                [beta_temp, ~] = SolveHomotopy(Y_norm(:,2:end), Y_norm(:, 1), ...
                    'maxIteration', optionalPar.maxIteration,...
                    'isNonnegative', optionalPar.isNonnegative, ...
                    'lambda', optionalPar.lambda, ...
                    'tolerance', optionalPar.tolerance);
                beta_temp = [max(abs(beta_temp)) + 0.1; abs(beta_temp)];
            else
                [beta_temp, ~] = SolveHomotopy([Y_norm(:, 1:del_idx - 1) Y_norm(:, del_idx + 1:end)], Y_norm(:, del_idx), ...
                    'maxIteration', optionalPar.maxIteration,...
                    'isNonnegative', optionalPar.isNonnegative, ...
                    'lambda', optionalPar.lambda, ...
                    'tolerance', optionalPar.tolerance);
                beta_temp = abs([beta_temp(1:del_idx - 1); max(abs(beta_temp)) + 0.1; beta_temp(del_idx:end)]);
            end
            similarity_all(i, :) = beta_temp'; 
        end
    end
    if strcmp(method, 'dsc')
        similarity_all = abs(Solver_DSC(Y_norm, Y_sub, optionalPar.mu, optionalPar.gamma , optionalPar.itr , optionalPar.norm))';
        for i = 1:n
            similarity_all(i, ind_sub(i)) = max(similarity_all(i, :)) + 0.01;
        end
    end
    % Find sub-clusters.
    fac = floor(n / 20);
    ind_mat = [];
    for i = 1:fac
        [~, index] = sort(similarity_all(1 + (i - 1) * 20:i * 20, :),2,'descend');
        ind_mat = [ind_mat; index(:, 1:dmax + 1)];
    end
    if mod(n, fac)
        [~, index] = sort(similarity_all(1 + fac * 20:n, :),2,'descend');
        ind_mat = [ind_mat; index(:, 1:dmax + 1)];
    end
    % Construct the sub-clusters
    C_cell = cell(1);
    for i = 1:n
        C_cell{i} = Y_norm(:, ind_mat(i, :));
    end
    % Find lambda
    lam_vec = [];
    for i = 1:length(C_cell)
        temp_vec = svd(C_cell{i} * C_cell{i}');
        temp_vec = temp_vec(1:dmax + 1);
        temp_vec(temp_vec<1e-7) = [];
        lam_vec = [lam_vec sqrt(sum((1 ./ temp_vec) .^ 2))];
    end
    lam = max((D * max(lam_vec)) ^ (-1), lam_min);
    Ridge_mat = cell(1);
    if D <= dmax + 1
        for i = 1:n
            Ridge_mat{i} = inv(C_cell{i} * C_cell{i}' + lam * eye(D));
        end
        % Construct the affinity matrix
        dis = zeros(n, n);
        for i = 1:n-1
            for j = i+1:n
                temp_resi_1 = norm(lam * Ridge_mat{j} * C_cell{i}, 'fro');
                temp_resi_2 = norm(lam * Ridge_mat{i} * C_cell{j}, 'fro');
                dis(i,j) = exp(-0.5 * (temp_resi_1 + temp_resi_2));
                dis(j,i) = dis(i,j);
            end
        end
    else
        for i = 1:n
            Ridge_mat{i} = inv(C_cell{i}' * C_cell{i} + lam * eye(dmax + 1));
        end
        dis = zeros(n, n);
        for i = 1:n-1
            for j = i+1:n
                Y_1 = C_cell{i};
                Y_2 = C_cell{j};
                C_1 = Ridge_mat{j} * Y_2' * Y_1;
                C_2 = Ridge_mat{i} * Y_1' * Y_2;
                dis(i, j) = exp(-0.5 * (norm(Y_1 - Y_2 * C_1, 'fro') + norm(Y_2 - Y_1 * C_2, 'fro')));
                dis(j, i) = dis(i, j);
            end
        end
    end
    label_bag = zeros(n, length(thre_vec));
    cnt = 0;
    for thre = thre_vec
        cnt = cnt + 1;
        dis_2 = dis;
        % Thresholding the matrix
        for i = 1:n
            v = dis_2(i,:);
            [~, ind] = sort(v, 'descend');
            dis_2(i, ind(thre:end))=0;
        end
        % Construct the affinity matrix and run spectral clustering
        % algorithm
        A = dis_2 + dis_2';
        groups_sub = SpectralClustering(A, K);
        label_bag(:, cnt) = groups_sub(:);
    end
    % Calculate the similarity between each pair of labels on subset based
    % on different thresholding parameters
    score_bag = ones(length(thre_vec));
    for s_1 = 1:length(thre_vec)
        for s_2 = 1 + s_1:length(thre_vec)
            score_bag(s_1, s_2) = evalAccuracy(label_bag(:, s_1), label_bag(:, s_2));
            score_bag(s_2, s_1) = score_bag(s_1,s_2);
        end
    end
    % Pick the thresholding parameter with best score
    id_bag = [];
    for i = 1:length(thre_vec)
        ind_temp = sum(score_bag(i,:)>0.95);
        if ind_temp > 1
            id_bag = [id_bag ,i];
        end
    end
    if ~isempty(id_bag)
        id_bag = id_bag(end);
    else
        v = sum(score_bag);
        id_bag = find (v == max(v));
    end
    % Finalize the label on the subset
    groups_sub = label_bag(:,id_bag(end));
    groups_sub_vec(b_num) = evalAccuracy(N_label(ind_sub), groups_sub);
    % Classify the remaining points
    count_vec = zeros(1, K);
    for k = 1:K
       count_vec(k) = sum(groups_sub == k);
    end
    count_vec = [0, cumsum(count_vec * (dmax + 1))];
    groups_idx = zeros((dmax + 1) * n, 3);
    for k = 1:K
        ind = find(groups_sub == k);
        ind_temp1 = [];
        ind_temp2 = [];
        for j = 1:length(ind)
            ind_temp1 = [ind_temp1, ind_mat(ind(j),1:(dmax+1))];
            ind_temp2 = [ind_temp2, ind_mat(ind(j),1) * ones(1, dmax + 1)];
        end
        groups_idx(count_vec(k) + 1: count_vec(k + 1), 1) = ind_temp1;
        groups_idx(count_vec(k) + 1: count_vec(k + 1), 3) = k;
        groups_idx(count_vec(k) + 1: count_vec(k + 1), 2) = ind_temp2;
    end
    unique_id = unique(groups_idx(:, 1));
    groups = zeros(N,1);
    for i = 1:length(unique_id)
        id = unique_id(i);
        id_temp1 = find(groups_idx(:, 1) == id);
        id_temp2 = find(groups_idx(:, 2) == id);
        if ~isempty(id_temp2)
            groups(id) = groups_idx(intersect(id_temp1, id_temp2), 3);
        else
            id_temp = unique(groups_idx(id_temp1, 3));
            if length(id_temp) == 1
                groups(id) = id_temp;
            end
        end            
    end    
    % The projection matrix we use to classify the points
    prj_mat = cell(1);
    for k = 1:K
        set = find(groups == k);
        m_temp = min(length(set), m);
        sample_set = datasample(set, m_temp,...
            'Replace', false);
        X = Y_norm(:, sample_set);
        prj_mat{k} = X * ((X' * X + lam * eye(m_temp)) \ X');
    end
    % residual minimization
    remain_ind = find(groups==0);
    resi=[];
    for k =1:K
        A = prj_mat{k};
        r = num2cell((eye(D) - A) * Y_norm(:, remain_ind), 1);
        temp = cellfun(@norm, r);
        resi = [resi; temp];
    end
    resi = num2cell(resi, 1);
    [~, min_ind] = cellfun(@min, resi);
    label_test(remain_ind, b_num) = min_ind;
    label_test(groups ~= 0, b_num) = groups(groups ~= 0);
end

for m = 1:bag_num
    label_re = zeros(size(label_test));
    label_re(:, m) = label_test(:, m);
    ind_cell = cell(1);
    for k = 1:K
        ind_cell{k} = find(label_test(:,m) == k);
    end
    for i= setdiff(1:bag_num, m)
        score_mat = zeros(K, K);
        for j = 1:K
            for q = 1:K
                score_mat(q, j) = length(intersect(ind_cell{j}, find(label_test(:,i) == q)))...
                    / min(length(ind_cell{j}), length(label_test(:,i) == q));
            end
        end
        [~, ind_1] = max(score_mat);
        if length(unique(ind_1)) == length(ind_1)
            for k = 1:K
                label_re(label_test(:, i) == ind_1(k), i) = k;
                score_fit(m) = score_fit(m) + score_mat(ind_1(k), k);
            end
        else
            [~, ind_2] = max(score_mat');
            act_set = 1:K;
            for k = 1:K
                if ind_2(ind_1(k)) == k
                    label_re(label_test(:, i) == ind_1(k), i) = k;
                    act_set = setdiff(act_set,k);
                    score_fit(m) = score_fit(m) + score_mat(ind_1(k), k);
                end
            end
            if ~isempty(act_set)
                remain_ind = unique(label_test(label_re(:, i) == 0, i));
                comb = perms(1:length(act_set));
                fit_max = -inf;
                max_id = 0;
                for j = 1:factorial(length(act_set))
                    fit_temp = 0;
                    for q = 1:length(act_set)
                        fit_temp = fit_temp + score_mat(act_set(q), remain_ind(comb(j, q)));
                    end
                    if fit_temp > fit_max
                        fit_max = fit_temp;
                        max_id = j;
                    end
                end
                for j = 1:length(act_set)
                    label_re(label_test(:, i) == remain_ind(comb(max_id, j)), i) = act_set(j);
                end
                score_fit(m) = score_fit(m) + fit_max;
            end
        end
    end
    % voting
    if bag_num > 1
        label_temp = mode(label_re')';
    else
        label_temp = label_re;
    end
    label_final = [label_final; label_temp'];
end

id = find(score_fit == max(score_fit));
label_final = label_final(id(end),:);
mean_sub_accr = mean(groups_sub_vec);