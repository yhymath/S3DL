clear all; clc; close all;

%% ==========load data==========
load scene15.mat;
DATA = featureMat;
[~, Label] = max(labelMat, [], 1);
DATA = DATA ./ repmat(sqrt(sum(DATA.^2)), [size(DATA,1),1]);

rng(1);
c = length(unique(Label));

pre_select_num = 40;
reduced_num_per_class = 39;
ki = 35;

K = c * ki;
alpha = 1e2;
beta  = 1e1;
lambda = 1;
eta = 1e3;
Max_Outer=1;
max_iter_q = 150;
learning_rate = 0.001;
tol = 1e-6;

accuracy1 = zeros(10,1);

for ii = 1:10
    fprintf('\n=== Run %d  ===\n', ii);
    tic_total = tic;

    %% ========== 1. Random Select Training Data ==========
    fprintf('split data...');
    candidate_data = [];
    candidate_labels = [];
    candidate_indices = [];

    for i = 1:c
        idx = find(Label == i);
        idx = idx(randperm(length(idx), pre_select_num));

        candidate_data  = [candidate_data DATA(:, idx)];
        candidate_labels = [candidate_labels Label(idx)];
        candidate_indices = [candidate_indices idx];
    end
  
    %% ========== 2. Testing data ==========
    test_indices = setdiff(1:size(DATA,2), candidate_indices);
    test_data  = DATA(:, test_indices);
    test_label = Label(test_indices);

    Htt = zeros(c, size(test_data,2));
    for i = 1:size(test_data,2)
        Htt(test_label(i), i) = 1;
    end

  fprintf('complete!\n');

    %% ========== 3. H Matrix ==========
    fprintf('H Matrix and Dictionary...\n');
    Htr = zeros(c, size(candidate_data,2));
    for i = 1:size(candidate_data,2)
        Htr(candidate_labels(i), i) = 1;
    end

    H_D = [];
    for i = 1:c
        H_D = blkdiag(H_D, ones(1, K/c));
    end

    %% ========== 4. Initialization ==========
    D = initializationDictionary(candidate_data, Htr, K, 10, 30);
    %% ==========5. Optimization==========  
  
    fprintf('learning Q matrix...\n');
    tic_q = tic;

    train_data = [];
    train_label = [];
    train_idx = [];
    for i=1:Max_Outer
    for class_id = 1:c
        mask = (candidate_labels == class_id);
        Xc = candidate_data(:, mask);
        yc = candidate_labels(mask);
        ic = candidate_indices(mask);

        if size(Xc,2) == 0
            continue;
        end

        Hc = labels2onehot(yc, c);
        actual_atoms = min(ki, reduced_num_per_class);

        [d, n] = size(Xc);
        Q = 0.1 * randn(n, actual_atoms);
        Dc = D(:, (class_id-1)*ki+1 : class_id*ki);
        Wc = 0.1 * randn(actual_atoms, actual_atoms);
        Pc = 0.1 * randn(d, actual_atoms);
        Cc = Hc(:, 1:actual_atoms);

        [Qopt, ~] = learnQMatrixPerClass( ...
            Xc, Hc, Q, Dc, Wc, Pc, Cc, ...
            alpha, eta, beta, max_iter_q, learning_rate, tol);

        score = sum(abs(Qopt), 2);
        [~, ord] = sort(score, 'descend');
        sel_local  = ord(1:min(reduced_num_per_class, length(ord)));
        sel_global = ic(sel_local);

        train_data  = [train_data Xc(:, sel_local)];
        train_label = [train_label yc(sel_local)];
        train_idx   = [train_idx sel_global];
    end

    qp_time(ii) = toc(tic_q);


   
    Htr = zeros(c, size(train_data,2));
    for i = 1:size(train_data,2)
        Htr(train_label(i), i) = 1;
    end

    H_D = [];
    for i = 1:c
        H_D = blkdiag(H_D, ones(1, K/c));
    end

    D = initializationDictionary(train_data, Htr, K, 10, 30);

    [D, P, W, ~] = Update_Other_Variable(train_data, D, Htr, H_D, K, c, alpha, beta, lambda);
 
    end
    %% ========== 6. Testing Process ==========
    T_train = H_D * P' * train_data;
    T_test  = H_D * P' * test_data;
    T_train = T_train ./ vecnorm(T_train);
    T_test  = T_test  ./ vecnorm(T_test);

    mdl = fitcknn(T_train', train_label);
    pred = predict(mdl, T_test');

    accuracy1(ii) = mean(test_label(:) == pred(:));
    fprintf('Accuracy = %.2f\n', accuracy1(ii)*100);
end

%% ========== Accuracy on Testing Data==========
fprintf('\n====== 10 run ======\n');
fprintf('Accuracy: %.2f ± %.2f (%%) \n', ...
mean(accuracy1)*100, std(accuracy1)*100);

