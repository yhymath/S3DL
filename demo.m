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
N_tilde = c * reduced_num_per_class;  

alpha = 1e2;
beta  = 1e1;
lambda = 1;
eta = 1e-3;           
max_iter = 2;          
max_iter_q = 50;       
learning_rate = 0.1;    
tol = 1e-6;




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
    
    [M, N] = size(candidate_data);
    
    Q = zeros(N, N_tilde);
    for j = 1:min(N, N_tilde)
        Q(j, j) = 1;
    end
    Q = Q + 0.01 * randn(N, N_tilde);  
    for j = 1:N_tilde
        if norm(Q(:, j)) > 0
            Q(:, j) = Q(:, j) / norm(Q(:, j));
        end
    end
    
    XQ = candidate_data * Q;
    W = (D' * D + 1e-6 * eye(K)) \ (D' * XQ);
    P = (XQ * XQ' + 1e-6 * eye(M)) \ (XQ * W');
    
    %% ==========5. Optimization Process ==========  
    fprintf('Global optimization (Q: %d x %d)...\n', N, N_tilde);
    
    for iter = 1:max_iter
        fprintf('  Main iteration %d/%d\n', iter, max_iter);
        
        % Step 1: Update Q using PGD
        Q = update_global_Q(candidate_data, D, W, Htr, H_D, P, Q, ...
            alpha, beta, eta, learning_rate, max_iter_q, tol);
        
        XQ = candidate_data * Q;
        LQ = Htr * Q;
        
        % Step 2: Update W using closed-form solution
        W = update_W(XQ, D, P, LQ, H_D, alpha, beta);
        
        % Step 3: Update P
        P = update_P(P,XQ, W, alpha, lambda);
        
        % Step 4: Update D using ADMM
        D= update_D(XQ, W, D);

    end
    

    XQ_final = candidate_data * Q;
    LQ_final = Htr * Q;
    
    [~, hard_labels] = max(LQ_final, [], 1);
    
    train_data = [];
    train_label = [];
    
    for i = 1:c
        class_mask = (hard_labels == i);
        if sum(class_mask) > 0
            class_data = XQ_final(:, class_mask);
            class_scores = sum(class_data.^2, 1);
            [~, sorted_idx] = sort(class_scores, 'descend');
            n_select = min(reduced_num_per_class, sum(class_mask));
            
            if n_select > 0
                selected = find(class_mask);
                selected = selected(sorted_idx(1:n_select));
                train_data = [train_data XQ_final(:, selected)];
                train_label = [train_label i * ones(1, n_select)];
            end
        end
    end
    
    fprintf('Selected %d meta-samples\n', size(train_data,2));
    
    %% ========== 7. Testing ==========
    if size(train_data,2) > 0
        T_train = H_D * P' * train_data;
        T_test  = H_D * P' * test_data;
        T_train = T_train ./ (vecnorm(T_train) + eps);
        T_test  = T_test  ./ (vecnorm(T_test) + eps);
        
        mdl = fitcknn(T_train', train_label);
        pred = predict(mdl, T_test');
        
        accuracy1= mean(test_label(:) == pred(:));
    else
        accuracy1 = 0;
    end
    
    
    fprintf('Accuracy = %.2f%%\n', accuracy1*100);




