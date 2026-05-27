clear all; clc; close all;

%% ========== Global parameters ==========
reduced_num_per_class = 9;
ki = 5;

alpha = 1e2;
beta  = 1e1;
lambda = 1;
eta = 1e-3;
max_iter = 8;
max_iter_q = 30;
learning_rate = 0.01;
tol = 1e-6;

accuracy = zeros(10,1);

for ii = 1:10
    fprintf('\n========== Split %d / 10 ==========\n', ii);

    %% ========== Load pre-split data ==========
    fprintf('Loading data...\n');
    load(sprintf('ar_%d.mat', ii));

    c = length(unique(train_label));

    K = c * ki;
    N_tilde = c * reduced_num_per_class;

    %% ========== H matrix ==========
    fprintf('Building H matrix...\n');
    Htr = zeros(c, size(train_data,2));
    for i = 1:size(train_data,2)
        Htr(train_label(i), i) = 1;
    end

    H_D = [];
    for i = 1:c
        H_D = blkdiag(H_D, ones(1, K/c));
    end

    %% ========== Dictionary initialization ==========
    D = initializationDictionary(train_data, Htr, K, 10, 30);

    [M, N] = size(train_data);

    %% ========== Q initialization ==========
    Q = zeros(N, N_tilde);
    for j = 1:min(N, N_tilde)
        Q(j, j) = 1;
    end
    Q = Q + 0.01 * randn(N, N_tilde);
    for j = 1:N_tilde
        Q(:, j) = Q(:, j) / (norm(Q(:, j)) + eps);
    end

    XQ = train_data * Q;
    W = (D' * D + 1e-6 * eye(K)) \ (D' * XQ);
    P = (XQ * XQ' + 1e-6 * eye(M)) \ (XQ * W');

    %% ========== Global optimization ==========
    fprintf('Global optimization (Q: %d x %d)...\n', N, N_tilde);
    for iter = 1:max_iter
        fprintf('  Iteration %d/%d\n', iter, max_iter);

        Q = update_global_Q(train_data, D, W, Htr, H_D, P, Q, ...
            alpha, beta, eta, learning_rate, max_iter_q, tol);

        XQ = train_data * Q;
        LQ = Htr * Q;

        W = update_W(XQ, D, P, LQ, H_D, alpha, beta);

        P = update_P(P, XQ, W, alpha, lambda);

        D = update_D(XQ, W, D);
    end

 
    XQ_final = train_data * Q;
    LQ_final = Htr * Q;
    [~, hard_labels] = max(LQ_final, [], 1);

    train_data_sel = [];
    train_label_sel = [];

    for i = 1:c
        class_mask = (hard_labels == i);
        if sum(class_mask) > 0
            scores = sum(XQ_final(:, class_mask).^2, 1);
            [~, idx] = sort(scores, 'descend');
            n_sel = min(reduced_num_per_class, sum(class_mask));
            sel = find(class_mask);
            sel = sel(idx(1:n_sel));

            train_data_sel = [train_data_sel XQ_final(:, sel)];
            train_label_sel = [train_label_sel i * ones(1, n_sel)];
        end
    end


    %% ========== Testing ==========
    fprintf('Testing...\n');

    T_train = H_D * P' * train_data_sel;
    T_test  = H_D * P' * test_data;
    T_train = T_train ./ (vecnorm(T_train) + eps);
    T_test  = T_test  ./ (vecnorm(T_test) + eps);

    mdl = fitcknn(T_train', train_label_sel);
    pred = predict(mdl, T_test');

    accuracy(ii) = mean(test_label(:) == pred(:));
    fprintf('Accuracy = %.2f%%\n', accuracy(ii)*100);
end

%% ========== Summary ==========
fprintf('\n====== 10 splits ======\n');
fprintf('Accuracy: %.2f ± %.2f (%%) \n', mean(accuracy)*100, std(accuracy)*100);
fprintf('Best: %.2f%%, Worst: %.2f%%\n', max(accuracy)*100, min(accuracy)*100);


