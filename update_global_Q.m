function Q_new = update_global_Q(X, D, W, H, H_D, P, Q, alpha, beta, eta, lr, max_iter, tol)
% Update Q using Proximal Gradient Descent (PGD)
[~, N] = size(X);
[~, N_tilde] = size(Q);
K = size(D, 2);

Q_new = Q;

% Precompute constant matrices
XtX = X' * X;
XtD = X' * D;
XtP = X' * P;
HtH = H' * H;
HtHD = H' * H_D;
PtX = P' * X;

const_term1 = XtD * W;
const_term2 = XtP * W;
const_term3 = HtHD * W;

for iter = 1:max_iter
    Q_old = Q_new;

    % Compute gradient of smooth part g(Q) 
    grad1 = 2 * (XtX * Q_old - const_term1);
    grad2 = 2 * alpha * (XtP * (PtX * Q_old) - const_term2);
    grad3 = 2 * beta * (HtH * Q_old - const_term3);

    grad = grad1 + grad2 + grad3;

    % Gradient clipping for stability
    grad_norm = norm(grad, 'fro');
    if grad_norm > 10
        grad = grad * (10 / grad_norm);
    end

    % Gradient descent step
    Q_temp = Q_old - lr * grad;

    % Proximal step for L1 regularization
    Q_new = soft_threshold(Q_temp, lr * eta);

    % Column normalization
    for j = 1:N_tilde
        col_norm = norm(Q_new(:, j));
        if col_norm > 1e-6
            Q_new(:, j) = Q_new(:, j) / col_norm;
        end
    end

    % Check convergence
    diff = norm(Q_new - Q_old, 'fro') / (norm(Q_old, 'fro') + eps);
    if diff < tol
        break;
    end
end
end
