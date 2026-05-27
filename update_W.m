function W= update_W(XQ, D, P, LQ, H_D, alpha, beta)
K = size(D, 2);
reg = 1e-4;

A = D' * D + alpha * eye(K) + beta * (H_D' * H_D) + reg * eye(K);
B = D' * XQ + alpha * P' * XQ + beta * H_D' * LQ;

W = A \ B;

W = max(W, 0);
end
