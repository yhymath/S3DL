function [Q_opt, losses] = learnQMatrixPerClass(X, H, Q_init, D_init, A_init, P_init, C_init, alpha, eta, beta, max_iter, lr, tol)


    
    Q = Q_init;
    D = D_init;
    A = A_init;
    P = P_init;
    C = C_init;
    
    losses = zeros(max_iter, 1);
    
    for iter = 1:max_iter
        % gradient
        term1 = X' * X * Q;
        term2 = alpha * X' * P * P' * X * Q;
        term3 = beta * H' * H * Q;
        term4 = X' * D * A;
        term5 = alpha * X' * P * A;
        term6 = beta * H' * C * A;
        
        grad_Q = 2 * (term1 + term2 + term3) - 2 * (term4 + term5 + term6);
        
        % gradient descent
        Q = Q - lr * grad_Q;
        
        % soft-thresholding
        Q = sign(Q) .* max(abs(Q) - lr * eta, 0);
        
        
        if mod(iter, 5) == 0
            D = (X * Q) * pinv(A);
            P = (X * Q) * pinv(A);
            C = (H * Q) * pinv(A);
        end
        
        % loss
        reconstruction_loss = norm(X * Q - D * A, 'fro')^2;
        projection_loss = alpha * norm(A - P' * X * Q, 'fro')^2;
        label_loss = beta * norm(H * Q - C * A, 'fro')^2;
        sparsity_loss = eta * sum(abs(Q(:)));
        
        total_loss = reconstruction_loss + projection_loss + label_loss + sparsity_loss;
        losses(iter) = total_loss;
        
        % check convergence
        if iter > 1 && abs(losses(iter) - losses(iter-1)) < tol
            break;
        end
  
        if iter > 1 && losses(iter) > losses(iter-1)
            lr = lr * 0.9;
        end
    end
    
    Q_opt = Q;
    losses = losses(1:iter);
end