function P = update_P(P,XQ, W, alpha, lambda)
v  = sqrt(sum(P .* P, 2) + eps);
V  = diag(1 ./ (v));
P =  inv(XQ * XQ' + alpha \ lambda * V) * XQ* W';
end
