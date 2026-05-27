function X_thresh = soft_threshold(X, threshold)
X_thresh = sign(X) .* max(abs(X) - threshold, 0);
end
