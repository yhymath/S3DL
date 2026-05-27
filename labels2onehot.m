function onehot_labels = labels2onehot(labels, num_classes)

    num_samples = length(labels);
    
  
    if nargin < 2
        num_classes = max(labels);
    end

    if min(labels) < 1
        error('Label values must start from 1. If labels start from 0, add 1 first.');
    end
    if max(labels) > num_classes
        error('The label value exceeds the specified number of classes.');
    end
    
 
    onehot_labels = zeros(num_classes, num_samples);
    
    indices = sub2ind([num_classes, num_samples], labels, 1:num_samples);
    onehot_labels(indices) = 1;
end

