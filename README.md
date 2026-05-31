# S3DL

run main.m or demo.m

The dataset can be downloaded from the link below:

https://figshare.com/articles/dataset/15-Scene_Image_Dataset/7007177?file=12855452

https://data.caltech.edu/records/mzrjq-6wc02

https://www.briancbecker.com/blog/research/pubfig83-lfw-dataset/

https://github.com/VILAlab/FERETsubset

https://cs231n.stanford.edu/tiny-imagenet-200.zip

https://zhuolinumd.github.io/projectlcksvd.html


Dataset Processing

AR: We used the 540-D features projected by a randomly normalized matrix.

CMU PIE: All images were normalized and resized to the size of 32 × 32 pixels in advance.

FERET: Each image was resized to a vector with the dimensionality of 1600.

PubFig 83 : We used the compositional features of histogram of oriented gradient (HOG), local binary pattern (LBP), and Gabor wavelets. 

Caltech 101: We used the spatial pyramid features.

15 Scene: We used an SIFT descriptor codebook with a size of 200 and a four-level spatial pyramid.

Tiny-ImageNet: The pre-trained EfficientNet-b0 network with the ImageNet dataset provided by MATLAB Deep Learning Toolbox is used to extract deep features. The features in “global_average_pooling” layer are used.

