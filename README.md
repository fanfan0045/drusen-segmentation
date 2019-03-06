# drusen-segmentation
This is the implementation code for paper submitted to Bioinformatics: "Unsupervised deep feature extraction via adaptive collaborative learning for drusen segmentation of fundus images".

# required environment
C++, matlab 

# Datasets
STARE dataset available: http://cecas.clemson.edu/~ahoover/stare/

DRIVE dataset available: https://www.isi.uu.nl/Research/Databases/DRIVE/

# Code
1. To train deep_net and SVM classifier based on training data:
    main_train.m (drusen_segmentation_demo.m; training.m; net_struc,m; updating_net.m)
2. To test the model based on test data:
    main_test.m(test_patches.m; test_evaluation.m )
