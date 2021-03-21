#!/bin/bash -ex

python experiments/2-common/4a-embed-patches.py           with architecture=EfficientNetB7 image_shape=[300,300,3] data_dir=/mnt/files/datasets/vgdb_2016/patches/random/ output_dir=/mnt/files/datasets/vgdb_2016/embed/patches/random_efficientnetb7/ batch_size=256
python experiments/2-common/4a-embed-patches.py           with architecture=InceptionV3    image_shape=[299,299,3] data_dir=/mnt/files/datasets/vgdb_2016/patches/random/ output_dir=/mnt/files/datasets/vgdb_2016/embed/patches/random_inceptionv3/    batch_size=256
python experiments/2-common/5-train-top-classifier.py     with layer="avg_pool" max_patches=50 data_dir=/mnt/files/datasets/vgdb_2016/embed/patches/random_efficientnetb7/ -F logs/vg/efficientnetb7-svc/
python experiments/2-common/6-generate-svm-predictions.py with layer="avg_pool" max_patches=50 data_dir=/mnt/files/datasets/vgdb_2016/embed/patches/random_efficientnetb7/ ckpt=logs/vg/efficientnetb7-svc/1/model.pkl -F logs/vg/efficientnetb7-svc/predictions/

python experiments/2-common/5-train-top-classifier.py     with layer="global_average_pooling2d" max_patches=20 data_dir=/mnt/files/datasets/vgdb_2016/embed/patches/min_inceptionv3/ -F logs/vg/min_inceptionv3/
python experiments/2-common/5-train-top-classifier.py     with layer="global_average_pooling2d" max_patches=10 data_dir=/mnt/files/datasets/vgdb_2016/embed/patches/min_inceptionv3/ -F logs/vg/min_inceptionv3/
python experiments/2-common/5-train-top-classifier.py     with layer="global_average_pooling2d" max_patches=5  data_dir=/mnt/files/datasets/vgdb_2016/embed/patches/min_inceptionv3/ -F logs/vg/min_inceptionv3/
python experiments/2-common/5-train-top-classifier.py     with layer="global_average_pooling2d" max_patches=2  data_dir=/mnt/files/datasets/vgdb_2016/embed/patches/min_inceptionv3/ -F logs/vg/min_inceptionv3/

python experiments/2-common/6-generate-svm-predictions.py with layer="global_average_pooling2d" max_patches=20 data_dir=/mnt/files/datasets/vgdb_2016/embed/patches/min_inceptionv3/ ckpt=logs/vg/min_inceptionv3/1/model.pkl -F logs/vg/min_inceptionv3/predictions/
python experiments/2-common/6-generate-svm-predictions.py with layer="global_average_pooling2d" max_patches=10 data_dir=/mnt/files/datasets/vgdb_2016/embed/patches/min_inceptionv3/ ckpt=logs/vg/min_inceptionv3/2/model.pkl -F logs/vg/min_inceptionv3/predictions/
python experiments/2-common/6-generate-svm-predictions.py with layer="global_average_pooling2d" max_patches=5  data_dir=/mnt/files/datasets/vgdb_2016/embed/patches/min_inceptionv3/ ckpt=logs/vg/min_inceptionv3/3/model.pkl -F logs/vg/min_inceptionv3/predictions/
python experiments/2-common/6-generate-svm-predictions.py with layer="global_average_pooling2d" max_patches=2  data_dir=/mnt/files/datasets/vgdb_2016/embed/patches/min_inceptionv3/ ckpt=logs/vg/min_inceptionv3/4/model.pkl -F logs/vg/min_inceptionv3/predictions/

python experiments/2-common/5-train-top-classifier.py     with layer="global_average_pooling2d" max_patches=20 data_dir=/mnt/files/datasets/vgdb_2016/embed/patches/max_inceptionv3/ -F logs/vg/max_inceptionv3/
python experiments/2-common/5-train-top-classifier.py     with layer="global_average_pooling2d" max_patches=10 data_dir=/mnt/files/datasets/vgdb_2016/embed/patches/max_inceptionv3/ -F logs/vg/max_inceptionv3/
python experiments/2-common/5-train-top-classifier.py     with layer="global_average_pooling2d" max_patches=5  data_dir=/mnt/files/datasets/vgdb_2016/embed/patches/max_inceptionv3/ -F logs/vg/max_inceptionv3/
python experiments/2-common/5-train-top-classifier.py     with layer="global_average_pooling2d" max_patches=2  data_dir=/mnt/files/datasets/vgdb_2016/embed/patches/max_inceptionv3/ -F logs/vg/max_inceptionv3/

python experiments/2-common/6-generate-svm-predictions.py with layer="global_average_pooling2d" max_patches=20 data_dir=/mnt/files/datasets/vgdb_2016/embed/patches/max_inceptionv3/ ckpt=logs/vg/max_inceptionv3/1/model.pkl -F logs/vg/max_inceptionv3/predictions/
python experiments/2-common/6-generate-svm-predictions.py with layer="global_average_pooling2d" max_patches=10 data_dir=/mnt/files/datasets/vgdb_2016/embed/patches/max_inceptionv3/ ckpt=logs/vg/max_inceptionv3/2/model.pkl -F logs/vg/max_inceptionv3/predictions/
python experiments/2-common/6-generate-svm-predictions.py with layer="global_average_pooling2d" max_patches=5  data_dir=/mnt/files/datasets/vgdb_2016/embed/patches/max_inceptionv3/ ckpt=logs/vg/max_inceptionv3/3/model.pkl -F logs/vg/max_inceptionv3/predictions/
python experiments/2-common/6-generate-svm-predictions.py with layer="global_average_pooling2d" max_patches=2  data_dir=/mnt/files/datasets/vgdb_2016/embed/patches/max_inceptionv3/ ckpt=logs/vg/max_inceptionv3/4/model.pkl -F logs/vg/max_inceptionv3/predictions/

python experiments/2-common/5-train-top-classifier.py     with layer="global_average_pooling2d" max_patches=20 data_dir=/mnt/files/datasets/vgdb_2016/embed/patches/random_inceptionv3/ -F logs/vg/random_inceptionv3/
python experiments/2-common/5-train-top-classifier.py     with layer="global_average_pooling2d" max_patches=10 data_dir=/mnt/files/datasets/vgdb_2016/embed/patches/random_inceptionv3/ -F logs/vg/random_inceptionv3/
python experiments/2-common/5-train-top-classifier.py     with layer="global_average_pooling2d" max_patches=5  data_dir=/mnt/files/datasets/vgdb_2016/embed/patches/random_inceptionv3/ -F logs/vg/random_inceptionv3/
python experiments/2-common/5-train-top-classifier.py     with layer="global_average_pooling2d" max_patches=2  data_dir=/mnt/files/datasets/vgdb_2016/embed/patches/random_inceptionv3/ -F logs/vg/random_inceptionv3/

python experiments/2-common/6-generate-svm-predictions.py with layer="global_average_pooling2d" max_patches=20 data_dir=/mnt/files/datasets/vgdb_2016/embed/patches/random_inceptionv3/ ckpt=logs/vg/random_inceptionv3/1/model.pkl -F logs/vg/random_inceptionv3/predictions/
python experiments/2-common/6-generate-svm-predictions.py with layer="global_average_pooling2d" max_patches=10 data_dir=/mnt/files/datasets/vgdb_2016/embed/patches/random_inceptionv3/ ckpt=logs/vg/random_inceptionv3/2/model.pkl -F logs/vg/random_inceptionv3/predictions/
python experiments/2-common/6-generate-svm-predictions.py with layer="global_average_pooling2d" max_patches=5  data_dir=/mnt/files/datasets/vgdb_2016/embed/patches/random_inceptionv3/ ckpt=logs/vg/random_inceptionv3/3/model.pkl -F logs/vg/random_inceptionv3/predictions/
python experiments/2-common/6-generate-svm-predictions.py with layer="global_average_pooling2d" max_patches=2  data_dir=/mnt/files/datasets/vgdb_2016/embed/patches/random_inceptionv3/ ckpt=logs/vg/random_inceptionv3/4/model.pkl -F logs/vg/random_inceptionv3/predictions/
