#!/bin/bash

# Train damper
python model/train_damper.py /scratch-nvme/ml-datasets/imagenet/torchvision_ImageFolder --epochs 1

# Train models
python training/train_imagenet.py path/to/dataset --name 'imagenet_noisedropdamper' --damper --damper_type noise_drop
python training/train_imagenet.py path/to/dataset --name 'imagenet_aprsp' --augmentation 'aprsp'
python training/train_imagenet.py path/to/dataset --name 'imagenet_aprp' --augmentation 'aprp'
python training/train_imagenet.py path/to/dataset --name 'imagenet_augmix' --augmentation 'augmix'

# Evaluate models
python training/eval_imagenet.py path/to/dataset path/to/dataset-c --name 'best_imagenet_noisedropdamper'
python training/eval_imagenet.py path/to/dataset path/to/dataset-c --name 'best_imagenet_aprsp'
python training/eval_imagenet.py path/to/dataset path/to/dataset-c --name 'best_imagenet_aprp'
python training/eval_imagenet.py path/to/dataset path/to/dataset-c --name 'best_imagenet_augmix'