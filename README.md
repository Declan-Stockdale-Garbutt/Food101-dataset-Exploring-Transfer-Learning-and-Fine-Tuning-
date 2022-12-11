# Food101 dataset Exploring Transfer Learning and Fine Tuning

## Overview

This project explores the use of three pretrained models available in the TensorFlow package for transfer learning. 

In part 1, all models are assessed no their performance on the ImageNet database to find the model with the best performance 

The models used are the Inception ResNet v2, Mobilenet v3 (small) and NASNet (mobile). All three models were initially trained on the ImageNet dataset.  
Each model will be slightly modified to remove the top layer used for prediction in the ImageNet database and substituted for a new top layer for predicting classes in the Food 101 dataset. The three models will then be trained to assess their performance on the Food 101 dataset with the most accurate performing model undergoing additional fine-tuning for further accuracy improvements. 
