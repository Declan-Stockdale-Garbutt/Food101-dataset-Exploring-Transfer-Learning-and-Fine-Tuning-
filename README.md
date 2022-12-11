# Food101 dataset Exploring Transfer Learning and Fine Tuning

The full report is available below. It also contains a discussion on the history and future of Deep Learning

https://github.com/Declan-Stockdale/Food101-dataset-Exploring-Transfer-Learning-and-Fine-Tuning-/blob/master/Assignment2_Report.docx


## Overview

This project explores the use of three pretrained models available in the TensorFlow package for transfer learning. 

The models used are the Inception ResNet v2, Mobilenet v3 (small) and NASNet (mobile). All three models were initially trained on the ImageNet dataset.  
Each model will be slightly modified to remove the top layer used for prediction in the ImageNet database and substituted for a new top layer for predicting classes in the Food 101 dataset. The three models will then be trained to assess their performance on the Food 101 dataset with the most accurate performing model undergoing additional fine-tuning for further accuracy improvements in part 2. 

## Part 1 - Transfer Learning from ImageNet to Food101 dataset

### How does transfer Learning work? 
The reason transfer learning works well out of the box is that the early layers in the model have already been trained to detect features such as edges, curves and shapes. They may also be able to determine relative distances between features which will be useful in the Food 101 dataset which would be useful for differentiating between a slice of pizza and a slice of cake which both have similar sector shapes but differ vastly in height. 
The models were trained on the ImageNet database and through the training process, were able to identify features that were later fed into the predictive layers for object classification. For our purposes, we are going to keep the earlier feature detection layers and substitute the predictive layers with untrained densely connected layers. We will then input our images into the model and retrain the model with the new predictive layer to predict our new image classes from the new dataset.  


Examples of foods in Food101 dataset

![image](https://user-images.githubusercontent.com/53500810/206884900-51bff7a9-afa2-47ad-b4be-1feddf80abb8.png)

From figure 1, it’s quite clear the images are noisy and are not framed particularly well in some cases. This is useful as it represents potential real-world images where the items may not be centered, or the lighting conditions are not optimal. The presentation of the food item can also vary widely within the same class either due to cooking process, plating or various additional sides such as in the various examples of steak in figure 2. 
![image](https://user-images.githubusercontent.com/53500810/206884910-3d8a0b5e-9a53-40a6-8601-f56df205668d.png)


### Train Validation Test split 
The dataset is structured in a way that each class has its own folder consisting of 100 images. The train and test images are designated in separate text files one for training and another for testing. 
The file path and image class labels were put into a pandas dataframe.  The training data was shuffled to avoid the model seeing a large number of the same class in one go. Lastly a validation set was split off from the training set. The final image count is 60600, 15149, 25249 for train, validation and test sets. 

### Model setups 
All various models were downloaded using the TensorFlow inbuilt application. When importing the models, the “include_top” value was set to False which removes the last predictive layer of the model. This was substituted with 2 x (256 densely connected layers both with ReLu activation function) and a final 101 densely connected layer using the SoftMax function to predict one of the 101 classes in the Food 101 dataset. 
The inception ResNet v3 model input images were set to (299,299,3) whereas the MobileNet and NASNet were set to (224,224,3). These values are defined in the TensorFlow documentation [43-45]. The colour channel was left as ‘rgb’ as its likely that information is encoded in the colour e.g., the model would need colour to differentiate between chocolate cake and tiramisu and possibly soups. 
The batch size was set to 32 and the number of epochs was set at 20. The optimizer used was Adam using the default value of 0.001. As the images went through a generator they were in vector format. The most relevant metric for the multiclass classification problem was categorical cross entropy and chosen metric was accuracy. 
Early Stopping was the only call back used as this initial testing stage was to compare how the models performed more or less out of the box. The validation loss was monitored, and the early stopping patience was set to 5. No additional hyperparameter tuning was performed.  

### Overall results 
Table 3 shows the accuracy, F1 score, recall and precision for each model. The metrics were calculated using the macro value as there is no class imbalance within the datasets. The highest values are in bold and its clear that the MobileNet v3 model outperformed the other two models. From figure 21, if we had set the epochs to a lower value, we may have ended up with NASNet as the best model however with extended epochs, MobileNet v3 was able to continue learning. 

![image](https://user-images.githubusercontent.com/53500810/206884961-d8579da0-a7f6-4153-bfaf-e66e91fa26fd.png)


![image](https://user-images.githubusercontent.com/53500810/206884966-8b0d5b20-670c-465e-83fa-b93eb94ceea6.png)


### Examples of similar looking images for various classes 
Figure 22 shows that there are some categories that appear identical. These are not limited to clam chowder and lobster bisque, pulled pork sandwiches and hamburgers and spaghetti bolognaise and spaghetti carbonara. These would be difficult for a human to determine. 

![image](https://user-images.githubusercontent.com/53500810/206884976-b6c52532-babd-4660-b00f-1f6552c48030.png)

### MobileNet v3 Fine tuning on Food 101 dataset 
The same process was used to generate the train/validation/test splits as described earlier. The MobileNet v3 was again implemented with the include_top set to False and the previously mentioned top layers of 2 x (256 densely connected layers both with ReLu activation function) and a final 101 densely connected layer using the SoftMax function for final prediction. As were fine tuning, the Adam learning rate was decreased to 0.0001. 
To fine tune a model, layers are unfrozen from the base model to allow for additional learning to occur. 
In this report the number of layers that are unfrozen are recorded in table  
 
The total number of parameters was 2,902,973 for all models where 2554968 came from the MobileNet v3 model and the remaining 348,005 come from the added top layers used for fine tuning. 
 
The layers are unfrozen starting from the last layer of the base model working backwards e.g. for the  model with an unfrozen layers value of 5 in table 4, only the last 5 layers are trainable not including the  additional prediction layers that were added for predictions. The number of trainable parameters for     each model are shown below.  

It can be seen that more unfrozen layers results in more trainable  parameters which would increase training time significantly. Additional callbacks were implemented. Early stopping was again applied but set to a patience of 10 with restore_best_weights set to true. Reduce_lr with a patience of 5, decrease   factor of 0.2 and min_lr of 0.00000001. Finally model checkpoint was also implemented incase the training failed at any point. 
 
![image](https://user-images.githubusercontent.com/53500810/206885014-aa87a972-94db-4eed-bfe2-49287a52b055.png)

![image](https://user-images.githubusercontent.com/53500810/206885030-55bc6068-d51f-43ab-af9d-6d9f855716c8.png)

### Conclusion 
Transfer learning using the Inception ResNet v2, MobileNet v3 and NASNet were applied to the Food 101 dataset. The top layer of the models was removed and replaced with three additional layers, 2x256 densely connected layers with ReLu followed by a third 101 densely connected layer with SoftMax for predictions. MobileNet v3 had the best accuracy of 48.6% compared to 45.3% for NASNet and 31% for Inception ResNet v2. 
The MobileNet v2 model was modified so that layers nearest the output could be trainable in steps of 
10. The model was the last 20 layers set to trainable had the highest accuracy of 60.83%, an improvement of over 12% compared to the out of the box implementation used in the initial stage. 



