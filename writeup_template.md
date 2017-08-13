#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./sign_count.png "Visualization"
[image2]: ./table1.png
[image3]: ./table2.png
[image4]: ./new_images/image1.jpg "Traffic Sign 1"
[image5]: ./new_images/image2.jpg "Traffic Sign 2"
[image6]: ./new_images/image3.JPG "Traffic Sign 3"
[image7]: ./new_images/image4.jpg "Traffic Sign 4"
[image8]: ./new_images/image5.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! 

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The figure shows the count of each sign for training set, validation set and test set, respectively. We can find that different signs are not uniformly distributed. 


![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. 

All the data is normalized to zero mean and equal variance to be ready for deep learning.
X_train = X_train / 255.0-0.5
X_valid = X_valid / 255.0-0.5
X_test = X_test / 255.0-0.5


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

![alt text][image2]
 

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The AdamOptimizer is applied in the model. The parameters are as follows: epochs = 10, batch_size = 128, learning rate = 0.001.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. 

After the data is normalized and with the current parameter setting, the training accuracy is pretty high close to 1, but the validation accuracy is relatively low, which means the model is over-fitting. Thus I add to dropout unit after two fully connected layers with keep_prob = 0.5, but then both training accuracy and validation accuracy become low, which means the model is under-fitting. Thus I increase keep_prob to 0.7, then finally then validation accuracy is over 0.93.

The final model results:
	•	training set accuracy of 0.991
	•	validation set accuracy of 0.933
	•	test set accuracy of 0.927
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] 
![alt text][image5] 
![alt text][image6] 
![alt text][image7] 
![alt text][image8]

The second, fourth and fifth images might be difficult to classify because of the watermark.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop      		| Stop    									| 
| Road work     			| Road work 										|
| Speed limit(60km/h)					| Speed limit(60km/h)											|
| Pedestrians	      		| Pedestrians					 				|
| Turn right ahead			| Turn right ahead      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

![alt text][image3]



