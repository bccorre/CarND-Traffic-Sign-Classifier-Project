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

[image1]: ./img1.png "Traffic sign example"
[image2]: ./imageAfterGrayscale.png "Image after grayscale and resizing"
[image3]: ./trainingBarPlot.png "Traffic sign distribution (training set)"
[image4]: ./validationBarPlot.png "Traffic sign distribution (validation set)"
[image5]: ./img1.png "German traffic sign from internet"
[image6]: ./img2.png "German traffic sign from internet"
[image7]: ./img3.png "German traffic sign from internet"
[image8]: ./img4.png "German traffic sign from internet"
[image9]: ./img5.png "German traffic sign from internet"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/bccorre/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 27839
* The size of the validation set is 6960
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training and test data sets.

![alt text][image3]

![alt text][image4]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

After that I applied a preprocessing function, that applies grayscale and scale features from [0,255.] to [0,1.]. Below follows an example of the resulting image, after this operation the image becames flat and instead of having 3 dimension (RGB) it has only one. This is because analyzing color images do not improve the results signigicantly.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image   							|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x32	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 				|
| Fully connected		| Input = 800. Output = 400. |
| RELU					|												|
| Fully connected		| Input = 400. Output = 120. |
| RELU					|						
| Fully connected		| Input = 120. Output = 43 |						|
| Softmax				| softmax cross entropy with logits. |



####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used AdamOptimizer, with softmax cross entropy with logits, and hyperparameters as:

Learning rate = 0.0001
Regularization factor = 0.00001

EPOCHS = 20
Batch size = 256

I used dropout in all intermediary layers with keep probality as:

Keep Probability layer 1: 0.9 (convolutional)
Keep Probability layer 2: 0.8 (convolutional)
Keep Probability layer 3: 0.5 (fully-connected)
Keep Probability layer 4: 0.5 (fully-connected)

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.996
* validation set accuracy of 0.991
* test set accuracy of 0.943

If an iterative approach was chosen:
* The first architecture tested in my approach was the LeNet architecture, found in later lessons of this course. LeNet was great but needed some tunning specially because it did not generalize well.
* I first decided to make the network deeper increasing the values of the convolutional layers, and started to see some improvement, but the generalization issue still remained.
* After this generalization problem, I implemented regularization and, after some tests, I found the value of 0.00001. But it still needed some extra technique to improve generalization and increase validation accuracy.
* Then, I decided to include dropout in every hidden layer of my architecture, choosing keeping probabilities of 0.9 and 0.8 for the convolutional layers, and 0.5 for the fully connected layers. This dropout approach really improved validation accuracy and makes the networks more robust.
* Right after I added an early stop to the training loop, since I was satisfied with validation accuracy of 0.99 and break the learning loop.

If a well known architecture was chosen:
* I Choose LeNet architeure with dropout layers and regulariztion
* It was very good at predicting hand written numbers ans could be easily adapted to traffic sign images or any other type of image recognition.
* The model generalizes well because the training and validation accuracies are sufficiently close and the test accuracy is above 0.93, that was the target of the project for the validation examples.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7]
![alt text][image8] ![alt text][image9]

The first image might be difficult to classify because it can be easily misinterpreted as a 60km/h sign instead of a 80 km/h. The second, third and fourth are very easy and it is expected that the model performs well on those. But the last is a little more difficult because of its lower resolution, that makes the symbol in the middle a little difficult to identify.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop Sign      		| Stop sign   									|
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.

For the first image, the model was not sure that this is a speed limit sign (60km/h) (probability of 0.0279), and the image does contain a speed limit 80km/h sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .0279         			| Speed Limit (60 km/h)   									|
| .0245     				| Go straight or right 										|
| .0119			| Speed limit (30km/h)											|
| .0115	      			| Speed limit (20km/h)					 				|
| .0101				    | Ahead only      							|


For the second image, the model was not sure that this is a stop sign (probability of 0.08), and the image does contain a speed a stop sign . The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .0819         			| Stop  									|
| .0296     				| Speed limit (30km/h)	 										|
| .0192			| Speed limit (50km/h)											|
| .0126	      			| Turn right ahead					 				|
| .0041				    | Speed limit (70km/h)	      							|

For the third image, the model was not sure that this is a speed limit 30 km/h (probability of 0.07), and the image does contain a speed limit 30 km/h sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .0773         			| speed limit (30 km/h)  									|
| .0539     				| Speed limit (20km/h)	 										|
| .0229			| Wild animals crossing										|
| .0189	      			| Stop					 				|
| .0116				    | Roundabout mandatory	      							|

For the fourth image, the model was not sure that this is a Right-of-way at the next intersection (probability of 0.07), and the image does contain a Roundabout mandatory sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .0552         			| Right-of-way at the next intersection 									|
| .0509     				| Priority road				|
| .0429			| Roundabout mandatory										|
| .0319	      			| Beware of ice/snow					 				|
| .0280				    | End of no passing by vehicles over 3.5 metric tons				|

And finally, for the fifth image, the model was not sure that this is a Children crossing sign (probability of 0.12), and the image does contain a Beware of ice/snow sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
28 23 30 20 29
| .121         			| Children crossing 									|
| .0761     				| Slippery road				|
| .0687			| Beware of ice/snow										|
| .0471	      			| General caution					 				|
| .0353				    | Bicycles crossing				|
