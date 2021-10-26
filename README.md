# IMDB-Wiki Age & Gender Prediction
In this notebook[imdb_wiki_gpu.ipynb], we shall build an advanced convolutional neural network with 2 output layers to predict age & gender for each image. Due to space constraints, I only used images from IMDB provided here: [https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/](url) [460723 images, 7GB].

# Dataset
Images are of varying dimension with 3 channels[RGB]: \
![alt text](https://github.com/kwquan/IMDB-wiki/blob/main/imdb_sample.jpg)

# Data Pre-processing
Before proceeding to modeling part, we have to first clean up our data & put it into their respective folders. \
We can do this by first downloading the data from the above link, then running imdb_wiki_preparation.py. \
Process is as follows: \
![alt text](https://github.com/kwquan/IMDB-wiki/blob/main/process.png)

My code is mostly based on this link: [https://medium.com/free-code-camp/how-to-build-an-age-and-gender-multi-task-predictor-with-deep-learning-in-tensorflow-20c28a1bd447](url)
with some minor changes added. Also, I added explanations for each part in the comments.

# Network Architecture
![alt text](https://github.com/kwquan/IMDB-Wiki/blob/main/imdb_wiki_nn.png)

The whole model consists of 2 branches: \
[AGE]
1) Convolution, 16 filters with size (3x3) [Relu activation with Batch Normalization] 
2) Max Pool with size (2x2) 
3) Convolution, 32 filters with size (3x3) [Relu activation with Batch Normalization] 
4) Max Pool with size (2x2) 
5) Convolution, 64 filters with size (3x3) [Relu activation with Batch Normalization] 
6) Max Pool with size (2x2) 
7) Convolution, 128 filters with size (3x3) [Relu activation with Batch Normalization] 
8) Convolution, 128 filters with size (3x3) [Relu activation with Batch Normalization] 
9) Max Pool with size (2x2) 
10) Dense(units = 32) [Relu activation] 
11) Dense(units = 1) [Relu activation] 

[GENDER]
1) Convolution, 16 filters with size (3x3) [Relu activation with Batch Normalization] 
2) Max Pool with size (2x2) 
3) Dropout(0.1)
4) Convolution, 32 filters with size (3x3) [Relu activation with Batch Normalization] 
5) Max Pool with size (2x2) 
6) Dropout(0.1)
7) Convolution, 64 filters with size (3x3) [Relu activation with Batch Normalization] 
8) Max Pool with size (2x2) 
9) Dropout(0.1)
7) Dense(units = 32) [Relu activation] 
8) Dense(units = 2) [Softmax activation]

# Model Performance
Model is able to achieve an decent mae of < 9 for age & < 0.6 crossentropy for gender on the validation set with 2 epochs. \
If we examine some of the images closely, we will realize that some of them have incorrect labels[especially for age]. \
This is something we cannot rectify[since we can't possibly go through every single image & check their age labels]. \
However, we can change our model hyper-params to give better results so feel free to use this as a starter code.
