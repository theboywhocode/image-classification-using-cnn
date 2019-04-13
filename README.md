# image-classification-using-cnn
**Introduction**

This repository is about the image classification using deep learning. To implement the image classification I have worked on the CIFAR 10 dataset which contains 10 class, toatal 60000 images of size 32 x 32 . [Download the dataset from here.](https://www.cs.toronto.edu/~kriz/cifar.html)

**Model Architecture**

Layer 1:
- Convolution with 32 different filters in size of (3x3) 
- Max Pooling by 2 
- ReLU activation function 
- Batch Normalization 

Layer 2:
- Convolution with 64 different filters in size of (3x3)
- Max Pooling by 2 
- ReLU activation function 
- Batch Normalization 

Layer 3:
- Convolution with 128 different filters in size of (3x3) 
- Max Pooling by 2 
- ReLU activation function 
- Batch Normalization 

Layer 4:
- Convolution with 256 different filters in size of (3x3) 
- Max Pooling by 2 
- ReLU activation function 
- Batch Normalization 

Layer 5:
- Convolution with 512 different filters in size of (3x3) 
- Max Pooling by 2 
- ReLU activation function 
- Batch Normalization 

Flattening the 3-D output of the last convolving operations. 

Dense Layer with 1024 units 
- Dropout(0.2)
- Batch Normalization 

Dense Layer with 512 units 
- Dropout(0.3)
- Batch Normalization 

Dense Layer with 256 units 
- Dropout (0.4)
- Batch Normalization 

Dense Layer with 128 units 
- Dropout (0.5)
- Batch Normalization 

Dense Layer with 10 units (number of image classes)

**Optimizer : [opt_rms](https://keras.io/optimizers/)**

**Loss : [categorical_crossentropy](https://keras.io/losses/)**

