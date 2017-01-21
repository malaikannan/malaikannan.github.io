---
layout: post
title: "How to Learn Deep Learning?"
date: "2017-01-16 23:42:51 -0800"
---

DeepLearning has been getting lot of good press for solving lot of complex problems and creating an impact. I am a CTO at Datalog.ai where we solve lot of cool problems using Deep Learning. ML Researchers and Engineers use lot of Deep Learning packages like Theano, Tensorflow, Torch, Keras etc. Packages are really good but when you want to get an understanding on how Deep Learning works, it is better to go back to basics and understand how it is done. This blog is at an attempt at that, it is going to be a 3 part of series with topics being

1. DeepLearning using Numpy
2. Why TensorFlow/Theano not Numpy?
3. Why Keras not TensorFlow/Theano?

![Neural Network with 1 hidden layer](assets/2017-01-21-DeepLearning Using Numpy/first-project-idea-1.png)

Deep learning refers to artificial neural networks that are composed of many layers like the one shown above. Deep Learning has many flavor's like Convolution Neural Networks, Recurrent Neural Networks, Reinforcement Learning, Feed Forward Neural Network etc. This blog is going to take the simplest of them Feed Forward Neural network as an example to explain.

Machine Learning deals with lot of Linear Algebra operations like dot product, transpose, reshape etc. If you are not familiar with it, I would suggest refer to my previous blog post [All about Math](http://malaikannan.io//2017/01/17/how-to-learn-deep-learning/) section.

{% gist 1b09fb8c64f25ca8d57df325d3aa28d6 %}

Deep Learning needs an activation function to squish real numbers to probability values between 0 and 1 , there are different activation functions like sigmoid, Tanh, RELU etc. For this toy example i have used sigmoid activation function.

![Sigmoid](assets/2017-01-21-DeepLearning Using Numpy/sigmoid.gif)
