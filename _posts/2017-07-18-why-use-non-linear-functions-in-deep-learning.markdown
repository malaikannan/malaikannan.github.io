---
layout: "post"
title: "Why use Activation Functions in Deep Learning ?"
date: "2017-07-18 20:31"
---

Machine Learning or Deep Learning is all about using Affine Maps. Affine map is a function which can be expressed as

f(x) = WX + b

Where W and X are matrix, and b (bias term) is a vector. Deep learning learns parameters W and b. In Deep Learning you can stack multiple affine maps on top of one another. for e.g
- f(x) = WX + b
- g(x) = VX + d

If we stack one affine map over the other then

- f(g(x)) = W (VX +d) + b
- f(g(x)) = WVx + Wd + b

WV is a matrix , Wd and b are vectors.

Deep learning requires lot of affine maps stacked on top of the other. But Composing one affine map over the other gives another affine map so stacking is not going to give the desired effect and it gives nothing more than what a single affine map is going to give. It still leaves us with a linear model. In a classification problem linear model will not be able to solve for a non-linear decision boundary. 

How do we solve this ? By introducing non-linearity between affine maps/layers. Most commonly used non-linear functions are

- Tanh
- Sigmoid
- RELU

When there are lot of non-linear functions why use only the above ones ? Because the derivatives of these functions are easier to compute which is how Deep Learning algorithms learn. Non-Linear functions are called Activation functions in Deep Learning world.

You can read more about the Activation functions in [wiki](https://en.wikipedia.org/wiki/Activation_function)
