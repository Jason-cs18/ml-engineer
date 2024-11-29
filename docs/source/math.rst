=================
Math notations
=================

In this post, I summarize some widely-used mathematical notations in machine learning.

Datasets
--------
Dataset :math:`S=\{ \mathbf{z}_i \}^n_{i=1}=\{(\mathbf{x}_i, \mathbf{y}_i) \}^n_{i=1}` is sampled from a distribution :math:`\mathcal{D}` over a domain :math:`\mathcal{Z} = \mathcal{X} \times \mathcal{Y}`. 

- :math:`\mathcal{X}` is the instances domain (a set)
- :math:`\mathcal{Y}` is the label domain (a set)
- :math:`\mathcal{Z}=\mathcal{X} \times \mathcal{Y}` is the example domain (a set)

Usually, :math:`\mathcal{X}` is a subset of :math:`\mathbb{R}^d` and :math:`\mathcal{Y}` is a subset of :math:`\mathbb{R}^{d_\text{o}}`, where :math:`d` is the input dimension, :math:`d_\text{o}` is the ouput dimension.

:math:`n=$#$S` is the number of samples. Wihout specification, :math:`S` and :math:`n` are for the training set.

Function
--------
A hypothesis space is denoted by :math:`\mathcal{H}`. A hypothesis function is denoted by :math:`f_{\mathbf{\theta}}(\mathbf{x})\in\mathcal{H}` or :math:`f(\mathbf{x};\mathbf{\theta})` with :math:`f_{\mathbf{\theta}}:\mathcal{X}\to\mathcal{Y}`.

:math:`\mathbf{\theta}` denotes the set of parameters of :math:`f_{\mathbf{\theta}}`.

If there exists a target function, it is denoted by :math:`f^{*}` or :math:`f^{*}:\mathcal{X}\to\mathcal{Y}` satisfying :math:`\mathbf{y}_i=f^*(\mathbf{x}_i)` for :math:`i=1,\dots,n`.


Loss function
-------------

A loss function, denoted by :math:`\ell:\mathcal{H}\times\mathcal{Z}\to\mathbb{R}_{+}:=[0,+\infty)` measures the difference between a predicted label and a true label, e.g.,

- :math:`L^2` loss: :math:`\ell(f_{\mathbf{\theta}},\mathbf{z})=(f_{\mathbf{\theta}}(\mathbf{x})-\mathbf{y})^2`, where :math:`\mathbf{z}=(\mathbf{x},\mathbf{y})`. :math:`\ell(f_{\mathbf{\theta}},\mathbf{z})` can also be written as :math:`\ell(f_{\mathbf{\theta}},\mathbf{y})` for convenience.

Empirical risk or training loss for a set :math:`S={(\mathbf{x}_i,\mathbf{y}_i)}^n_{i=1}` is denoted by :math:`L_S(\mathbf{\theta})` or :math:`L_n(\mathbf{\theta})` or :math:`R_S(\mathbf{\theta})` or :math:`R_n(\mathbf{\theta})`,

.. math::
  L_S(\mathbf{\theta})=\frac{1}{n}\sum^n_{i=1}\ell(f_{\mathbf{\theta}}(\mathbf{x}_i),\mathbf{y}_i).


The population risk or expected loss is denoted by :math:`L_{\mathcal{D}}` or :math:`R_{\mathcal{D}}`,

.. math::
  L_{\mathcal{D}}(\mathbf{\theta})=\mathbb{E}_{\mathcal{D}}\ell(f_{\mathbf{\theta}}(\mathbf{x}),\mathbf{y}),


where :math:`\mathbf{z}=(\mathbf{x},\mathbf{y})` follows the distribution :math:`\mathcal{D}`.


Activations
-----------

An activation function is denoted by :math:`\sigma(x)`.

**Example 1**. Some commonly used activation functions are

- :math:`\sigma(x)=\text{ReLU}(x)=\text{max}(0,x)`
- :math:`\sigma(x)=\text{sigmoid}(x)=\dfrac{1}{1+e^{-x}}`
- :math:`\sigma(x)=\tanh(x)`
- :math:`\sigma(x)=\cos x, \sin x`


Two-layer neural network
-------------------------

The neuron number of the hidden layer is denoted by :math:`m`, The two-layer neural network is

.. math::
  f_{\mathbf{\theta}}(\mathbf{x})=\sum^{m}_{j=1}a_j\sigma(\mathbf{w}_j\cdot\mathbf{x}+b_j),


where :math:`\sigma` is the activation function, :math:`\mathbf{w}_j` is the input weight, :math:`a_j` is the output weight, :math:`b_j` is the bias term. We denote the set of parameters by

.. math::
  \mathbf{\theta}=(a_1,\ldots,a_m,\mathbf{w}_1,\ldots,\mathbf{w}_m,b_1,\cdots,b_m).


General deep neural network
----------------------------

The counting of the layer number excludes the input layer. An :math:`L`-layer neural network is denoted by

.. math::
  f_{\mathbf{\theta}}(\mathbf{x})=\mathbf{W}^{[L-1]}\sigma\circ(\mathbf{W}^{[L-2]})\sigma\circ(\cdots(\mathbf{W}^{[1]}\sigma\circ(\mathbf{W}^{[0]}\mathbf{x}+\mathbf{b}^{[0]})+\mathbf{b}^{[1]})\cdots)+\mathbf{b}^{[L-2]})+\mathbf{b}^{[L-1]},


where :math:`\mathbf{W}^{[l]}\in\mathbb{R}^{m_{l+1}\times m_l}`, :math:`\mathbf{b}^{[l]}=\mathbb{R}^{m_{l+1}}`, :math:`m_0=d_\text{in}=d`, :math:`m_{L}=d_\text{o}`, :math:`\sigma` is a scalar function and ":math:`\circ`" means entry-wise operation. We denote the set of parameters by

.. math::
  \mathbf{\theta}=(\mathbf{W}^{[0]},\mathbf{W}^{[1]},\dots,\mathbf{W}^{[L-1]},\mathbf{b}^{[0]},\mathbf{b}^{[1]},\dots,\mathbf{b}^{[L-1]}).


This can also be defined recursively,

.. math::
  f^{[0]}_{\mathbf{\theta}}(\mathbf{x})=\mathbf{x},


.. math::
  f^{[l]}_{\mathbf{\theta}}(\mathbf{x})=\sigma\circ(\mathbf{W}^{[l-1]}f^{[l-1]}_{\mathbf{\theta}}(\mathbf{x})+\mathbf{b}^{[l-1]}), \quad 1\le l\le L-1,


.. math::
  f_{\mathbf{\theta}}(\mathbf{x})=f^{[L]}_{\mathbf{\theta}}(\mathbf{x})=\mathbf{W}^{[L-1]}f^{[L-1]}_{\mathbf{\theta}}(\mathbf{x})+\mathbf{b}^{[L-1]}, \quad 1\le l\le L-1.

Others
-------

The Gradient Descent is oftern denoted by GD. THe Stochastic Gradient Descent is often denoted by SGD.

A batch set is denoted by :math:`B` and the batch size is denoted by :math:`|B|`.

The learning rate is denoted by :math:`\eta`.

The convolution operation is denoted by :math:`*`.