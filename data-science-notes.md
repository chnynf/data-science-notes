---
title: Data Science Notes
date: 2021-08-09 00:50:30
updated: 2021-09-19
tags:
---

## Intro
This is going to be a place where I (slowly) update my data science notes to keep my memory refreshed. I've created another Github reporsitory chnynf/data-science-notes to host this markdown file and use Github Actions to automatically push the md file to this post. I'll be glad if anyone can contribute together to the notes. Most of the snippets here will be the ones I pick up from the internet, from me and my friend's notes, and from the book All of Statistics: A Concise Course in Statistical Inference, from Larry A. Wasserman.

---

&nbsp;

&nbsp;

## Random Variables
A random variable is a mapping {% katex %}X : \Omega \rightarrow \mathbb{R}{% endkatex %} that assigns a real number {% katex %}X(\omega){% endkatex %} to each outcome {% katex %}\omega{% endkatex %}.

### Some important discrete random variables
**The Point Mass Distribution:**

{% katex %}
F(x) = \left\{ \begin{array}{l}
     0 \ \ \ x< a \\
     1\ \ \ x\geq a
 \end{array} \right.
{% endkatex %}

**The Discrete Uniform Distribution:**

{% katex %}
f(x) = \left\{ \begin{array}{l}
     1/k \ \ \ for \ x = 1,...,k \\
     0\ \ \ otherwise.
 \end{array} \right.
{% endkatex %}

**The Bernoulli Distribution:**

{% katex %}
X \sim Bernoulli(p)
{% endkatex %}
The probability function is:
{% katex %}
f(x) = p^x(1-p)^{1-x}
{% endkatex %} for {% katex %} x \in \left \{ 0, 1 \right \} {% endkatex %}

**The Binomial Distribution:**

Flip the coin n times and let X be the number of heads, then 
{% katex %}
X \sim Binomial(p)
{% endkatex %}

The probability function is:

{% katex %}
f(x) = \left\{ \begin{array}{l}
     \binom{n}{x}p^x(1-p)^{n-x} \ \ \ for \ x = 0,...,n \\
     0\ \ \ otherwise.
 \end{array} \right.
{% endkatex %}

 (Sum of binomials are also binomials.)

**The Geometric Distribution:**
 
The number of flips needed until the first heads when flipping a coin:
{% katex %}
X \sim Geom(p)
{% endkatex %}

The probability function is:

{% katex %}
\mathbb{P}(X-k) = p(1-p)^{k-1}, \ \ k\geq 1
{% endkatex %}

**The Poisson Distribution:**

Count of rare events like radioactive decay and traffic accidents:
{% katex %}
X \sim Poisson(\lambda)
{% endkatex %}

The probability function is:

{% katex %}
f(x) = e^{-\lambda}\frac{\lambda^x}{x!} \ \ \ x \geq 0
{% endkatex %}

(Sum of Poissons are also Poissons.)

&nbsp;

### Some important continuous random variables
**The Uniform Distribution:**
{% katex %}
f(x) = \left\{ \begin{array}{l}
     \frac{1}{b-a} \ \ \ for \ x \in [a,b] \\
     0\ \ \ otherwise.
 \end{array} \right.
{% endkatex %}

**The Normal (Gaussian) Distribution:**
{% katex %}
X \sim N(\mu, \sigma^2)
{% endkatex %}

The probability function is:

{% katex %}
f(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left \{-\frac{1}{2\sigma^2}(x-\mu)^2  \right \}
{% endkatex %}

**The Exponential Distribution:**
Usually used to model the lifetimes of electronic components and the waiting times between rare events:
{% katex %}
X \sim Exp(\beta)
{% endkatex %}

The probability function is:
{% katex %}
f(x) = \frac{1}{\beta}e^{-x/\beta}, \ \ \ x > 0
{% endkatex %}

**The Gamma Distribution:**
For alpha > 0, the Gamma function is defined as: 
{% katex %}
\Gamma(\alpha) = \int_{0}^{\infty}y^{a-1}e^{-y}dy
{% endkatex %}

X has a Gamma distribution denoted by:
{% katex %}
X \sim Gamma(\alpha, \beta)
{% endkatex %}

if:

{% katex %}
f(x) = \frac{1}{\beta^\alpha\Gamma(\alpha)}x^{\alpha-1}e^{-x/\beta}, \ \ x > 0, \ \ \alpha,\beta > 0
{% endkatex %}

The exponential distribution is a just a Gamma distribution with alpha equal 1.

**The Beta Distribution:**
{% katex %}
X \sim Beta(\alpha, \beta)
{% endkatex %}

if:
{% katex %}
f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}
{% endkatex %}

in which Beta function is defined as:
{% katex %}
B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha + \beta)}
{% endkatex %}

**The t and Cauchy Distribution:**
{% katex %}
X \sim t_\nu 
{% endkatex %}

if:
{% katex %}
f(x) = \frac{\Gamma(\frac{\nu+1}{2})}{\Gamma(\frac{\nu}{2})}\frac{1}{(1+\frac{x^2}{\nu})^{(\nu+1)/2}}
{% endkatex %}

The t distribution is similar to a Normal but it has thicker tails. In fact, the Normal corresponds to a t with nu equaling infinity. The Cauchy distribution is a special case of the t distribution corresponding to a t with nu of 1. The density is:
{% katex %}
f(x) = \frac{1}{\pi(1+x^2)}
{% endkatex %}

**The Chi-Square Distribution:**
{% katex %}
X \sim \chi^2_p
{% endkatex %}

if:
{% katex %}
f(x) = \frac{x^{(p/2)-1}e^{-x/2}}{\Gamma(p/2)2^{p/2}}, \ \ x > 0
{% endkatex %}

If Z_1, ..., Z_p are independent standard Normal random variables then 
{% katex %}
\sum^p_{i=1}Z_i^2 \sim \chi_p^2
{% endkatex %}

---

&nbsp;

&nbsp;

## Machine Learning and Deep Learning

### Activation Functions
**Sigmoid**
- 0 to 1
- Lose gradient at both ends
- Computation is exponential term

**Tanh**
- -1 to 1 (centered at 0)
- Lose gradient at both ends
- Still computationally heavy

**ReLU**
- No saturation on positive end
- Can cause dead neuron (if x <= 0)
- Cheap to compute

**Leaky ReLU**
- Learnable parameter
- No saturation
- No dead neuron
- Still cheap to compute

*No one activation is best. ReLU is typical starting point. Sigmoid is typically avoided in deep learning due to the computation cost.

&nbsp;

### Back Propagation
![](/data-science-notes/backpropagation_example.png)

&nbsp;

### Initialization
Ideally, we'd like to maintain the variance at the output to be similar 
to that of input.

**Xavier Initialization**

![](/data-science-notes/xavier_initialization.png)

In practice, simpler versions perform empirically well:
For tanh or similar activations:
{% katex %}
N(0, 1) * \sqrt{\frac{1}{n_j}}
{% endkatex %}

For ReLu activations: 
{% katex %}
N(0, 1) * \sqrt{\frac{1}{n_j/2}}
{% endkatex %}

&nbsp;

### Optimizers

![](/data-science-notes/optimizers.gif)

&nbsp;

### Regularization
**L1**

{% katex %}
L = |y - Wx_i|^2 + \lambda|W| 
{% endkatex %}

*Better for feature selection.

**L2**

{% katex %}
L = |y - Wx_i|^2 + \lambda|W|^2
{% endkatex %}

**Dropout Regularization**

For each node, keep its output with probability p; Activations of deactivated nodes are essentially zero. During testing, no nodes are dropped. During test time, scale outputs (or equivalently weights) by p, so that train and test-time input/output can have similar distributions.



&nbsp;

### Evaluation
![](/data-science-notes/confusion_matrix.png)

&nbsp;

### Sizes of convolution and pooling layer outputs
![](/data-science-notes/cnn_size_example.png)

- Valid convolution -- no padding
- Same convolution -- enough padding added maintain same input and output size
- Full convolution -- enough zeros are added for every pixel to be visited k times in each direction, resulting in an output image of width m + k − 1

![](/data-science-notes/types_of_paddings.gif)


---

&nbsp;

&nbsp;

## Miscellaneous
### Statistics / Data Mining Dictionary

| Statistics             | Computer Science      | Meaning                                                                                                          |
| ---------------------- | --------------------- | ---------------------------------------------------------------------------------------------------------------- |
| estimation             | learning              | using data to estimate an unknown quantity                                                                       |
| classification         | supervised learning   | predicting a discrete Y from {% katex %}X \in \chi{% endkatex %}                                                 |
| clustering             | unsupervised learning | putting data into groups                                                                                         |
| data                   | sample                |                                                                                                                  |
| covariates             | features              | the {% katex %}X_i{% endkatex %}s                                                                                |
| classifier             | hypothesis            | a map from covariates to outcomes                                                                                |
| hypothesis             |                       | subset of a parameter space {% katex %}\theta{% endkatex %}                                                      |
| confidence interval    |                       | interval that contains unknown quantity with a predicted frequency                                               |
| directed acyclic graph | Bayes net             | multivariate distribution with specified conditional independence relations                                      |
| Bayesian inference     | Bayesian inference    | statistical methods for using data to update subjective beliefs                                                  |
| frequentist inference  |                       | statistical methods for producing point estimates and confidence intervals with guarantees on frequency behavior |
| large deviation bounds | PAC learning          | uniform bounds on probability of errors                                                                          |


### Common Derivatives
![](/data-science-notes/common_derivatives.png)
![](/data-science-notes/derivative_rules.png)