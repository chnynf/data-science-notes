---
title: Data Science Notes
date: 2021-08-09 00:50:30
updated: 2021-08-09
tags:
---

## Intro
This is going to be a place where I (slowly) update my data science notes to keep my memory refreshed. I've created another Github reporsitory chnynf/data-science-notes to host this markdown file and use Github Actions to automatically push the md file to this post. I'll be glad if anyone can contribute together to the notes. Most of the snippets here will be the ones I pick up from the internet, from me and my friend's notes, and from the book All of Statistics: A Concise Course in Statistical Inference, from Larry A. Wasserman.

---

## Random Variables
A random variable is a mapping {% katex %}X : \Omega \rightarrow \mathbb{R}{% endkatex %} that assigns a real number {% katex %}X(\omega){% endkatex %} to each outcome {% katex %}\omega{% endkatex %}.
### Some important discrete random variables
**The point mass distribution:**

{% katex %}
F(x) = \left\{ \begin{array}{l}
     0 \ \ \ x< a \\
     1\ \ \ x\geq a
 \end{array} \right.
{% endkatex %}

**The discrete uniform distribution:**

{% katex %}
f(x) = \left\{ \begin{array}{l}
     1/k \ \ \ for \ x = 1,...,k \\
     0\ \ \ otherwise.
 \end{array} \right.
{% endkatex %}

**The bernoulli distribution:**

Let X represent a coin flip. Then P(X = 1) = p and P(X = 0) = 1-p for some p between 0 and 1. We say that X has a Bernoulli distribution written 
{% katex %}
X \sim Bernoulli(p)
{% endkatex %}
The probability function is:
{% katex %}
f(x) = p^x(1-p)^{1-x}
{% endkatex %}
for {% katex %} x \in \left \{ 0, 1 \right \} {% endkatex %}

**The binomial distribution:**

Flip the coin n times and let X be the number of heads, then 
{% katex %}
X \sim Binomial(p)
{% endkatex %}
{% katex %}
f(x) = \left\{ \begin{array}{l}
     \binom{n}{x}p^x(1-p)^{n-x} \ \ \ for \ x = 0,...,n \\
     0\ \ \ otherwise.
 \end{array} \right.
{% endkatex %}

 (Sum of binomials are also binomials.)


---
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
