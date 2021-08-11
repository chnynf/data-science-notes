---
title: Data Science Notes
date: 2021-08-09 00:50:30
updated: 2021-08-09
tags:
---

## Intro
This is going to be a place where I (slowly) update my data science notes to keep my memory refreshed. I've created another Github reporsitory chnynf/data-science-notes to host this markdown file and use Github Actions to automatically push the md file to this post. I'll be glad if anyone can contribute together to the notes. Most of the snippets here will be the ones I pick up from the internet, from me and my friend's notes, and from the book All of Statistics: A Concise Course in Statistical Inference, from Larry A. Wasserman.

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
F(x) = \left\{ \begin{array}{l}
     1/k \ \ \ for x = 1,...,k \\
     0\ \ \ otherwise.
 \end{array} \right.
{% endkatex %}



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
