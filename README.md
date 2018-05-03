# pima_indians_diabetes
The solution for below problem
https://machinelearningmastery.com/spot-check-classification-machine-learning-algorithms-python-scikit-learn/

<h1>Problem description </h1>

Spot-checking is a way of discovering which algorithms perform well on your machine learning problem.

You cannot know which algorithms are best suited to your problem before hand. You must trial a number of methods and focus attention on those that prove themselves the most promising.

In this post you will discover 6 machine learning algorithms that you can use when spot checking your classification problem in Python with scikit-learn.

Let’s get started.

Update Jan/2017: Updated to reflect changes to the scikit-learn API in version 0.18.
Update March/2018: Added alternate link to download the dataset as the original appears to have been taken down.
Spot-Check Classification Machine Learning Algorithms in Python with scikit-learn
Spot-Check Classification Machine Learning Algorithms in Python with scikit-learn
Photo by Masahiro Ihara, some rights reserved

Algorithm Spot Checking
You cannot know which algorithm will work best on your dataset before hand.

You must use trial and error to discover a short list of algorithms that do well on your problem that you can then double down on and tune further. I call this process spot checking.

The question is not:

What algorithm should I use on my dataset?

Instead it is:

What algorithms should I spot check on my dataset?

You can guess at what algorithms might do well on your dataset, and this can be a good starting point.

I recommend trying a mixture of algorithms and see what is good at picking out the structure in your data.

Try a mixture of algorithm representations (e.g. instances and trees).
Try a mixture of learning algorithms (e.g. different algorithms for learning the same type of representation).
Try a mixture of modeling types (e.g. linear and nonlinear functions or parametric and nonparametric).
Let’s get specific. In the next section, we will look at algorithms that you can use to spot check on your next machine learning project in Python.

Algorithms Overview
We are going to take a look at 6 classification algorithms that you can spot check on your dataset.

2 Linear Machine Learning Algorithms:

Logistic Regression
Linear Discriminant Analysis
4 Nonlinear Machine Learning Algorithms:

K-Nearest Neighbors
Naive Bayes
Classification and Regression Trees
Support Vector Machines

