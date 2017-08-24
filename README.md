# Naive Bayes Digit Classifier
[View the Jupyter Notebook](https://nbviewer.jupyter.org/github/Carpetfizz/naive_bayes_classifier/blob/master/Naive%20Bayes%20Digit%20Classifier.ipynb)

![Camera](http://i.imgur.com/ufH0sdc.jpg)

Trying to classify digits by trying to teach a computer specific rules about what a 1 or a 2 looks like is difficult and does not scale well. Try writing a 5 multiple times on a piece of paper, do they all appear exactly the same? It is difficult to classify your own handwriting, let alone classify the handwriting of some arbitrary person. Instead we will use probability theory, specifically Bayes Rule, to build a simple classifier.

The `camera_demo` is simply a frontend to consume the trained classifier which is detailed in the included Jupyter Notebook. Although Naive Bayes Classifiers aren't very good to begin with (achieves ~84% accuracy on MNIST dataset), the `camera_demo` can be improved with better pre-processing. For now, it's pretty bad.
