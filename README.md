# **ML Classifier Applications on Word2VecData**

#### Introduction 

For this report we will focus on using pre-recorded data to make predictions on whether a review will be positive or negative. To achieve this, we will use multiple machine learning classifiers that will use a text embedding dataset, which is essentially a series of vectors acquired from the word2vec model, and categorize it based on the vector’s specifications. In order to minimize data discrepancies or biases, the data was split into a training and testing set, where we will use the training data to predict the testing data. Some of the classifiers utilized include: K-Nearest Neighbor, Perceptron, Logistic Regression and SVM.

#### *Preprocessing Data*
With the use of the function ” readTrainTestData(trainFile, testFile)”, we where able to take the training and testing data files, convert each line to a string and then convert it again to a numerical data type. After the file conversion, loops where used to separate each file into two separate ones, one being the labels and the other word2vec vector. As a result we ended up with four separate datasets: Train_vec, Train_lab(label), Test_vec and Test_lab.
In the case of cross validation, we combined the *vec* and *lab* datasets to form X_val and y_val.

#### *K-Nearest Neighbor*
This is a classification algorithm widely used in machine learning. As a supervised model, previous input is needed in order for the algorithm to classify future objects. So based on our model “sentimentClassKNN(train_vec, train_lab, test_vec, test_lab, k, distanceMeasure)” we can get a hint of what is required for the algorithm to work. To break it down, KNN takes multiple labelled data points and then determines the label of a new point based on the proximity of the established points around it. This is a where “k” come into play because it dictates the amount of points it checks before deciding the label.

