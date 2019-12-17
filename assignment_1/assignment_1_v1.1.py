import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
print(cancer.keys())


'Question 0'
# You should write your whole answer within the function provided. The autograder will call
# this function and compare the return value against the correct solution value
def answer_zero():
    # This function returns the number of features of the breast cancer dataset, which is an integer.
    # The assignment question description will tell you the general format the autograder is expecting
    return len(cancer['feature_names'])

# You can examine what your function returns by calling it in the cell. If you have questions
# about the assignment formats, check out the discussion forums for any FAQs
print(answer_zero())


'Question 1'
'''
Scikit-learn works with lists, numpy arrays, scipy-sparse matrices, and pandas DataFrames, so converting the dataset to a DataFrame is not necessary for training this model. Using a DataFrame does however help make many things easier such as munging data, so let's practice creating a classifier with a pandas DataFrame.
Convert the sklearn.dataset cancer to a DataFrame.
This function should return a (569, 31) DataFrame with
columns =
['mean radius', 'mean texture', 'mean perimeter', 'mean area',
'mean smoothness', 'mean compactness', 'mean concavity',
'mean concave points', 'mean symmetry', 'mean fractal dimension',
'radius error', 'texture error', 'perimeter error', 'area error',
'smoothness error', 'compactness error', 'concavity error',
'concave points error', 'symmetry error', 'fractal dimension error',
'worst radius', 'worst texture', 'worst perimeter', 'worst area',
'worst smoothness', 'worst compactness', 'worst concavity',
'worst concave points', 'worst symmetry', 'worst fractal dimension',
'target']
and index =

RangeIndex(start=0, stop=569, step=1)

'''

def answer_one():
    # pass
    df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
    df['target'] = pd.Series(cancer['target'])
    # print(cancer['target'])
    return df
# print(answer_one())


'''
Question 2
What is the class distribution? (i.e. how many instances of malignant (encoded 0) and how many benign (encoded 1)?)
This function should return a Series named target of length 2 with integer values and index = ['malignant', 'benign']
'''
def answer_two():
    cancerdf = answer_one()
    # print(cancer['target_names'])
    # print(cancerdf.target[cancerdf.target == 1].count())
    malignant = cancerdf.target[cancerdf.target == 0].count()
    benign = cancerdf.target[cancerdf.target == 1].count()
    target = pd.Series([malignant, benign], index=['malignant', 'benign'])
    return target
# print(answer_two())


'''
Question 3
Split the DataFrame into X (the data) and y (the labels).
This function should return a tuple of length 2: (X, y), where
X, a pandas DataFrame, has shape (569, 30)
y, a pandas Series, has shape (569,).
'''

def answer_three():
    cancerdf = answer_one()
    X = cancerdf.iloc[:, :-1]
    y = cancerdf.iloc[:, -1]
    print(X.shape, y.shape)
    return X, y
# print(answer_three())



'''
Question 4
Using train_test_split, split X and y into training and test sets (X_train, X_test, y_train, and y_test).
Set the random number generator state to 0 using random_state=0 to make sure your results match the autograder!
This function should return a tuple of length 4: (X_train, X_test, y_train, y_test), where

X_train has shape (426, 30)
X_test has shape (143, 30)
y_train has shape (426,)
y_test has shape (143,)
'''

from sklearn.model_selection import train_test_split

def answer_four():
    X, y = answer_three()
    'Use it to split by samples! random_state = 0 means 75% train and 25% test shares'
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X_train, X_test, y_train, y_test
# print(answer_four())

'''
Question 5
Using KNeighborsClassifier, fit a k-nearest neighbors (knn) classifier with X_train, y_train and using one nearest neighbor (n_neighbors = 1).

This function should return a  sklearn.neighbors.classification.KNeighborsClassifier.
'''

from sklearn.neighbors import KNeighborsClassifier

def answer_five():
    X_train, X_test, y_train, y_test = answer_four()
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    print('Accuracy score ==> ', knn.score(X_test, y_test))
    return knn
print(answer_five())


'''
Question 6
Using your knn classifier, predict the class label using the mean value for each feature.
Hint: You can use cancerdf.mean()[:-1].values.reshape(1, -1) which gets the mean value for each feature, ignores the target column, and reshapes the data from 1 dimension to 2 (necessary for the precict method of KNeighborsClassifier).
This function should return a numpy array either array([ 0.]) or array([ 1.])
'''

def answer_six():
    cancerdf = answer_one()     # => DataFrame
    knn = answer_five()     # knn classifier

    # print(cancerdf.mean()[:-1].values.reshape(1, -1))
    # print('Type ==>', type(cancerdf))
    #
    # means_series = cancerdf.mean()[:-1]     # => mean() converts DataFrame to Series
    # # print(means_series)
    # print('Type ==>', type(means_series))
    #
    # means_values = cancerdf.mean()[:-1].values    # => values converts Series to numpy array
    # # print(means_values)
    # print('Type ==>', type(means_values))
    # print(means_values.shape)

    means = cancerdf.mean()[:-1].values.reshape(1, -1)
    '''
    numpy allow us to give one of new shape parameter as -1 (eg: (2,-1) or (-1,3) but not (-1, -1)). It simply means that it is an unknown dimension and we want numpy to figure it out
    New shape as (1,-1). i.e, row is 1, column unknown
    if we try to provide both dimension as unknown i.e new shape as (-1,-1). It will throw an error.
    can only specify one unknown dimension
    '''
    # https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape
    # print(means.shape)

    prediction = knn.predict(means)
    # print(type(prediction))
    return prediction
# print(answer_six()


'''
Question 7
Using your knn classifier, predict the class labels for the test set X_test.
This function should return a numpy array with shape (143,) and values either 0.0 or 1.0.
'''

def answer_seven():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    prediction = knn.predict(X_test)

    # print(type(prediction))
    # print(prediction.shape)
    # print(prediction)
    return prediction
# print(answer_seven())



'''
Question 8
Find the score (mean accuracy) of your knn classifier using X_test and y_test.
This function should return a float between 0 and 1
'''

def answer_eight():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    score = knn.score(X_test, y_test)
    return score
# print(answer_eight())

import warnings
warnings.filterwarnings('ignore')
def accuracy_plot():
    import matplotlib.pyplot as plt
    X_train, X_test, y_train, y_test = answer_four()

    # Find the training and testing accuracies by target value (i.e. malignant, benign)
    # print(X_train.head())
    # print(y_train.head())
    mal_train_X = X_train[y_train==0]
    # print(mal_train_X.head())
    mal_train_y = y_train[y_train==0]
    # print(mal_train_y.head())
    ben_train_X = X_train[y_train==1]
    ben_train_y = y_train[y_train==1]

    mal_test_X = X_test[y_test==0]
    mal_test_y = y_test[y_test==0]
    ben_test_X = X_test[y_test==1]
    ben_test_y = y_test[y_test==1]

    knn = answer_five()

    scores = [knn.score(mal_train_X, mal_train_y), knn.score(ben_train_X, ben_train_y),
              knn.score(mal_test_X, mal_test_y), knn.score(ben_test_X, ben_test_y)]

    plt.figure()
    # Plot the scores as a bar chart
    bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])


    # directly label the score onto the bars
    for bar in bars:
        height = bar.get_height()
        plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2), ha='center', color='w', fontsize=11)

    # remove all the ticks (both axes), and tick labels on the Y axis
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')


    for object in plt.gca().get_children():
        print(object)

    print()

    ' Delete frame except the bottom line (x axis)'

    for spine in plt.gca().spines:
        print(spine)
        if spine != 'bottom':
            plt.gca().spines[spine].set_visible(False)

    plt.xticks([0,1,2,3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8)
    plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)
    plt.show()
accuracy_plot()









