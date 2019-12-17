import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


'''
Question 1
Import the data from fraud_data.csv. What percentage of the observations in the dataset are instances of fraud?

This function should return a float between 0 and 1.
'''
def answer_one():
    df = pd.read_csv('fraud_data.csv')

    # print(df)
    #     print(df['Class'][df['Class'] == 1].size) # 1 is froad

    return df['Class'][df['Class'] == 1].size / df['Class'].size
# print(answer_one())


from sklearn.model_selection import train_test_split

df = pd.read_csv('fraud_data.csv')

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


'''
Question 2
Using X_train, X_test, y_train, and y_test (as defined above), train a dummy classifier that classifies everything as the majority class of the training data. What is the accuracy of this classifier? What is the recall?

This function should a return a tuple with two floats, i.e. (accuracy score, recall score).
'''


def answer_two():
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import recall_score

    dummy_clf = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
    y_dummy_predictions = dummy_clf.predict(X_test)
    # print(y_dummy_predictions)
    # print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_dummy_predictions)))
    # '='
    # print('Accuracy: {:.2f}'.format(dummy_clf.score(X_test, y_test)))
    # print('Accuracy: {:.2f}'.format(recall_score(y_test, y_dummy_predictions)))


    return (accuracy_score(y_test, y_dummy_predictions), recall_score(y_test, y_dummy_predictions))
# print(answer_two())


'''
Question 3
Using X_train, X_test, y_train, y_test (as defined above), train a SVC classifer using the default parameters. What is the accuracy, recall, and precision of this classifier?

This function should a return a tuple with three floats, i.e. (accuracy score, recall score, precision score).

'''

def answer_three():
    from sklearn.metrics import accuracy_score, recall_score, precision_score
    from sklearn.svm import SVC

    SVC_clf = SVC().fit(X_train, y_train)
    # print(SVC_clf)
    y_SVC_prediction = SVC_clf.predict(X_test)
    # print(y_SVC_prediction)
    # print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_SVC_prediction)))
    # print('Accuracy: {:.2f}'.format(recall_score(y_test, y_SVC_prediction)))
    # print('Accuracy: {:.2f}'.format(precision_score(y_test, y_SVC_prediction)))

    return  (accuracy_score(y_test, y_SVC_prediction), recall_score(y_test, y_SVC_prediction), precision_score(y_test, y_SVC_prediction))
# print(answer_three())


'''
Question 4
Using the SVC classifier with parameters {'C': 1e9, 'gamma': 1e-07}, what is the confusion matrix when using a threshold of -220 on the decision function. Use X_test and y_test.

This function should return a confusion matrix, a 2x2 numpy array with 4 integers.
'''


def answer_four():
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC

    SVC_clf = SVC(C=1e9, gamma=1e-07).fit(X_train, y_train)
    # print(SVC_clf)
    y_decision_function = SVC_clf.decision_function(X_test) > -220
    # print(len(y_decision_function))
    # print(y_decision_function)

    confusion = confusion_matrix(y_test, y_decision_function)
    # print(confusion)

    return confusion
print(answer_four())


'''
Question 5
Train a logisitic regression classifier with default parameters using X_train and y_train.

For the logisitic regression classifier, create a precision recall curve and a roc curve using y_test and the probability estimates for X_test (probability it is fraud).

Looking at the precision recall curve, what is the recall when the precision is 0.75?

Looking at the roc curve, what is the true positive rate when the false positive rate is 0.16?

This function should return a tuple with two floats, i.e. (recall, true positive rate).

'''


def answer_five():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_curve, roc_curve
    import matplotlib.pyplot as plt

    linear_reg_clf = LogisticRegression().fit(X_train, y_train)
    # print(linear_reg_clf)
    # y_scores = linear_reg_clf.score(X_test, y_test)
    # y_scores = linear_reg_clf.decision_function(X_test)
    # print(y_scores)
    y_prediction_scores = linear_reg_clf.predict(X_test)
    # print(y_prediction_scores)
    precision, recall, thresholds = precision_recall_curve(y_test, y_prediction_scores)
    fpr, tpr, _ = roc_curve(y_test, y_prediction_scores)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])

    closest_zero = np.argmin(np.abs(thresholds))
    closest_zero_p = precision[closest_zero]
    closest_zero_r = recall[closest_zero]

    ax1.plot(precision, recall, label='Precision-Recall Curve')
    ax1.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)
    ax1.set_xlabel('Precision', fontsize=16)
    ax1.set_ylabel('Recall', fontsize=16)
    # plt.axes().set_aspect('equal')

    ax2.plot(fpr, tpr, lw=3, label='LogRegr')
    ax2.set_xlabel('False Positive Rate', fontsize=16)
    ax2.set_ylabel('True Positive Rate', fontsize=16)
    plt.show()


    return  (0.83, 0.94)
# print(answer_five())



'''
Question 6
Perform a grid search over the parameters listed below for a Logisitic Regression classifier, using recall for scoring and the default 3-fold cross validation.

'penalty': ['l1', 'l2']

'C':[0.01, 0.1, 1, 10, 100]

From .cv_results_, create an array of the mean test scores of each parameter combination. i.e.

l1	l2
0.01	?	?
0.1	?	?
1	?	?
10	?	?
100	?	?


This function should return a 5 by 2 numpy array with 10 floats.

Note: do not return a DataFrame, just the values denoted by '?' above in a numpy array. You might need to reshape your raw result to meet the format we are looking for.

'''


def answer_six():
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression

    Cs = [0.01, 0.1, 1, 10, 100]
    penalty = ['l1', 'l2']
    param_grid = {'C': Cs, 'penalty': penalty}

    logistic_reg_clf = LogisticRegression().fit(X_train, y_train)
    grid_clf_logreg = GridSearchCV(logistic_reg_clf, param_grid=param_grid, scoring='recall', cv=3)
    # print(grid_clf_logreg)
    grid_clf_logreg.fit(X_train, y_train)

    # y_prediction = grid_clf_logreg.score(X_test, y_test)
    # print(y_prediction)
    # print(grid_clf_logreg.cv_results_)
    # print(grid_clf_logreg.cv_results_.keys())
    # print(grid_clf_logreg.cv_results_['mean_test_score'])
    mean_test_score = grid_clf_logreg.cv_results_['mean_test_score']
    # print(type(mean_test_score))
    print(mean_test_score.reshape(5, 2))
    print(type(mean_test_score.reshape(5, 2)))
    print(np.array(mean_test_score.reshape(5, 2)))
    print(type(np.array(mean_test_score.reshape(5, 2))))


    # return  mean_test_score.reshape(5, 2)
print(answer_six())