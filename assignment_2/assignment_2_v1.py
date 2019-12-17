import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


'''
Part 1 - Regression¶
First, run the following block to set up the variables needed for later sections.
'''

np.random.seed(0)
n = 15
x = np.linspace(0, 10, n) + np.random.randn(n) / 5
y = np.sin(x) + x / 6 + np.random.randn(n) / 10

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
# print(X_train)


# You can use this function to help you visualize the dataset by
# plotting a scatterplot of the data points
# in the training and test sets.
def part1_scatter():
    plt.figure()
    plt.scatter(X_train, y_train, label='training data')
    plt.scatter(X_test, y_test, label='test data')
    plt.legend(loc=4)
    plt.show()

# NOTE: Uncomment the function below to visualize the data, but be sure
# to **re-comment it before submitting this assignment to the autograder**.
# part1_scatter()

'''
Question 1
Write a function that fits a polynomial LinearRegression model on the training data X_train for degrees 1, 3, 6, and 9. (Use PolynomialFeatures in sklearn.preprocessing to create the polynomial features and then fit a linear regression model) For each model, find 100 predicted values over the interval x = 0 to 10 (e.g. np.linspace(0,10,100)) and store this in a numpy array. The first row of this array should correspond to the output from the model trained on degree 1, the second row degree 3, the third row degree 6, and the fourth row degree 9.
'''


def answer_one():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    # Your code here
    predict = np.linspace(0, 10, 100).reshape(100, 1)
    # print(predict)
    # print(X_train)
    X_train_reshaped = X_train.reshape(X_train.size, 1)
    # print(X_train_reshaped)



    poly = PolynomialFeatures(degree=1)
    X_poly1 = poly.fit_transform(X_train_reshaped)
    y_poly1 = poly.fit_transform(predict)
    # print('\nAddition of many polynomial features often leads to\n\
    # overfitting, so we often use polynomial features in combination\n\
    # with regression that has a regularization penalty, like ridge\n\
    # regression.\n')
    linreg1 = LinearRegression().fit(X_poly1, y_train)
    array1 = linreg1.predict(y_poly1)
    # print(X_poly1)

    poly = PolynomialFeatures(degree=3)
    X_poly3 = poly.fit_transform(X_train_reshaped)
    y_poly3 = poly.fit_transform(predict)
    linreg3 = LinearRegression().fit(X_poly3, y_train)
    array3 = linreg3.predict(y_poly3)

    poly = PolynomialFeatures(degree=6)
    X_poly6 = poly.fit_transform(X_train_reshaped)
    y_poly6 = poly.fit_transform(predict)
    linreg6 = LinearRegression().fit(X_poly6, y_train)
    array6 = linreg6.predict(y_poly6)

    poly = PolynomialFeatures(degree=9)
    X_poly9 = poly.fit_transform(X_train_reshaped)
    y_poly9 = poly.fit_transform(predict)
    linreg9 = LinearRegression().fit(X_poly9, y_train)
    array9 = linreg9.predict(y_poly9)

    return np.vstack((array1, array3, array6, array9))
    # return np.vstack(arr9)

# print(answer_one())

# print(list(enumerate([1,2,3])))

# feel free to use the function plot_one() to replicate the figure
# from the prompt once you have completed question one
def plot_one(degree_predictions):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    plt.plot(X_train, y_train, 'o', label='training data', markersize=10)
    plt.plot(X_test, y_test, 'o', label='test data', markersize=10)
    # plt.plot(np.linspace(0,10,100), degree_predictions, alpha=0.8, lw=2, label='degree={}'.format(1))
    for i, degree in enumerate([1,3,6,9]):
        plt.plot(np.linspace(0,10,100), degree_predictions[i], alpha=0.8, lw=2, label='degree={}'.format(degree))
    plt.ylim(-1,2.5)
    plt.legend(loc=4)
    plt.show()
# plot_one(answer_one())



'''
Question 2
Write a function that fits a polynomial LinearRegression model on the training data X_train for degrees 0 through 9. For each model compute the  R2R2 (coefficient of determination) regression score on the training data as well as the the test data, and return both of these arrays in a tuple.

This function should return one tuple of numpy arrays (r2_train, r2_test). Both arrays should have shape (10,)
'''

best_degree = list()
def answer_two():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics.regression import r2_score

    X_train_reshaped = X_train.reshape(X_train.size, 1)
    X_test_reshaped = X_test.reshape(X_test.size, 1)
    # print(X_train_reshape)

    r2_train, r2_test = [], []

    for degree in range(0, 10):
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train_reshaped)
        X_test_poly = poly.fit_transform(X_test_reshaped)
        linreg = LinearRegression().fit(X_train_poly, y_train)
        # print('degree =', degree)
        # print(linreg.score(X_train_poly, y_train))
        # print(linreg.score(X_test_poly, y_test))

        r2_train.append(linreg.score(X_train_poly, y_train))
        r2_test.append(linreg.score(X_test_poly, y_test))
        best_degree.append((degree, linreg.score(X_train_poly, y_train), linreg.score(X_test_poly, y_test)))

    return np.asarray(r2_train), np.asarray(r2_test)
# answer_two()
# print(best_degree)
# x = []
# for degree, train_core, test_score in best_degree:
#     x.append((test_score, degree))
# print(x)
# print(max(x))
# print('best degree is {} with {} accuracy'.format(max(x)[1], max(x)[0]))
# print(max(x, key=lambda x: x[0]))



'''
Question 3
Based on the  R2R2  scores from question 2 (degree levels 0 through 9), what degree level corresponds to a model that is underfitting? What degree level corresponds to a model that is overfitting? What choice of degree level would provide a model with good generalization performance on this dataset?

Hint: Try plotting the  R2R2  scores from question 2 to visualize the relationship between degree level and  R2R2 . Remember to comment out the import matplotlib line before submission.

This function should return one tuple with the degree values in this order: (Underfitting, Overfitting, Good_Generalization). There might be multiple correct solutions, however, you only need to return one possible solution, for example, (1,2,3).
'''
# print(len(answer_two()[0]))
def answer_three():
    result = []

    for i in range(len(answer_two()[0])):
        # print(i)
        result.append((i, answer_two()[0][i], answer_two()[1][i]))


    # print(list(enumerate(zip(result[6], result[4]))))
    Underfitting = min(result, key=lambda x: x[1])[0]
    Overfitting = max(result, key=lambda x: x[1])[0]
    Good_Generalization = max(result, key=lambda x: x[2])[0]
    return (Underfitting, Overfitting, Good_Generalization)
# print(answer_three())



'''
Question 4
Training models on high degree polynomial features can result in overly complex models that overfit, so we often use regularized versions of the model to constrain model complexity, as we saw with Ridge and Lasso linear regression.

For this question, train two models: a non-regularized LinearRegression model (default parameters) and a regularized Lasso Regression model (with parameters alpha=0.01, max_iter=10000) both on polynomial features of degree 12. Return the  R2R2  score for both the LinearRegression and Lasso model's test sets.

This function should return one tuple (LinearRegression_R2_test_score, Lasso_R2_test_score)
'''
def answer_four():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Lasso, LinearRegression
    from sklearn.metrics.regression import r2_score

    X_train_reshape = X_train.reshape(X_train.size, 1)
    # print(X_train_reshape)
    X_test_reshape = X_test.reshape(X_test.size, 1)

    polynomial_features = PolynomialFeatures(degree=12)
    X_train_polynomial = polynomial_features.fit_transform(X_train_reshape)
    X_test_polynomial = polynomial_features.fit_transform(X_test_reshape)


    liniar_reg = LinearRegression().fit(X_train_polynomial, y_train)
    lasso_reg = Lasso(alpha=0.01, max_iter=10000).fit(X_train_polynomial, y_train)


    # LinearRegression_R2_test_score = liniar_reg.score(X_test_polynomial, y_test)
    LinearRegression_R2_test_score = r2_score(y_test, liniar_reg.predict(X_test_polynomial))
    # Lasso_R2_test_score = lasso_reg.score(X_test_polynomial, y_test)
    Lasso_R2_test_score = r2_score(y_test, lasso_reg.predict(X_test_polynomial))

    return (LinearRegression_R2_test_score, Lasso_R2_test_score)
# print(answer_four())





'''
Part 2 - Classification
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

pd.set_option('display.max_rows', 10)  # let print max rows
pd.set_option('display.max_columns', 1000)  # let print max columns

mush_df = pd.read_csv('mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)
# print(mush_df2)


X_mush = mush_df2.iloc[:,2:]
# print(X_mush)
y_mush = mush_df2.iloc[:,1]
# print(y_mush)

# use the variables X_train2, y_train2 for Question 5
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

# For performance reasons in Questions 6 and 7, we will create a smaller version of the
# entire mushroom dataset for use in those questions.  For simplicity we'll just re-use
# the 25% test split created above as the representative subset.
#
# Use the variables X_subset, y_subset for Questions 6 and 7.
X_subset = X_test2
y_subset = y_test2



def answer_five():
    from sklearn.tree import DecisionTreeClassifier

    tree_classifier = DecisionTreeClassifier(random_state=0).fit(X_train2, y_train2)
    df = pd.DataFrame({'feature': X_train2.columns, 'importance': tree_classifier.feature_importances_})
    # print(df)

    return df.sort_values(by=['importance'], ascending=False).feature.head(5).tolist()
# print(answer_five())


def answer_six():
    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve

    # support_vector_classifier = SVC(C=1, kernel='rbf', random_state=0).fit(X_train2, y_train2)


    # print(validation_curve(support_vector_classifier, X_subset, y_subset, param_name='gamma', param_range=np.linspace(0.0001, 10, 6)))
    # print(np.logspace(1, 60, 6))
    train_scores, test_scores = validation_curve(SVC(kernel='rbf', C=1, random_state=0), X_train2, y_train2, param_name='gamma', param_range=np.logspace(-4,1,6))
    # print(train_scores, test_scores)
    # print(type(np.array(list(map(np.mean, train_scores)))))
    # print(tuple(np.array(list(map(np.mean, train_scores)))))
    # print(type(tuple(np.array(list(map(np.mean, train_scores))))))


    # print(tuple(np.array(list(map(np.mean, train_scores))), np.array(list(map(np.mean, test_scores)))))
    # print(type(print((np.array(list(map(np.mean, train_scores))).shape, np.array(list(map(np.mean, test_scores))).shape))))
    # return (np.array(list(map(np.mean, train_scores))), np.array(list(map(np.mean, test_scores))))
# print(answer_six())


def answer_six():
    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve


    svc = SVC(C=1, kernel='rbf', random_state=0)
    gamma = np.logspace(-4, 1, 6)
    train_scores, test_scores = validation_curve(svc, X_subset, y_subset,
                                                 param_name='gamma',
                                                 param_range=gamma,
                                                 scoring='accuracy'
                                                 )
    # print(train_scores)
    # print(type(train_scores))    # array
    # print('axis 0', train_scores.mean(axis=0))
    # print('axis 1', train_scores.mean(axis=1))

    scores = (train_scores.mean(axis=1), test_scores.mean(axis=1))
    # print()
    return scores

print(answer_six())

