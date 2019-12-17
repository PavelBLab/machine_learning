import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import warnings
warnings.filterwarnings('ignore')

start_time = time.time()
fruits = pd.read_table('fruit_data_with_colors.txt')
print(fruits.head())
# print('Len of the DataFrame -> ', len(fruits))
# print(fruits.shape)
# print(fruits.columns)

print('='.center(60, '='))
lookup_fruit_name = dict(zip(fruits['fruit_label'].unique(), fruits.fruit_name.unique()))
print(lookup_fruit_name)

X = fruits[['mass', 'width', 'height','color_score']]
y = fruits['fruit_label']
# print(X)

print('='.center(60, '='))
# ### Create train-test split
# For this example, we use the mass, width, and height features of each fruit instance
'Use it to split by samples'
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0) # random_state = 0 means 75% train and 25% test shares
print(X_train.head())
print('=============')
print(y_train.head())

from matplotlib import cm
from pandas.plotting import scatter_matrix
cmap = cm.get_cmap('gnuplot')
## scatter = pd.scatter_matrix(X_train, c = y_train, marker = 'o', s = 40, hist_kwds = {'bins':15}, figsize = (9,9), cmap = cmap)

# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1, projection = '3d')
#
# ax.scatter(X_train['width'], X_train['height'], X_train['color_score'], c = y_train, marker = 'o', s = 50)
#
# ax.set_xlabel('width')
# ax.set_ylabel('height')
# ax.set_zlabel('color_score')
# plt.show()


'=============== Create classifier object ======================='
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
# print(knn)

# ### Train the classifier (fit the estimator) using the training data
knn.fit(X_train[['mass', 'width', 'height']], y_train)
# print(knn.fit(X_test[['mass', 'width', 'height']], y_train))

# ### Estimate the accuracy of the classifier on future data, using
knn.score(X_test[['mass', 'width', 'height']], y_test)
print('Accuracy', knn.score(X_test[['mass', 'width', 'height']], y_test))

# ### Use the trained k-NN classifier model to classify new, previously unseen objects
# first example: a small fruit with mass 20g, width 4.3 cm, height 5.5 cm, color. Important length of list in prediction should be equl with number of columns in X_test
fruit_prediction = knn.predict([[20, 4.3, 5.5]])
# print(fruit_prediction)
print(lookup_fruit_name[fruit_prediction[0]])

# second example: a larger, elongated fruit with mass 100g, width 6.3 cm, height 8.5 cm
fruit_prediction = knn.predict([[100, 6.3, 8.5]])
print(lookup_fruit_name[fruit_prediction[0]])

# ### Plot the decision boundaries of the k-NN classifier
# from adspy_shared_utilities import plot_fruit_knn

# plot_fruit_knn(X_train, y_train, 5, 'uniform')   # we choose 5 nearest neighbors


'============= How sensitive is k-NN classification accuracy to the choice of the "k" parameter? ============='
k_range = range(1,20)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20])
plt.show()

### How sensitive is k-NN classification accuracy to the train/test split proportion?
t = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
knn = KNeighborsClassifier(n_neighbors = 5)

plt.figure()
for s in t:
    scores = []
    for i in range(1, 1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1-s)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    plt.plot(s, np.mean(scores), 'bo')

plt.xlabel('Training set proportion (%)')
plt.ylabel('accuracy')
plt.show()