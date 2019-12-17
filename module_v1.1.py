from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
print(os.listdir())

fruits = pd.read_table('fruit_data_with_colors.txt')
print(fruits.head(2))
print(fruits.shape)

lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))
print(lookup_fruit_name)


''' Make Test Train Split Test '''

X = fruits[['mass', 'width', 'height', 'color_score']]
y = fruits['fruit_label']
# print(type(X))
# print(type(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# print(X_train.head(), y_train.head())
# print(X_train.shape)
print(X_train.shape[0]/fruits.shape[0])


# plotting a 3D scatter plot
from mpl_toolkits.mplot3d import Axes3D
# print(X_train['width'])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train['width'], X_train['height'], X_train['color_score'], c=y_train, marker='o', s=100)
ax.set_xlabel('width')
ax.set_ylabel('height')
ax.set_zlabel('color_score')
# plt.show()

'''Create classifier object'''
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)

''' Train classifier'''
knn.fit(X_train, y_train)

''' Estimate the accuracy of classifier on future data '''
print('accuracy ==>', knn.score(X_test, y_test))


fruit_prediction = knn.predict([[20, 4.3, 5.5, 0.55]])
print(fruit_prediction)
print(lookup_fruit_name[fruit_prediction[0]])


k_range = range(1, 20)
score = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    score.append(knn.score(X_test, y_test))

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, score)
plt.xticks(range(1, 21, 5))
plt.show()



















