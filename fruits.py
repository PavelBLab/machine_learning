import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

df = pd.read_table('fruit_data_with_colors.txt')
# print(df.columns)
# print(df.head())

look_up_fruit_name = dict(zip(df['fruit_label'].unique(), df['fruit_name'].unique()))
print(look_up_fruit_name)


X = df.loc[:, ['mass', 'width', 'height']]
y = df.loc[:, 'fruit_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)


# n = range(1, 15)
# for i in n:
#     knn_clf = KNeighborsClassifier(n_neighbors=i)
#     knn_clf.fit(X_train, y_train)
#     print(knn_clf.score(X_test, y_test))


scores = [(f'knn={i}', KNeighborsClassifier(n_neighbors=i)
         .fit(X_train, y_train)
         .score(X_test, y_test)) for i in range(1, 15)]
print(scores)

knn_clf = KNeighborsClassifier(n_neighbors=5)
# print(knn_clf)
knn_clf.fit(X_train, y_train)
# print(knn_clf.predict([[20, 4.3, 5.5]]))
print(look_up_fruit_name[knn_clf.predict([[20, 4.3, 5.5]])[0]])
print(look_up_fruit_name[knn_clf.predict([[100, 6.3, 8.5]])[0]])






















