import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from matplotlib.colors import ListedColormap

# importing and preparing our dataset
dataset = pd.read_csv('D:\Code\Python\Decision tree classifier\CarData.csv')


# Preprocessing
le = preprocessing.LabelEncoder()
encodedDataset = dataset.apply(le.fit_transform)



X = encodedDataset.iloc[:,:6].values
y = encodedDataset.iloc[:,6].values 


# Splitting data into training and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.10, random_state = 0)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the training set
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train,y_train)

#  prediction the test
y_pred = classifier.predict(X_test)

# Decoding encoded values
y_pred_decoded = list(le.inverse_transform(y_pred))

y_test_decoded = list(le.inverse_transform(y_test))



# making the confusion matrix
cm = confusion_matrix(y_test, y_pred)

dataset2 = dataset.drop(columns=['class'])

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()

export_graphviz(classifier, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,
                feature_names = dataset2.columns)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())












X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green','blue','yellow','orange','purple')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classification (Test set)')
plt.legend()
plt.show()



