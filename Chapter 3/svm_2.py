from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.svm import SVC

svm = SVC(kernel='rbf', C=1.0, random_state=0, gamma=0.2)
svm.fit(X_train_std, y_train)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

import matplotlib.pyplot as plt
from pdr import plot_decision_regions
plot_decision_regions(X=X_combined_std,
                      y=y_combined,
                      classifier=svm,
                      test_idx=range(105,150))

plt.title('SVM - RBF Kernel')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()

svm = SVC(kernel='rbf', C=1.0, random_state=0, gamma=100)
svm.fit(X_train_std, y_train)

plot_decision_regions(X=X_combined_std,
                      y=y_combined,
                      classifier=svm,
                      test_idx=range(105, 150))

plt.title('SVM - RBF Kernel')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()
