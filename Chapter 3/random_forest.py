from sklearn.ensemble import RandomForestClassifier
forest  = RandomForestClassifier(criterion='entropy',
                                 n_estimators=10,
                                 random_state=0)

from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

forest.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

from pdr import plot_decision_regions
import matplotlib.pyplot as plt
plot_decision_regions(X=X_combined,
                      y=y_combined,
                      classifier=forest,
                      test_idx=range(105,150))
plt.title('Random Forest')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')
plt.show()