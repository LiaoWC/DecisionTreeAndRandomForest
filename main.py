# My source codes
from decision_tree import DecisionTree
from random_forest import RandomForest

# Existing source codes
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import time
import matplotlib.pyplot as plt
import numpy as np

# Tools
from tools import plot_confusion_matrix

# Load data and do train-valid-split
wine = load_wine()
x, y = wine.data, wine.target
x_train, x_valid, y_train, y_valid = train_test_split(x, y, shuffle=True, random_state=0, test_size=0.5)

# Decision Tree (Basic)
tree = DecisionTree()
tree.fit(x=x_train, y=y_train)
tree.view(attr_names=wine.feature_names, target_names={i: wine.target_names[i] for i in range(len(wine.target_names))})
y_pred = tree.pred(x_valid)
cm = confusion_matrix(y_true=y_valid, y_pred=y_pred)
print(f'{accuracy_score(y_true=y_valid, y_pred=y_pred):.2%}')
plot_confusion_matrix(cm=cm, title='DecisionTreeCM', xticks=[0, 1, 2], yticks=[0, 1, 2], save_name='DecisionTreeCM')

# Random Forest (Basic)
r = RandomForest(n_trees=100, tree_bagging=True, attr_bagging=True, tree_samples_ratio=2 / 3)
r.fit(x_train, y_train)
y_pred = r.pred(x_valid)
cm = confusion_matrix(y_true=y_valid, y_pred=y_pred)
print(f'{accuracy_score(y_true=y_valid, y_pred=y_pred):.2%}')
plot_confusion_matrix(cm=cm, title='RandomForestCM', xticks=[0, 1, 2], yticks=[0, 1, 2], save_name='RandomForestCM')

# Experiments: different validation subset sizes
results = []  # list of dicts
for ratio in [i / 10 for i in range(1, 10)]:
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, shuffle=True, random_state=0, test_size=ratio)
    d = DecisionTree()
    d.fit(x_train, y_train)
    d_pred = d.pred(x_valid)
    r = RandomForest(n_trees=100, tree_bagging=True, attr_bagging=True, tree_samples_ratio=2 / 3)
    r.fit(x_train, y_train)
    r_pred = r.pred(x_valid)
    results.append({
        'ratio': ratio,
        'd_acc': accuracy_score(y_true=y_valid, y_pred=d_pred),
        'r_acc': accuracy_score(y_true=y_valid, y_pred=r_pred)
    })
plt.plot([result['ratio'] for result in results],
         [result['d_acc'] for result in results], color='b', label='Decision Tree')
plt.plot([result['ratio'] for result in results],
         [result['r_acc'] for result in results], color='g', label='Random Forest')
plt.xlabel('Validation Data Size Compared with All Data')
plt.ylabel('Accuracy')
plt.title('Accuracies of Different Validation Subset Sizes')
plt.legend()
plt.show()

# Experiment: number of trees in the forest
n_trees = (np.arange(100) + 1) * 1
results = []
x_train, x_valid, y_train, y_valid = train_test_split(x, y, shuffle=True, random_state=0, test_size=ratio)
for n_tree in n_trees:
    r = RandomForest(n_trees=n_tree, tree_bagging=True, attr_bagging=True, tree_samples_ratio=2 / 3)
    r.fit(x_train, y_train)
    r_pred = r.pred(x_valid)
    results.append(accuracy_score(y_true=y_valid, y_pred=r_pred))
plt.plot(n_trees, results)
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.yticks((np.arange(10) + 1) * 0.1)
plt.title('Different Numbers of Trees in the Random Forest')
plt.show()
