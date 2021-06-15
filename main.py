# My source codes
from decision_tree import DecisionTree
from random_forest import RandomForest

# Existing source codes
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm as mpl_cm

# Tools modules
from tools import plot_cm

# Default constants
DF_TREE_NUM = 20
DF_TREE_SAMPLE_RATIO = 2 / 3
DF_RF_MAX_ATTR = 4
DF_DT_MAX_ATTR = None
DF_TEST_SIZE = 0.5
DF_RAND_STATE = 0
DF_RAND_STATES = [0, 1, 2, 3, 4]
DF_N_FOLD = 3
DF_KF = KFold(n_splits=DF_N_FOLD, shuffle=True, random_state=DF_RAND_STATE)
DF_KFS = [KFold(n_splits=DF_N_FOLD, shuffle=True, random_state=rand_state) for rand_state in DF_RAND_STATES]

# Load data and do train-valid-split
wine = load_wine()
x, y = wine.data, wine.target
###################################################################################################################

# Decision Tree (Basic)
cms = []
accs = []
for rand_state, kf in zip(DF_RAND_STATES, DF_KFS):
    for i, (tr, te) in enumerate(kf.split(x)):
        x_train, x_valid, y_train, y_valid = x[tr], x[te], y[tr], y[te]
        tree = DecisionTree(max_attrs=DF_DT_MAX_ATTR)
        tree.fit(x_train, y_train)
        tree.view(attr_names=wine.feature_names,
                  target_names={i: wine.target_names[i] for i in range(len(wine.target_names))},
                  filename=f'DecisionTreeOutput/DecisionTreeOutput{i + 1}_RandState{rand_state}')
        y_pred = tree.pred(x_valid)
        cms.append(confusion_matrix(y_valid, y_pred))
        accs.append(accuracy_score(y_valid, y_pred))
avg_cm = np.sum(np.array(cms), axis=0) / len(cms)
avg_acc = sum(accs) / len(accs)
print(f'{avg_acc:.2%}')
plot_cm(cm=avg_cm, title='DecisionTreeAvgCM', xticks=[0, 1, 2], yticks=[0, 1, 2], save_name='DecisionTreeAvgCM')

# Random Forest (Basic)
cms = []
accs = []
for rand_state, kf in zip(DF_RAND_STATES, DF_KFS):
    for i, (tr, te) in enumerate(kf.split(x)):
        x_train, x_valid, y_train, y_valid = x[tr], x[te], y[tr], y[te]
        r = RandomForest(n_trees=DF_TREE_NUM, tree_bagging=True, tree_samples_ratio=DF_TREE_SAMPLE_RATIO,
                         max_attrs=DF_RF_MAX_ATTR)
        r.fit(x_train, y_train)
        y_pred = r.pred(x_valid)
        cms.append(confusion_matrix(y_valid, y_pred))
        accs.append(accuracy_score(y_valid, y_pred))
avg_cm = np.sum(np.array(cms), axis=0) / len(cms)
avg_acc = sum(accs) / len(accs)
print(f'{avg_acc:.2%}')
plot_cm(cm=avg_cm, title='RandomForestAvgCM', xticks=[0, 1, 2], yticks=[0, 1, 2], save_name='RandomForestAvgCM')

# E.1 #################################################################################

# Experiment: number of trees in the forest. P.S. This takes lots of time!
n_trees = (np.arange(50) + 1) * 1
results = []
for n_tree in n_trees:
    print(n_tree)
    accs = []
    for rand_state, kf in zip(DF_RAND_STATES, DF_KFS):
        for i, (tr, te) in enumerate(kf.split(x)):
            x_train, x_valid, y_train, y_valid = x[tr], x[te], y[tr], y[te]
            r = RandomForest(n_trees=n_tree, tree_bagging=True, tree_samples_ratio=DF_TREE_SAMPLE_RATIO,
                             max_attrs=DF_RF_MAX_ATTR)
            r.fit(x_train, y_train)
            r_pred = r.pred(x_valid)
            accs.append(accuracy_score(y_valid, r_pred))
    results.append(sum(accs) / len(accs))
plt.figure(figsize=(6.4, 3.6))
plt.plot(n_trees, results, color='orange', label='Random Forest')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.yticks((np.arange(9)) * 0.025 + 0.775)
plt.title('Different Numbers of Trees in the Random Forest')
plt.legend()
plt.savefig('NumTreesInRF')
plt.show()

# E.2 #################################################################################

# Experiments: different validation subset sizes with different num of trees
results = []  # list of dicts
for ratio in [i / 10 for i in range(1, 10)]:
    print(ratio)
    d_accs = []
    r10_accs = []
    r20_accs = []
    r50_accs = []
    for i, (tr, te) in enumerate(DF_KF.split(x)):
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, shuffle=True, random_state=DF_RAND_STATE,
                                                              test_size=ratio)
        d = DecisionTree(max_attrs=DF_DT_MAX_ATTR)
        d.fit(x_train, y_train)
        d_pred = d.pred(x_valid)
        r10 = RandomForest(n_trees=DF_TREE_NUM, tree_bagging=True, tree_samples_ratio=DF_TREE_SAMPLE_RATIO,
                           max_attrs=DF_RF_MAX_ATTR)
        r10.fit(x_train, y_train)
        r10_pred = r10.pred(x_valid)
        r20 = RandomForest(n_trees=DF_TREE_NUM, tree_bagging=True, tree_samples_ratio=DF_TREE_SAMPLE_RATIO,
                           max_attrs=DF_RF_MAX_ATTR)
        r20.fit(x_train, y_train)
        r20_pred = r20.pred(x_valid)
        r50 = RandomForest(n_trees=DF_TREE_NUM, tree_bagging=True, tree_samples_ratio=DF_TREE_SAMPLE_RATIO,
                           max_attrs=DF_RF_MAX_ATTR)
        r50.fit(x_train, y_train)
        r50_pred = r50.pred(x_valid)
        d_accs.append(accuracy_score(y_true=y_valid, y_pred=d_pred))
        r10_accs.append(accuracy_score(y_true=y_valid, y_pred=r10_pred))
        r20_accs.append(accuracy_score(y_true=y_valid, y_pred=r20_pred))
        r50_accs.append(accuracy_score(y_true=y_valid, y_pred=r50_pred))
    results.append({
        'ratio': ratio,
        'd_avg_acc': sum(d_accs) / len(d_accs),
        'r10_avg_acc': sum(r10_accs) / len(r10_accs),
        'r20_avg_acc': sum(r20_accs) / len(r20_accs),
        'r50_avg_acc': sum(r50_accs) / len(r50_accs)
    })
plt.plot([result['ratio'] for result in results],
         [result['d_avg_acc'] for result in results], color='b', label='Decision tree', marker='.')
plt.plot([result['ratio'] for result in results],
         [result['r10_avg_acc'] for result in results], color='mediumseagreen', label='Random forest (10 trees)',
         marker='.',
         linestyle='--')
plt.plot([result['ratio'] for result in results],
         [result['r20_avg_acc'] for result in results], color='olivedrab', label='Random forest (20 trees)',
         marker='.', linestyle='--')
plt.plot([result['ratio'] for result in results],
         [result['r50_avg_acc'] for result in results], color='teal', label='Random forest (50 trees)', marker='.',
         linestyle='--')
plt.xlabel('Validation Data Size Compared with All Data')
plt.ylabel('Accuracy')
plt.title('Accuracies of Different Validation Subset Sizes')
plt.legend()
plt.show()

# E.3 #################################################################################

# # Experiment: max number of attributes => decision tree & 20 50 tree rand forests
# max_attrs = np.arange(13) + 1  # 1~13 (Wine dataset has 13 attrs. Case of only one => extremely random forest
# results = []  # list of dicts
# for max_attr in max_attrs:
#     print(max_attr)
#     d_accs = []
#     r_accs = []
#     d50_accs = []
#     r50_accs = []
#     for i, (tr, te) in enumerate(DF_KF.split(x)):
#         x_train, x_valid, y_train, y_valid = x[tr], x[te], y[tr], y[te]
#         d = DecisionTree(max_attrs=max_attr)
#         d.fit(x_train, y_train)
#         d_pred = d.pred(x_valid)
#         # It seems I turn off attr bagging but actually max_attrs make an effect like attr bagging in my codes.
#         r = RandomForest(n_trees=20, tree_bagging=True, tree_samples_ratio=DF_TREE_SAMPLE_RATIO, max_attrs=max_attr)
#         r.fit(x_train, y_train)
#         r_pred = r.pred(x_valid)
#         r50 = RandomForest(n_trees=50, tree_bagging=True, tree_samples_ratio=DF_TREE_SAMPLE_RATIO, max_attrs=max_attr)
#         r50.fit(x_train, y_train)
#         r50_pred = r50.pred(x_valid)
#         # Record
#         d_accs.append(accuracy_score(y_true=y_valid, y_pred=d_pred))
#         r_accs.append(accuracy_score(y_true=y_valid, y_pred=r_pred))
#         r50_accs.append(accuracy_score(y_true=y_valid, y_pred=r50_pred))
#     results.append({
#         'max_attrs': max_attr,
#         'd_acc': sum(d_accs) / len(d_accs),
#         'r_acc': sum(r_accs) / len(r_accs),
#         'r50_acc': sum(r50_accs) / len(r50_accs)
#     })
# plt.plot([result['max_attrs'] for result in results],
#          [result['d_acc'] for result in results], color='b', linestyle='-',
#          label=f'Decision Tree', marker='.')
# plt.plot([result['max_attrs'] for result in results],
#          [result['r_acc'] for result in results], color='g', linestyle='-',
#          label=f'Random Forest ({20} trees)', marker='.')
# plt.plot([result['max_attrs'] for result in results],
#          [result['r50_acc'] for result in results], color='y', linestyle='-',
#          label=f'Random Forest ({50} trees)', marker='.')
# plt.xlabel('Different Max Number of Attributes When Splitting')
# plt.ylabel('Accuracy')
# plt.title('Max Attributes')
# plt.legend(loc='lower right')
# plt.show()
#
# Experiment: Try more rand state -- max number of attributes => decision tree & 20 50 tree rand forests
max_attrs = np.arange(13) + 1  # 1~13 (Wine dataset has 13 attrs. Case of only one => extremely random forest
results = []  # list of dicts
n_tree = 20
for max_attr in max_attrs:
    print('max_attr', max_attr)
    same_max_attr_results = []
    for rand_state, kf in zip(DF_RAND_STATES, DF_KFS):
        print(f'--- random state: {rand_state} ---')
        r_accs = []
        for i, (tr, te) in enumerate(kf.split(x)):
            x_train, x_valid, y_train, y_valid = x[tr], x[te], y[tr], y[te]
            # It seems I turn off attr bagging but actually max_attrs make an effect like attr bagging in my codes.
            r = RandomForest(n_trees=n_tree, tree_bagging=True, tree_samples_ratio=DF_TREE_SAMPLE_RATIO,
                             max_attrs=max_attr)
            r.fit(x_train, y_train)
            r_pred = r.pred(x_valid)
            r_accs.append(accuracy_score(y_true=y_valid, y_pred=r_pred))
        same_max_attr_results.append({
            'max_attrs': max_attr,
            'r_acc': sum(r_accs) / len(r_accs),
        })
    results.append(same_max_attr_results)
results_s = []
for i in range(len(results[0])):
    result_ = []
    for item in results:
        result_.append(item[i])
    results_s.append(result_)
res_s = []
plt.figure(figsize=(8, 5.6))
for i, results_ in enumerate(results_s):
    res = [result['r_acc'] for result in results_]
    plt.plot([result['max_attrs'] for result in results_], res, linestyle='-',
             label=f'Random Forest {i + 1} ({n_tree} trees)', marker='.',
             color=mpl_cm.get_cmap('Greens')(((i + 1) / 5) * 0.9))
    res_s.append(res)
res_s = np.array(res_s)
plt.plot([result['max_attrs'] for result in results_s[0]], np.sum(res_s, axis=0) / res_s.shape[0], linestyle='-',
         label=f'Average', marker='.', color='r')
plt.plot([result['max_attrs'] for result in results_s[0]], np.median(res_s, axis=0), linestyle='-', color='b',
         label=f'Median', marker='.')
plt.xlabel('Different Max Number of Attributes When Splitting')
plt.ylabel('Accuracy')
plt.title('Max Attributes')
plt.legend(loc='lower left')
plt.show()

# E.4 #################################################################################

# Experiment: Number of Samples Each Tree Gets in Tree Bagging
tree_sample_ratios = (np.arange(3) + 1) * 3.2 / 10
all_results = []
# n_trees = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]  # [25, 50, 75, 100]
n_trees = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # [25, 50, 75, 100]
colors = ['springgreen', 'mediumseagreen', 'teal', 'olivedrab']
for n_tree in n_trees:
    print('n_tree:', n_tree)
    results = []
    for tree_sample_ratio in tree_sample_ratios:
        print(tree_sample_ratio)
        accs = []
        for rand_state, kf in zip(DF_RAND_STATES, DF_KFS):
            for i, (tr, te) in enumerate(kf.split(x)):
                x_train, x_valid, y_train, y_valid = x[tr], x[te], y[tr], y[te]
                r = RandomForest(n_trees=n_tree, tree_bagging=True, tree_samples_ratio=tree_sample_ratio,
                                 max_attrs=DF_RF_MAX_ATTR)
                r.fit(x_train, y_train)
                r_pred = r.pred(x_valid)
                accs.append(accuracy_score(y_true=y_valid, y_pred=r_pred))
        results.append(sum(accs) / len(accs))
    all_results.append(results)
plt.figure(figsize=(5.6, 4.8))
for results, n_tree in zip(all_results, n_trees):
    plt.plot(tree_sample_ratios, results, linestyle='--', label=f'{n_tree} trees', marker='.',
             color=mpl_cm.get_cmap('Greens')(n_tree / 100))
plt.plot(tree_sample_ratios, np.sum(np.array(all_results), axis=0) / len(n_trees), color='r', linestyle='-',
         label='Average')
plt.plot(tree_sample_ratios, np.median(np.array(all_results), axis=0), color='b', linestyle='-',
         label='Median')
plt.xlabel('Ratio of Samples Gotten by Each Tree')
plt.xticks(tree_sample_ratios)
plt.ylabel('Accuracy')
plt.title('Numbers of Samples Each Tree Gets in Tree Bagging')
plt.legend(loc='lower right')
plt.show()
plt.figure(figsize=(5.6, 3.2))
for results, tree_sample_ratio in zip(np.transpose(np.array(all_results))[::-1], tree_sample_ratios.tolist()[::-1]):
    plt.plot(n_trees, results, linestyle='--', label=f'Ratio {tree_sample_ratio:.2f}')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Numbers of Samples Each Tree Gets in Tree Bagging')
plt.legend()
plt.show()

# E.5 #################################################################################

# Experiment: ablation comparing: normal vs without-tree-bagging vs without attr-bagging
n_trees = (np.arange(10) + 1) * 10  # (np.arange(5) + 1) * 20
all_results = []
names = ['Use both', 'Without tree bagging', 'Without attr bagging']
for tree_bagging, attr_bagging in [[True, True], [False, True], [True, False]]:
    results = []
    print('==================')
    for n_tree in n_trees:
        print(n_tree)
        rand_state_results = []
        for rand_state, kf in zip(DF_RAND_STATES, DF_KFS):
            for i, (tr, te) in enumerate(kf.split(x)):
                x_train, x_valid, y_train, y_valid = x[tr], x[te], y[tr], y[te]
                if attr_bagging:
                    r = RandomForest(n_trees=n_tree, tree_bagging=tree_bagging, tree_samples_ratio=2 / 3,
                                     max_attrs=DF_RF_MAX_ATTR)
                else:
                    r = RandomForest(n_trees=n_tree, tree_bagging=tree_bagging, tree_samples_ratio=2 / 3,
                                     max_attrs=None)
                r.fit(x_train, y_train)
                r_pred = r.pred(x_valid)
                rand_state_results.append(accuracy_score(y_true=y_valid, y_pred=r_pred))
        results.append(sum(rand_state_results) / len(rand_state_results))
    all_results.append(results)
for i, results in enumerate(all_results):
    plt.plot(n_trees, results, label=f'{names[i]}', marker='.')
plt.xlabel('Number of Trees')
plt.xticks((np.arange(10) + 1) * 10)
plt.ylabel('Accuracy')
# plt.yticks((np.arange(15) + 1) * 0.01 + 0.85)
plt.title('Numbers of Trees in RF W/O Tree Bagging & Attr Bagging')
plt.legend()
plt.show()

# E.6 #################################################################################

# Experiment: out-of-bag error => normal vs without-tree-bagging vs without attr-bagging
n_trees = (np.arange(10) + 1) * 10  # (np.arange(5) + 1) * 20
#
# x_train, x_valid, y_train, y_valid = train_test_split(x, y, shuffle=True, random_state=DF_RAND_STATE,
#                                                       test_size=0.2)
# d = RandomForest(n_trees=10, tree_bagging=True, tree_samples_ratio=2 / 3, max_attrs=DF_RF_MAX_ATTR)
# oob_error = d.fit(x_train, y_train, return_oob_error=True)
# print(accuracy_score(y_valid, d.pred(x_valid)))
# print('ooberror:', oob_error)
#
all_results = []
names = ['With attr bagging', 'Without attr bagging']
for tree_bagging, attr_bagging in [[True, True], [True, False]]:
    results = []
    print('==================')
    for n_tree in n_trees:
        print(n_tree)
        rand_state_results = []
        for rand_state, kf in zip(DF_RAND_STATES, DF_KFS):
            for i, (tr, te) in enumerate(kf.split(x)):
                x_train, x_valid, y_train, y_valid = x[tr], x[te], y[tr], y[te]
                if attr_bagging:
                    r = RandomForest(n_trees=n_tree, tree_bagging=tree_bagging, tree_samples_ratio=2 / 3,
                                     max_attrs=DF_RF_MAX_ATTR)
                else:
                    r = RandomForest(n_trees=n_tree, tree_bagging=tree_bagging, tree_samples_ratio=2 / 3,
                                     max_attrs=None)
                oob_error = r.fit(x_train, y_train, return_oob_error=True)
                rand_state_results.append(oob_error)
        results.append(sum(rand_state_results) / len(rand_state_results))
    all_results.append(results)
for i, results in enumerate(all_results):
    plt.plot(n_trees, results, label=f'{names[i]}')
plt.xlabel('Number of Trees')
plt.xticks((np.arange(10) + 1) * 10)
plt.ylabel('Out-of-bag Error')
# plt.yticks((np.arange(15) + 1) * 0.01 + 0.85)
plt.title('Numbers of Trees in RF W/O Tree Bagging & Attr Bagging')
plt.legend()
plt.show()
