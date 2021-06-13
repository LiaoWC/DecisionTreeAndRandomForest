from decision_tree import DecisionTree
from typing import Optional, List
import numpy as np
import math
from copy import deepcopy


def _tree_bagging(x, y, tree_samples_ratio):
    n_samples = x.shape[0]
    if tree_samples_ratio > 1 or tree_samples_ratio < 0:
        raise ValueError(f'"tree_samples_ratio" must be < 1 and > 0 but got {tree_samples_ratio}')
    indices = np.random.choice(n_samples, max(1, math.floor(n_samples * tree_samples_ratio)), replace=False)
    return x[indices], y[indices]


def _attr_bagging(x):
    n_attrs = x.shape[1]
    n_tree_attrs = math.floor(math.sqrt(n_attrs))
    indices = np.random.choice(n_attrs, n_tree_attrs, replace=False)
    return x[:, indices], indices


class RandomForest:
    def __init__(self, n_trees: int, tree_bagging: bool, attr_bagging: bool, tree_samples_ratio: float):
        """
        P.S. Attr bagging use `floor(n_attrs^(1/2))` attrs
        P.S. Each tree use `max(1, floor(n_samples * tree_sample_ration))`
        :param n_trees:
        :param tree_bagging:
        :param attr_bagging:
        :param tree_samples_ratio:
        """
        self.n_trees: int = n_trees
        self.tree_samples_ratio: float = tree_samples_ratio
        self.trees: List[DecisionTree] = [DecisionTree() for _ in range(self.n_trees)]
        self.tree_attr_indices: List[np.ndarray] = []
        self.tree_bagging: bool = tree_bagging
        self.attr_bagging: bool = attr_bagging

    def fit(self, x, y):
        """

        :param x: 2d float numpy arr; dim 0 is samples; dim 1 is attrs
        :param y: 1d float numpy arr; dim 0 is samples' classes
        :return:
        """
        for tree in self.trees:
            _x, _y = deepcopy(x), deepcopy(y)
            if self.tree_bagging:
                _x, _y = _tree_bagging(_x, _y, self.tree_samples_ratio)
            if self.attr_bagging:
                _x, attr_indices = _attr_bagging(_x)
                self.tree_attr_indices.append(attr_indices)
            tree.fit(_x, _y)

    def pred(self, x, regression: bool = False):
        """
        If regression = True, average the voting result, else pick most frequent class

        :param x:
        :param regression:
        :return:
        """
        preds = []
        for i, tree in enumerate(self.trees):
            if self.attr_bagging:
                pred = tree.pred(x[:, self.tree_attr_indices[i]])
            else:
                pred = tree.pred(x)
            preds.append(pred)
        preds = np.array(preds)  # Dim 0 is tree's pred; dim 1 is each sample in x's pred by trees
        preds = np.transpose(preds)  # Now, dim 1 is tree's pred; dim 0 is each sample in x's pred by trees
        voted_preds = []
        for pred in preds:
            if regression:
                voted_preds.append(sum(pred) / len(pred))
            else:
                voted_preds.append(np.bincount(pred.astype('int')).argmax())
        return voted_preds

