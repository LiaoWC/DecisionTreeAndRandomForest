from decision_tree import DecisionTree
from typing import Optional, List
import numpy as np
import math
from copy import deepcopy


class RandomForest:
    def __init__(self, n_trees: int, tree_bagging: bool, tree_samples_ratio: float, max_attrs: Optional[int]):
        """
        P.S. Attr bagging use `floor(n_attrs^(1/2))` attrs
        P.S. Each tree use `max(1, floor(n_samples * tree_sample_ration))`
        :param n_trees:
        :param tree_bagging:
        :param tree_samples_ratio:
        """
        self.n_trees: int = n_trees
        self.tree_samples_ratio: float = tree_samples_ratio
        self.trees: List[DecisionTree] = [DecisionTree(max_attrs=max_attrs) for _ in range(self.n_trees)]
        self.tree_bagging_use_indices: List[np.ndarray] = []
        self.tree_bagging: bool = tree_bagging
        self.max_attrs: Optional[float] = max_attrs

    @staticmethod
    def _tree_bagging(x, y, tree_samples_ratio):
        n_samples = x.shape[0]
        if tree_samples_ratio > 1 or tree_samples_ratio < 0:
            raise ValueError(f'"tree_samples_ratio" must be < 1 and > 0 but got {tree_samples_ratio}')
        indices = np.random.choice(n_samples, max(1, math.floor(n_samples * tree_samples_ratio)), replace=True)
        return x[indices], y[indices], indices

    def fit(self, x, y, return_oob_error: bool = False):
        """
:
        :param x: 2d float numpy arr; dim 0 is samples; dim 1 is attrs
        :param y: 1d float numpy arr; dim 0 is samples' classes
        :param return_oob_error
        :return:
        """
        for tree in self.trees:
            _x, _y = deepcopy(x), deepcopy(y)
            if self.tree_bagging:
                _x, _y, sample_indices = RandomForest._tree_bagging(_x, _y, self.tree_samples_ratio)
                self.tree_bagging_use_indices.append(sample_indices)
            tree.fit(_x, _y)

        if return_oob_error:
            n_incorrect = 0
            for idx in range(x.shape[0]):
                # Make those trees didn't use this sample to vote
                preds = []
                for i, tree in enumerate(self.trees):
                    if idx not in self.tree_bagging_use_indices[i]:
                        pred = tree.pred(x[idx])
                        preds.append(pred)
                preds = np.array(preds)
                if len(preds) > 0:
                    voted_pred = np.bincount(preds.astype('int')).argmax()
                    n_incorrect += 1 if voted_pred != y[idx] else 0
            return n_incorrect / x.shape[0]

    def pred(self, x, regression: bool = False):
        """
        If regression = True, average the voting result, else pick most frequent class

        :param x:
        :param regression:
        :return:
        """
        preds = []
        for i, tree in enumerate(self.trees):
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
