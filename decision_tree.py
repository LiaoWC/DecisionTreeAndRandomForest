import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, List
from copy import deepcopy
from graphviz import Digraph
from exception import *


def check_transform_xy(x: ArrayLike, y: ArrayLike) -> (np.ndarray, np.ndarray):
    """ Check dimensions and turn to numpy arrays.
    :param x: training input samples
    :param y: targets values
    :return: checked x and checked y
    """
    # Turn to numpy array and transform dtype
    x, y = np.array(x).astype('float'), np.array(y).astype('float')
    # Check dimensions
    x_dim, y_dim = len(x.shape), len(y.shape)
    if x_dim != 2 or y_dim != 1:
        raise ValueError(
            "Dimensions of x and y must be 2 and 1 respectively, but got: x's dim={} and y's dim={}.".format(
                x_dim, y_dim))
    return x, y


def calculate_gini(y: np.ndarray) -> float:
    """ Calculate Gini index of the input target values.
    :param y: 1d numpy array.
    :return: Gini index of this group.
    """
    total_num = len(y)
    uniques, counts = np.unique(y, return_counts=True)
    cur_num = 1.0
    for count in counts:
        cur_num -= np.power(count / float(total_num), 2)
    return cur_num


class Node:
    """Decision tree's node.
    A child's threshold attr value <= this node's will be the left child; otherwise(>), it will be the right child.
    When calculating, we view all int values as float values even it's a class number.
    """

    def __init__(self, gini: Optional[float] = None):
        """
        :param gini: Construction of a node must comes with its Gini index value.
        """
        self.attr_idx: Optional[int] = None  # Index of picked attribute for node splitting.
        self.gini: Optional[float] = gini if gini is not None else None  # Gini index of samples this node gets.
        self.threshold: Optional[float] = None  # Node splitting threshold value
        self.n_samples: Optional[int] = None  # Number of samples this node gets in node splitting.
        self.values: Optional[np.ndarray] = None  # This node's selected attr's samples' values.
        self.targets: Optional[np.ndarray] = None  # "y" data gets in node splitting.
        self.left_child: Optional[Node] = None  # Left child node.
        self.right_child: Optional[Node] = None  # Right child node.
        self.is_leaf: bool = False  # If it's a leaf node.
        # Temporary variable for calculation
        self.cur_best_attr = 0
        self.cur_best_threshold = None
        self.cur_best_remaining_impurity = 999999  # INF
        self.cur_best_left_group = np.array([])
        self.cur_best_right_group = np.array([])
        self.cur_best_left_group_gini = None
        self.cur_best_right_group_gini = None

    def fit(self, x: np.ndarray, y: np.ndarray, is_root: bool, max_attrs: Optional[int], rand_state: Optional[int]):
        """ Fit data to this node.
        :param x: 2d float numpy arr; dim 0 is samples; dim 1 is attrs
        :param y: 1d float numpy arr; dim 0 is samples' classes
        :param max_attrs: When splitting, random choose attrs up to this number to consider.
        :param rand_state: Force it use a specific random state to pick up things.
        :param is_root: Identify if it's root. For checking params at root.
        :return: None.
        """
        # Check x, y
        if is_root:
            x, y = check_transform_xy(x, y)
            self.gini = calculate_gini(y=y)

        attrs: np.ndarray = np.arange(x.shape[1])
        # Attribute bagging
        if max_attrs is not None:  # Pick all if max_attrs is None
            n_attrs_get = min(len(attrs), max_attrs)
            if rand_state is not None:
                rng = np.random.RandomState(rand_state)
                attrs = np.array(attrs[rng.choice(attrs.shape[0], n_attrs_get, replace=False)])
            else:
                attrs = np.array(attrs[np.random.choice(attrs.shape[0], n_attrs_get, replace=False)])
        # Check each attr, picking one
        for attr in attrs:
            # Concatenate x & y
            xy = np.concatenate((x, np.expand_dims(y, axis=1)), axis=1)
            # Sort by attr
            xy = xy[xy[:, attr].argsort()]
            # Pick threshold
            for i in range(len(xy) - 1):
                threshold = (xy[i][attr] + xy[i + 1][attr]) / 2
                # Divide into 2 groups
                left_group = np.array([sample for sample in xy if sample[attr] <= threshold])
                right_group = np.array([sample for sample in xy if sample[attr] > threshold])

                # Calculate Gini index
                left_gini = calculate_gini(left_group[:, -1]) if len(left_group) > 0 else 0
                right_gini = calculate_gini(right_group[:, -1]) if len(right_group) > 0 else 0
                total_remaining_impurity = len(left_group) * left_gini + len(right_group) * right_gini
                # Check if better
                if total_remaining_impurity < self.cur_best_remaining_impurity:
                    self.cur_best_remaining_impurity = deepcopy(total_remaining_impurity)
                    self.cur_best_attr = deepcopy(attr)
                    self.cur_best_threshold = deepcopy(threshold)
                    self.cur_best_left_group = deepcopy(left_group)
                    self.cur_best_right_group = deepcopy(right_group)
                    self.cur_best_left_group_gini = deepcopy(left_gini)
                    self.cur_best_right_group_gini = deepcopy(right_gini)
        # Save samples
        self.attr_idx = self.cur_best_attr
        self.threshold = self.cur_best_threshold
        self.n_samples = len(x)
        self.values = x[:, self.cur_best_attr]
        self.targets = deepcopy(y)
        # Recursive
        if len(self.cur_best_left_group) > 0 and len(self.cur_best_right_group) > 0 and self.gini != 0.0:
            self.left_child = Node(gini=self.cur_best_left_group_gini)
            self.left_child.fit(x=self.cur_best_left_group[:, :-1], y=self.cur_best_left_group[:, -1], is_root=False,
                                max_attrs=max_attrs, rand_state=rand_state)
            self.right_child = Node(gini=self.cur_best_right_group_gini)
            self.right_child.fit(x=self.cur_best_right_group[:, :-1], y=self.cur_best_right_group[:, -1], is_root=False,
                                 max_attrs=max_attrs, rand_state=rand_state)
        else:
            # TODO: Gini calculated by parent, but
            self.is_leaf = True

    def _attr_str(self, attr_names: Optional[List[str]] = None):
        """ Make string of this node's selected attribute's name.
        :param attr_names: names of indices of x
        :return: string
        """
        return attr_names[self.attr_idx] + '\n' if attr_names else f'X[{self.attr_idx}]'

    def _target_str(self, target_names: Optional[dict] = None):
        """ Make string of this leaf node's name of target.
        :param target_names: names of targets
        :return: string
        """
        target_value = self.targets[
            0]  # Only leaf nodes call this function. A leaf node's target values are all the same.
        if target_names:
            for key in target_names:
                if key == target_value:  # "==" Allow int and float e.g. 1 == 1.0
                    return target_names[key]
            raise ValueError(
                f'The target value "{target_value}" cannot find corresponding key in the target_names dict.')
        else:
            return target_value

    def view(self, digraph: Digraph, node_name: str = 'root',
             attr_names: Optional[List[str]] = None,
             target_names: Optional[dict] = None):
        """ For viewing a decision tree's structure and outputting it as a PDF file. Recursively.
        :param digraph: digraph for constructing
        :param node_name: this node's name
        :param attr_names: names of attributes of x
        :param target_names:  names of targets of y
        :return: None
        """
        # Make node information string
        node_label = ''
        if not self.is_leaf:
            node_label += '{} <= {:.2f}\n'.format(self._attr_str(attr_names), self.threshold)
        node_label += 'gini = {:.2f}\n'.format(self.gini)
        node_label += 'n_samples = {}\n'.format(self.n_samples)
        if self.is_leaf:
            node_label += 'TARGET = {}'.format(self._target_str(target_names))
        # Add node and edge recursively
        if self.is_leaf:
            digraph.node(name=node_name, label=node_label, color='lightgreen')
        else:
            digraph.node(name=node_name, label=node_label)
        if not self.is_leaf:
            if self.left_child:
                lchild_node_name = node_name + '-l'
                self.left_child.view(digraph, node_name=lchild_node_name,
                                     attr_names=attr_names,
                                     target_names=target_names)  # -l: left
                digraph.edge(node_name, lchild_node_name,
                             label='{} <= {:.2f}'.format(self._attr_str(attr_names), self.threshold))
            if self.right_child:
                rchild_node_name = node_name + '-r'
                self.right_child.view(digraph, node_name=rchild_node_name,
                                      attr_names=attr_names,
                                      target_names=target_names)  # -r: right
                digraph.edge(node_name, rchild_node_name,
                             label='{} > {:.2f}'.format(self._attr_str(attr_names), self.threshold))


class DecisionTree:
    def __init__(self, max_attrs: Optional[int]):
        """ Construct a decision tree.
        :param max_attrs: For attribute bagging. If None, use all attributes.
        """
        self.root: Optional[Node] = None  # Keep root node.
        self.n_attrs: Optional[int] = None  # Total number of attributes.
        self.max_attrs: Optional[int] = max_attrs  # For attribute bagging. If None, use all attributes.

    def fit(self, x: ArrayLike, y: ArrayLike):
        """ Fit this decision tree.
        :param x: training input samples
        :param y: targets
        :return: None
        """
        x, y = check_transform_xy(x, y)
        self.n_attrs = x.shape[1]
        self.root = Node()
        self.root.fit(x, y, is_root=True, max_attrs=self.max_attrs, rand_state=None)

    def pred(self, x: ArrayLike):
        """ Pred a single sample or an numpy array of simples.
        :param x: 2d float numpy arr(dim 0 is samples; dim 1 is attrs) or 1d array with only one sample to pred.
        :return: 1d or 2d numpy array depending on your input x's dimension.
        """
        # Check if fitted
        if self.n_attrs is None:
            raise NotFittedError("Please fit the model before using to predict.")
        # If input is 1d make it be 2d
        x_dim1 = False
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=0)
            x_dim1 = True
        # Check input num of attrs
        if x.shape[1] != self.n_attrs:
            raise NumAttrError("Number of attrs of x is not the same as the number of attrs your fitting data uses.")
        # Pred
        pred = []
        for sample in x:
            cur_node = self.root
            while True:
                if cur_node.is_leaf:
                    pred.append(cur_node.targets[0])
                    break
                if sample[cur_node.attr_idx] <= cur_node.threshold:
                    cur_node = cur_node.left_child
                else:
                    cur_node = cur_node.right_child
        pred = np.array(pred)
        #
        if x_dim1:
            pred = pred[0]
        return pred

    def view(self, filename: str = 'DecisionTreeOutput',
             attr_names: Optional[List[str]] = None,
             target_names: Optional[dict] = None):
        """ Output a decision tree's structure & details as a PDF file.
        :param filename: Output file name.
        :param attr_names: Give each attr idx a name corresponding to the index of name in the list given.
        :param target_names: A dict of target value => name.
        :return: None
        """
        dg = Digraph('DecisionTree', filename=filename, node_attr={'color': 'lightblue2', 'style': 'filled'})
        self.root.view(digraph=dg, attr_names=attr_names, target_names=target_names)
        dg.view()
