import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, List
from copy import deepcopy
from graphviz import Digraph


def check_transform_xy(x: ArrayLike, y: ArrayLike) -> (np.ndarray, np.ndarray):
    # Turn to numpy array and transform dtype
    x, y = np.array(x).astype('float'), np.array(y).astype('float')
    # Check dimensions
    x_dim, y_dim = len(x.shape), len(y.shape)
    if x_dim != 2 or y_dim != 1:
        raise ValueError("Shapes of x and y must be 2 and 1 respectively, but got: x's dim={} and y's dim={}.".format(
            x_dim, y_dim))
    return x, y


def calculate_gini(y: np.ndarray) -> float:
    """

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
    # <= will be the left child, > will be the right child
    # When calculating, we view all int values as float values
    def __init__(self, gini: Optional[float] = None):
        self.attr_idx: Optional[int] = None
        self.gini: Optional[float] = gini if gini is not None else None
        self.threshold: Optional[float] = None
        self.n_samples: Optional[int] = None
        self.values: Optional[np.ndarray] = None  # This node's selected attr's samples' values
        self.targets: Optional[np.ndarray] = None  # y
        self.left_child: Optional[Node] = None
        self.right_child: Optional[Node] = None
        self.is_leaf: bool = False
        #
        self.cur_best_attr = 0
        self.cur_best_threshold = None
        self.cur_best_remaining_impurity = 999999  # INF
        self.cur_best_left_group = np.array([])
        self.cur_best_right_group = np.array([])
        self.cur_best_left_group_gini = None
        self.cur_best_right_group_gini = None

    def fit(self, x: np.ndarray, y: np.ndarray, is_root: bool):
        """

        :param is_root: Identify if it's root. For checking params at root.
        :param x: 2d float numpy arr; dim 0 is samples; dim 1 is attrs
        :param y: 1d float numpy arr; dim 0 is samples' classes
        :return:
        """
        # Check x, y
        if is_root:
            x, y = check_transform_xy(x, y)
            self.gini = calculate_gini(y=y)

        # Check each attr, picking one
        attrs: List = np.arange(x.shape[1])
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
                print('ttt:', total_remaining_impurity, self.cur_best_remaining_impurity, len(left_group), left_gini,
                      len(right_group), right_gini)
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
            print('Cur best l r gini:', self.cur_best_left_group_gini, self.cur_best_right_group_gini)
            self.left_child = Node(gini=self.cur_best_left_group_gini)
            self.left_child.fit(x=self.cur_best_left_group[:, :-1], y=self.cur_best_left_group[:, -1], is_root=False)
            self.right_child = Node(gini=self.cur_best_right_group_gini)
            self.right_child.fit(x=self.cur_best_right_group[:, :-1], y=self.cur_best_right_group[:, -1], is_root=False)
        else:
            # TODO: Gini calculated by parent, but
            self.is_leaf = True

    def pred(self):
        pass

    def _attr_str(self, attr_names: Optional[List[str]] = None):
        return attr_names[self.attr_idx] + '\n' if attr_names else f'X[{self.attr_idx}]'

    def _target_str(self, target_names: Optional[dict] = None):
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
    def __init__(self):
        self.root: Optional[Node] = None

    def fit(self, x: ArrayLike, y: ArrayLike):
        x, y = check_transform_xy(x, y)
        self.root = Node()
        self.root.fit(x, y, is_root=True)

    def pred(self):
        pass

    def view(self, filename: str = 'DecisionTreeOutput',
             attr_names: Optional[List[str]] = None,
             target_names: Optional[dict] = None):
        """

        :param filename:
        :param attr_names: Give each attr idx a name corresponding to the index of name in the list given.
        :param target_names: A dict of target value => name
        :return:
        """
        dg = Digraph('DecisionTree', filename=filename, node_attr={'color': 'lightblue2', 'style': 'filled'})
        self.root.view(digraph=dg, attr_names=attr_names, target_names=target_names)
        dg.view()


from sklearn.datasets import load_iris

iris = load_iris()
d = DecisionTree()
d.fit(x=iris.data, y=iris.target)
d.view(attr_names=iris.feature_names, target_names={i: iris.target_names[i] for i in range(len(iris.target_names))})

# TODO: Plot tree

#
# from graphviz import Digraph
#
# dot = Digraph(comment='The Round Table')
#
# dot.node('A', 'King Arthur\ngini=0')
# dot.node('B', 'Sir Bedevere the Wise')
# dot.node('L', 'Sir Lancelot the Brave')
#
# dot.edges(['AB', 'AL'])
# # dot.edge('B', 'L', constraint='false')
# dot.edge('B', 'L')
#
# print(dot.source)
#
# dot.render('test-output/round-table.gv', view=True)
#

#
# u = Digraph('unix', filename='unix.gv',
#             node_attr={'color': 'lightblue2', 'style': 'filled'})
# u.edge('5th Edition', '6th Edition')
# u.edge('5th Edition', 'PWB 1.0')
# u.edge('6th Edition', 'LSX')
# u.edge('6th Edition', '1 BSD')
# u.edge('6th Edition', 'Mini Unix')
# u.edge('6th Edition', 'Wollongong')
# u.edge('6th Edition', 'Interdata')
# u.edge('Interdata', 'Unix/TS 3.0')
# u.edge('Interdata', 'PWB 2.0')
# u.edge('Interdata', '7th Edition')
# u.edge('7th Edition', '8th Edition')
# u.edge('7th Edition', '32V')
# u.edge('7th Edition', 'V7M')
# u.edge('7th Edition', 'Ultrix-11')
# u.edge('7th Edition', 'Xenix')
# u.edge('7th Edition', 'UniPlus+')
# u.edge('V7M', 'Ultrix-11')
# u.edge('8th Edition', '9th Edition')
# u.edge('1 BSD', '2 BSD')
# u.edge('2 BSD', '2.8 BSD')
# u.edge('2.8 BSD', 'Ultrix-11')
# u.edge('2.8 BSD', '2.9 BSD')
# u.edge('32V', '3 BSD')
# u.edge('3 BSD', '4 BSD')
# u.edge('4 BSD', '4.1 BSD')
# u.edge('4.1 BSD', '4.2 BSD')
# u.edge('4.1 BSD', '2.8 BSD')
# u.edge('4.1 BSD', '8th Edition')
# u.edge('4.2 BSD', '4.3 BSD')
# u.edge('4.2 BSD', 'Ultrix-32')
# u.edge('PWB 1.0', 'PWB 1.2')
# u.edge('PWB 1.0', 'USG 1.0')
# u.edge('PWB 1.2', 'PWB 2.0')
# u.edge('USG 1.0', 'CB Unix 1')
# u.edge('USG 1.0', 'USG 2.0')
# u.edge('CB Unix 1', 'CB Unix 2')
# u.edge('CB Unix 2', 'CB Unix 3')
# u.edge('CB Unix 3', 'Unix/TS++')
# u.edge('CB Unix 3', 'PDP-11 Sys V')
# u.edge('USG 2.0', 'USG 3.0')
# u.edge('USG 3.0', 'Unix/TS 3.0')
# u.edge('PWB 2.0', 'Unix/TS 3.0')
# u.edge('Unix/TS 1.0', 'Unix/TS 3.0')
# u.edge('Unix/TS 3.0', 'TS 4.0')
# u.edge('Unix/TS++', 'TS 4.0')
# u.edge('CB Unix 3', 'TS 4.0')
# u.edge('TS 4.0', 'System V.0')
# u.edge('System V.0', 'System V.2')
# u.edge('System V.2', 'System V.3')
# u.render('test-output/round-table.gv', view=True)
# # u.view()


# from graphviz import Source
# temp = """
# digraph G{
# edge [dir=forward]
# node [shape=plaintext]
#
# No.0 [label="0 (None)\nHola"]
# 0 -> 5 [label="root"]
# 1 [label="1 (Hello)"]
# 2 [label="2 (how)"]
# 2 -> 1 [label=">="]
# 3 [label="3 (are)"]
# 4 [label="4 (you)"]
# 5 [label="5 (doing)"]
# 5 -> 3 [label="aux"]
# 5 -> 2 [label="advmod"]
# 5 -> 4 [label="nsubj"]
# }
# """
# s = Source(temp, filename="test.gv", format="png")
# s.view()
#
#
# import matplotlib as mpl
# from matplotlib import cm
# import colorsys
# import functools
# import graphviz as gv
#
# def add_nodes(graph, nodes):
#     for n in nodes:
#         if isinstance(n, tuple):
#             graph.node(n[0], **n[1])
#         else:
#             graph.node(n)
#     return graph
#
# A = [[517, 1, [409], 10, 6],
#      [534, 1, [584], 10, 12],
#      [614, 1, [247], 11, 5],
#      [679, 1, [228], 13, 7],
#      [778, 1, [13], 14, 14]]
#
# nodesgv = []
# Arange = [ a[0] for a in A]
# norm = mpl.colors.Normalize(vmin = min(Arange), vmax = max(Arange))
# cmap = cm.jet
#
# for index, i in enumerate(A):
#     x = i[0]
#     m = cm.ScalarMappable(norm = norm, cmap = cmap)
#     mm = m.to_rgba(x)
#     M = colorsys.rgb_to_hsv(mm[0], mm[1], mm[2])
#     nodesgv.append((str(i[0]),{'label': str((i[1])), 'color': "%f, %f, %f" % (M[0], M[1], M[2]), 'style': 'filled'}))
#
# graph = functools.partial(gv.Graph, format='svg', engine='neato')
# add_nodes(graph(), nodesgv).render(('img/test'))
