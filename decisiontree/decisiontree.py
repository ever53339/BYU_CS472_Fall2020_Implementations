import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import copy

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score

class DTClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self,counts=None, max_depth=np.inf, purity_threshold=1.0):
        """ Initialize class with chosen hyperparameters.
        Args:
        Optional Args (Args we think will make your life easier):
            counts: A list of Ints that tell you how many types of each feature there are
        Example:
            DT  = DTClassifier()
            or
            DT = DTClassifier(count = [2,3,2,2])
            Dataset = 
            [[0,1,0,0],
            [1,2,1,1],
            [0,1,1,0],
            [1,2,0,1],
            [0,0,1,1]]

        """
        self.counts = counts
        self.max_depth = max_depth
        self.purity_threshold = purity_threshold

    def fit(self, X, y):
        """ Fit the data; Make the Desicion tree

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets

        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        # self.attribute_prob = []
        # for i in range(X.shape[1]):
        #     count = self._count_diff_values(X[:, i].reshape(-1, 1))
        #     total = 0
        #     for value in count:
        #         total += count[value]
        #     for value in count:
        #         count[value] /= total
        #     self.attribute_prob.append(count)
        X = copy.deepcopy(X)
        self._pre_process_data(X)
        self.tree = self._complete_split(X, y, 0)

        return self

    def predict(self, X):
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        X = copy.deepcopy(X)
        self._pre_process_data(X)
        num_instance = X.shape[0]
        '''suppose for now each label is represented as an integer '''

        y = np.zeros([num_instance, 1])
        for i in range(num_instance):
            y[i][0] = self._one_predict(self.tree, X[i])

        return np.round(y)

    def _one_predict(self, tree, x):
        if not isinstance(tree, dict):
            return tree

        attribute = list(tree.keys())[0]
        value = x[attribute]
        if value not in tree[attribute]:
            # print('encounter unseen attribute value')
            potential_values = list(tree[attribute].keys())
            # prob = [tree[attribute][value][1] for value in potential_values]
            result = 0
            for v in potential_values:
                result += tree[attribute][v][1] * self._one_predict(tree[attribute][v][0], x)

            # value = np.random.choice(potential_values, p=prob)
        else:
            result = self._one_predict(tree[attribute][value][0], x)
        # return self._one_predict(tree[attribute][value][0], x, weight)
        return result

    def score(self, X, y):
        """ Return accuracy(Classification Acc) of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array of the targets 
        """
        output = self.predict(X)
        if y.shape[1] == 1:
            accuracy = (output == y).mean()
        else:
            accuracy = (output.argmax(axis=1) == y.argmax(axis=1)).mean()

        return accuracy

    def _complete_split(self, X, y, depth):
        # each split route is a list of tuples
        # [(attr_0, value_0), (attr_1, value_1)...(attr_n, value_n)]
        # a complete split returns all split routes
        # split_routes = []
        major_class, purity = self._majority(y)
        if depth >= self.max_depth or depth >= X.shape[1] or purity >= self.purity_threshold:
            return major_class

        '''
        subsets is a dict {diff_value_0: (X_0, y_0),...,diff_value_n: (X_n, y_n)}
        attribute is the index of chosen attribute (column index of 2-d array X)
        '''
        subsets, attribute = self._one_split(X, y)
        values = list(subsets.keys())
        tree = {attribute: {values[i]: [None, subsets[values[i]][0].shape[0]/X.shape[0]] for i in range(len(values))}}

        for i in range(len(values)):
            value = values[i]
            new_X, new_y = subsets[value]
            # for route in self._complete_split(new_X, new_y, depth+1):
            #     split_routes.append(route.insert(0, local_split[i]))
            tree[attribute][value][0] = self._complete_split(new_X, new_y, depth+1)

        return tree

    def _majority(self, y):
        label_count = self._count_diff_values(y)
        max_count = 0
        total_count = 0
        for label in label_count:
            total_count += label_count[label]
            if label_count[label] > max_count:
                max_count = label_count[label]
                major_class = label

        return major_class, max_count/total_count

    def _one_split(self, X, y):
        num_instance = X.shape[0]
        num_attribute = X.shape[1]
        info_gain = np.zeros(num_attribute)
        intrinsic_info = np.zeros(num_attribute)
        prev_info = self._compute_info(y)
        for i in range(num_attribute):
            after_info = 0
            split = self._split_with_attribute(X, y, i)
            for value in split:
                after_info += split[value][1].shape[0] / num_instance * self._compute_info(split[value][1])
            info_gain[i] = prev_info - after_info
            intrinsic_info[i] = max(self._compute_intrinsic_info(split), 0.0001)

        info_gain = info_gain * (info_gain > info_gain.mean())
        info_gain = info_gain/intrinsic_info
            # intrinsic_info
        attribute = info_gain.argmax()

        return self._split_with_attribute(X, y, attribute), attribute

    def _split_with_attribute(self, X, y, attribute):
        num_instance = X.shape[0]
        values = list(set(X[:, attribute]))
        subsets = [[[], []] for i in range(len(values))]
        for i in range(num_instance):
            num_subset = values.index(X[i][attribute])
            subsets[num_subset][0].append(X[i])
            subsets[num_subset][1].append(y[i])

        for i in range(len(values)):
            subsets[i][0] = np.array(subsets[i][0])
            subsets[i][1] = np.array(subsets[i][1])

        return {values[i]: subsets[i] for i in range(len(values))}

    def _compute_info(self, y):
        label_count = self._count_diff_values(y)
        num_instance = y.shape[0]
        p = np.array([label_count[i]/num_instance for i in label_count])
        return -np.dot(p, np.log2(p))

    def _count_diff_values(self, y):
        count = {}
        num_instance = y.shape[0]
        for i in range(num_instance):
            if y[i][0] in count:
                count[y[i][0]] += 1
            else:
                count[y[i][0]] = 1

        return count

    def _compute_intrinsic_info(self, split):
        p = np.array([split[value][1].shape[0] for value in split])
        p = p/p.sum()
        return -np.dot(p, np.log2(p))

    def _pre_process_data(self, X):
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if np.isnan(X[i][j]):
                    X[i][j] = 0.1