import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

SMALL = 1e-8
BIG = 1e3

class KNNClassifier(BaseEstimator,ClassifierMixin):


    def __init__(self, columntype=[], labeltype='classification', weight_type='inverse_distance', num_neighbour=5): ## add parameters here
        """
        Args:
            columntype for each column tells you if continues[real] or if nominal[categoritcal].
            weight_type: inverse_distance voting or if non distance weighting. Options = ["no_weight","inverse_distance"]
        """
        self.columntype = columntype #Note This won't be needed until part 5
        self.weight_type = weight_type
        self.num_neighbour = num_neighbour
        self.labeltype = labeltype
        assert self.labeltype in ['classification', 'regression']
        if self.columntype != []:
            for ct in self.columntype:
                assert ct in ['continuous', 'nominal']

    def fit(self, data, labels):
        """ Fit the data; run the algorithm (for this lab really just saves the data :D)
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        self.X = data
        self.y = labels

        if 'nominal' in self.columntype:
            self.hVDM_Cache = []

            label_values = set(self.y[:, 0])
            for i in range(len(self.columntype)):
                if self.columntype[i] == 'continuous':
                    sigma = np.nanstd(self.X[:, i])
                    self.hVDM_Cache.append(4 * sigma)
                else:
                    value_cnt = {}
                    column = self.X[:, i]
                    for j in range(len(column)):
                        value = column[j]
                        if not np.isnan(value):
                            if value not in value_cnt:
                                value_cnt[value] = {lv: 0 for lv in label_values}
                                value_cnt[value]['total'] = 1
                                value_cnt[value][self.y[j][0]] = 1
                            else:
                                value_cnt[value]['total'] += 1
                                value_cnt[value][self.y[j][0]] += 1
                    probs = {value: {lv: value_cnt[value][lv]/value_cnt[value]['total'] for lv in label_values} for value in value_cnt}
                    vdm = {}
                    for v1 in probs:
                        for v2 in probs:
                            # if v2 != v1:
                            vdm[(v1, v2)] = 0
                            for lv in label_values:
                                vdm[(v1, v2)] += np.abs(probs[v1][lv] - probs[v2][lv])
                    self.hVDM_Cache.append(vdm)
                    # compute value distance
        return self
    def predict(self, data):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        num_instance = data.shape[0]
        labels = np.zeros([num_instance, self.y.shape[1]])
        for i in range(num_instance):
            # neighbours = self._find_nearest_neighbours(data[i])
            neighbours = self._alt_find_nearest_neighbours(data[i])
            if self.weight_type is 'no_weight':
                if self.labeltype == 'classification':
                    vote = {}
                    for idx, dist in neighbours:
                        key = self.y[idx][0] if self.y.shape[1] == 1 else np.argmax(self.y[idx][0])

                        if key in vote:
                            vote[key] += 1
                        else:
                            vote[key] = 1

                    if self.y.shape[1] == 1:
                        labels[i] = max(vote, key=lambda x: vote[x])
                    else:
                        labels[i][max(vote, key=lambda x: vote[x])] = 1.0
                elif self.labeltype == 'regression':
                    labels[i] = np.mean([self.y[j[0]] for j in neighbours], axis=0)
            elif self.weight_type == 'inverse_distance':
                if self.labeltype == 'classification':
                    vote = {}
                    for idx, dist in neighbours:
                        key = self.y[idx][0] if self.y.shape[1] == 1 else np.argmax(self.y[idx][0])

                        if key in vote:
                            vote[key] += 1.0 / (dist if dist != 0 else SMALL)
                        else:
                            vote[key] = 1.0 / (dist if dist != 0 else SMALL)

                    if self.y.shape[1] == 1:
                        labels[i] = max(vote, key=lambda x: vote[x])
                    else:
                        labels[i][max(vote, key=lambda x: vote[x])] = 1.0
                elif self.labeltype == 'regression':
                    labels[i] = np.sum([self.y[j[0]] * 1.0/(j[1] if j[1] != 0 else SMALL) for j in neighbours], axis=0) / np.sum([1.0/(j[1] if j[1] != 0 else SMALL) for j in neighbours])

        return labels

    #Returns the Mean score given input data and labels
    def score(self, X, y):
            """ Return accuracy of model on a given dataset. Must implement own score function.
            Args:
                    X (array-like): A 2D numpy array with data, excluding targets
                    y (array-like): A 2D numpy array with targets
            Returns:
                    score : float
                            Mean accuracy of self.predict(X) wrt. y.
            """
            output = self.predict(X)
            if self.labeltype == 'classification':
                if y.shape[1] == 1:
                    accuracy = (output == y).mean()
                else:
                    accuracy = (output.argmax(axis=1) == y.argmax(axis=1)).mean()

            elif self.labeltype == 'regression':
                accuracy = np.sum(np.mean(np.square(y - output), axis=0))

            return accuracy

    def _find_nearest_neighbours(self, instance):
        neighbours = []
        for i in range(self.X.shape[0]):
            distance = self._euclidean_dist(self.X[i], instance)

            if len(neighbours) < self.num_neighbour:
                neighbours.append((i, distance))
            else:
                if distance < np.max(neighbours, axis=0)[1]:
                    neighbours[np.argmax(neighbours, axis=0)[1]] = (i, distance)

        return neighbours

    def _alt_find_nearest_neighbours(self, instance):
        distances = []
        if 'nominal' in self.columntype:
            # distances = []
            for i in range(self.X.shape[0]):
                distance = self._hVDM(self.X[i], instance)
                distances.append(distance)
        else:
            distances = np.sum(np.square((self.X - instance)), axis=1)

        neighbours = np.argsort(distances, kind='mergesort')

        return [(neighbours[i], distances[neighbours[i]]) for i in range(self.num_neighbour)]

    def _euclidean_dist(self, a, b):
        # return np.sqrt(np.square(a - b).sum())
        return np.square(a - b).sum()

    def _hVDM(self, a, b):
        l = []
        for i in range(len(self.columntype)):
            if np.isnan(a[i]) or np.isnan(b[i]):
                l.append(1)
            elif self.columntype[i] == 'nominal':
                if (a[i], b[i]) in self.hVDM_Cache[i]:
                    d = self.hVDM_Cache[i][(a[i], b[i])]
                    l.append(np.square(d))
                else:
                    l.append(BIG)
            else:
                d = np.abs(a[i] - b[i])/self.hVDM_Cache[i]
                l.append(np.square(d))

        return np.sum(l)