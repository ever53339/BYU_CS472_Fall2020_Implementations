import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import copy

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified. The output of
#   get_weights must be in the same format as the example provided.

from sklearn.linear_model import Perceptron

class PerceptronClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, lr=.1, shuffle=True, deterministic=None, max_epoch=1000, improve_tolerance=0.01, stop_threshold=5):
        """ Initialize class with chosen hyperparameters.

        Args:
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
        """
        self.lr = lr
        self.shuffle = shuffle
        self.improve_tolerance = improve_tolerance
        # self.deterministic = deterministic
        if deterministic is None:
            self.max_epoch = max_epoch
            self.stop_threshold = stop_threshold
        else:
            self.max_epoch = deterministic
            self.stop_threshold = np.inf

    def fit(self, X, y, initial_weights=None):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights

        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        assert len(X.shape) == 2 and len(y.shape) == 2, 'the dimension of features and labels should be 2.'
        assert y.shape[1] == 1, 'Each label should only have 1 binary value for a single perceptron node.'
        if initial_weights is not None:
            assert type(initial_weights) == np.ndarray, 'initial_weights should be an numpy array.'
            initial_weights = initial_weights.astype('float64')
            if len(initial_weights.shape) == 1:
                initial_weights = np.expand_dims(initial_weights, axis=0)

        data_size = X.shape[0]
        # X_with_bias = np.append(X, bias, axis=1)
        X_with_bias = self._add_bias_to_data(X)
        y_copy = copy.deepcopy(y)
        feature_num = X_with_bias.shape[1]
        label_length = y_copy.shape[1]
        self.weights = self.initialize_weights([label_length, feature_num]) if initial_weights is None else initial_weights

        output = np.zeros(y_copy.shape)
        epoch_num = 0
        break_indicator = 0
        old_accuracy = self.score(X, y)
        self.accuracy_variation = [old_accuracy]

        while epoch_num < self.max_epoch and break_indicator < self.stop_threshold:
            for i in np.arange(data_size):
                self._compute_single_output(X_with_bias, i, label_length, output)

                error = y_copy[i] - output[i]
                for j in np.arange(label_length):
                    self.weights[j] += error[j] * X_with_bias[i] * self.lr

            epoch_num += 1

            new_accuracy= self.score(X, y)
            self.accuracy_variation.append(new_accuracy)

            if np.abs(new_accuracy - old_accuracy) > self.improve_tolerance:
                break_indicator = 0
            else:
                break_indicator += 1
            old_accuracy = new_accuracy

            if self.shuffle:
                X_with_bias, y_copy = self._shuffle_data(X_with_bias, y_copy)

        self.epoch = epoch_num

        return self

    def predict(self, X):
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        assert len(X.shape) == 2, 'the dimension of features should be 2.'

        data_size = X.shape[0]
        X_with_bias = self._add_bias_to_data(X)

        label_length = self.weights.shape[0]
        output = np.zeros([data_size, label_length])

        for i in np.arange(data_size):
            self._compute_single_output(X_with_bias, i, label_length, output)

        return np.squeeze(output)

    def initialize_weights(self, shape):
        """ Initialize weights for perceptron. Don't forget the bias!

        Returns:

        """
        # weights are initialized as a real number between -1 and 1.
        return np.random.random(shape) * 2 - 1

    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets

        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """

        assert len(X.shape) == 2 and len(y.shape) == 2, 'the dimension of features and labels should be 2.'
        assert y.shape[1] == 1, 'Each label should only have 1 binary value for a single perceptron node.'

        output = np.expand_dims(self.predict(X), axis=1)
        true_cnt = np.count_nonzero(output == y)

        return true_cnt/y.size

    def _shuffle_data(self, X, y, seed=None):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """

        assert X.shape[0] == y.shape[0]
        if seed is not None:
            shuffled_idx = np.random.RandomState(seed=seed).permutation(X.shape[0])
        else:
            shuffled_idx = np.random.permutation(X.shape[0])

        return X[shuffled_idx], y[shuffled_idx]

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        return np.squeeze(self.weights)

    def _compute_single_output(self, X, i, label_length, output):
        net = np.dot(self.weights, X[i])
        for j in np.arange(label_length):
            if net[j] > 0:
                output[i][j] = 1
            else:
                output[i][j] = 0

    def _add_bias_to_data(self, X):
        bias = np.ones([X.shape[0], 1])
        return np.append(X, bias, axis=1)

    def split_data(self, X, y, training_proportion=0.7, seed=None):
        assert len(X.shape) == 2 and len(y.shape) == 2, 'the dimension of features and labels should be 2.'
        assert X.shape[0] == y.shape[0], 'dateset sizes for features and labels should match.'
        assert 0 < training_proportion < 1, 'training proportion should be in (0, 1).'

        X_shuffle, y_shuffle = self._shuffle_data(X, y, seed)
        size = X_shuffle.shape[0]
        training_X, test_X = np.split(X_shuffle, [int(size * training_proportion)])
        training_y, test_y = np.split(y_shuffle, [int(size * training_proportion)])

        return training_X, training_y, test_X, test_y
