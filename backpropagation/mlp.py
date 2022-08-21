import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import copy
from sklearn.model_selection import train_test_split

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified.

class MLPClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self,lr=.1, momentum=0, shuffle=True,hidden_layer_widths=None, deterministic=None, validation_size=0.0, max_epoch=1000, stop_threshold=5):
        """ Initialize class with chosen hyperparameters.

        Args:
            lr (float): A learning rate / step size.
            shuffle(boolean): Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
            momentum(float): The momentum coefficent 
        Optional Args (Args we think will make your life easier):
            hidden_layer_widths (list(int)): A list of integers which defines the width of each hidden layer if hidden layer is none do twice as many hidden nodes as input nodes.
        Example:
            mlp = MLPClassifier(lr=.2,momentum=.5,shuffle=False,hidden_layer_widths = [3,3]),  <--- this will create a model with two hidden layers, both 3 nodes wide
        """
        self.hidden_layer_widths = hidden_layer_widths
        self.lr = lr
        self.momentum = momentum
        self.shuffle = shuffle
        self.validation_size = validation_size

        # self.improve_tolerance = improve_tolerance
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
        Optional Args (Args we think will make your life easier):
            initial_weights (array-like): allows the user to provide initial weights
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        '''data structure description:
        X_with_bias, y_copy: 2d np array
        X_with_bias[i], y_copy[i]: np vector
        activations: python list of np vectors representing layers of hidden nodes
        self.weights: python list of 2d np arrays representing weights for each layer
        output: np vector
        delta_weights: python list of 2d np arrays representing change of weights for each layer
        '''

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.validation_size, shuffle=self.shuffle)

        X_with_bias = self._add_bias_to_data(X_train)
        y_copy = copy.deepcopy(y_train)

        data_size = X_with_bias.shape[0]
        feature_num = X_with_bias.shape[1]
        label_length = y_copy.shape[1]
        '''build the network first'''
        # self.network = {'layer_num': 2 if self.hidden_layer_widths is None else len(self.hidden_layer_widths) + 1,
        #                 # 'input_instance': X_with_bias,
        #                 # 'target': y_copy,
        #                 'activations': []}
        layer_num = 2 if self.hidden_layer_widths is None else len(self.hidden_layer_widths) + 1
        self.activations = []
        if self.hidden_layer_widths is None:
            self.activations.append(np.append(np.zeros((feature_num - 1) * 2), 1.0))
        else:
            for i in self.hidden_layer_widths:
                self.activations.append(np.append(np.zeros(i), 1.0))

        self.weights = self.initialize_weights(feature_num, label_length, layer_num) if not initial_weights else initial_weights
        delta_weights = [np.zeros_like(self.weights[i]) for i in range(len(self.weights))]

        epoch_num = 0
        break_indicator = 0
        self.val_accuracy = [self.score(X_val, y_val)]
        self.train_mse = [self._compute_mse(X_train, y_train)]
        self.val_mse = [self._compute_mse(X_val, y_val)]
        bssf_val_mse = self.val_mse[0]
        self.final_weights = copy.deepcopy(self.weights)
        self.best_epoch = 0

        while epoch_num < self.max_epoch and break_indicator < self.stop_threshold:
            for i in range(data_size):
                output = self._forward(X_with_bias[i], label_length)
                delta_weights = self._list_add(self._backward(X_with_bias[i], output, y_copy[i]), delta_weights, self.momentum)
                for j in range(len(self.weights)):
                    self.weights[j] += delta_weights[j]

            epoch_num += 1

            self.val_accuracy.append(self.score(X_val, y_val))
            self.train_mse.append(self._compute_mse(X_train, y_train))

            new_val_mse = self._compute_mse(X_val, y_val)
            self.val_mse.append(new_val_mse)
            if new_val_mse < bssf_val_mse or X_val.shape[0] == 0:
                bssf_val_mse = new_val_mse
                break_indicator = 0
                self.final_weights = copy.deepcopy(self.weights)
                self.best_epoch = epoch_num
            else:
                break_indicator += 1
            # if np.abs(new_accuracy - old_accuracy) > self.improve_tolerance:
            #     break_indicator = 0
            # else:
            #     break_indicator += 1
            # old_accuracy = new_accuracy
            if self.shuffle:
                X_with_bias, y_copy = self._shuffle_data(X_with_bias, y_copy)
        self.weights = copy.deepcopy(self.final_weights)
        return self

    def predict(self, X):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        X_with_bias = self._add_bias_to_data(X)
        data_size = X_with_bias.shape[0]
        label_length = self.weights[-1].shape[0]
        outputs = np.zeros((data_size, label_length))

        for i in range(data_size):
            outputs[i] = self._forward(X_with_bias[i], label_length)

        if label_length == 1:
            real_outputs = 1.0 * (outputs > 0.5)
        else:
            real_outputs = np.zeros_like(outputs)
            real_outputs[np.arange(real_outputs.shape[0]), np.argmax(outputs, axis=1)] = 1.0
        return real_outputs

    def initialize_weights(self, feature_num, label_length, layer_num):
        """ Initialize weights for perceptron. Don't forget the bias!

        Returns:

        """
        '''weights of each layer is a matrix randomly initialized at (-1, 1)'''
        weights = [np.random.rand(self.activations[0].shape[0] - 1, feature_num) * 2 - 1]

        for i in range(layer_num - 2):
            weights.append(np.random.rand(self.activations[i+1].shape[0] - 1, self.activations[i].shape[0]) * 2 - 1)

        weights.append(np.random.rand(label_length, self.activations[-1].shape[0]) * 2 - 1)

        return weights

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
        if y.shape[1] == 1:
            accuracy = (output == y).mean()
        else:
            accuracy = (output.argmax(axis=1) == y.argmax(axis=1)).mean()

        return accuracy

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
        # return self.weights
        return self.weights

    def _add_bias_to_data(self, X):
        bias = np.ones([X.shape[0], 1])
        return np.append(X, bias, axis=1)

    def _forward(self, input_with_bias, label_length):
        for i in range(len(self.activations)):
            for j in range(self.activations[i].shape[0] - 1):
                if i == 0:
                    self.activations[i][j] = np.dot(self.weights[i][j], input_with_bias)
                else:
                    self.activations[i][j] = np.dot(self.weights[i][j], self.activations[i-1])
                self.activations[i][j] = 1/(1 + np.exp(-self.activations[i][j]))

        output = np.zeros(label_length)
        for j in range(label_length):
            output[j] = np.dot(self.weights[-1][j], self.activations[i])
        output = 1/(1 + np.exp(-output))

        return output

    def _backward(self, input_with_bias, output, target):
        delta_target = (target - output) * output * (1 - output)
        delta_activations = [np.zeros(self.activations[i].shape[0] - 1) for i in range(len(self.activations))]
        for j in range(delta_activations[-1].shape[0]):
            delta_activations[-1][j] = np.dot(delta_target, self.weights[-1][:, j]) * self.activations[-1][j] * (1 - self.activations[-1][j])
        for i in range(-2, -len(self.activations) - 1, -1):
            for j in range(delta_activations[i].shape[0]):
                delta_activations[i][j] = np.dot(delta_activations[i+1], self.weights[i][:, j]) * self.activations[i][j] * (1 - self.activations[i][j])

        delta_weights = [np.zeros(self.weights[i].shape) for i in range(len(self.weights))]
        delta_weights[0] = self.lr * np.matmul(np.expand_dims(delta_activations[0], 1), np.expand_dims(input_with_bias, 0))
        for i in range(1, len(delta_weights) - 1):
            delta_weights[i] = self.lr * np.matmul(np.expand_dims(delta_activations[i+1], 1), np.expand_dims(self.activations[i], 0))
        delta_weights[-1] = self.lr * np.matmul(np.expand_dims(delta_target, 1), np.expand_dims(self.activations[-1], 0))

        return delta_weights

    def _list_add(self, a, b, b_weight):
        return [a[i] + b_weight * b[i] for i in range(len(a))]

    def _compute_mse(self, X, y):
        output = self.predict(X)
        delta = y - output
        return (delta * delta).sum() / delta.shape[0]