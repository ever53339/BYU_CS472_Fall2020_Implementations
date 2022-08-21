from backpropagation import mlp
import numpy as np
from tools.arff import Arff
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV

def debug():
    c = mlp.MLPClassifier(lr=0.1, momentum=0.5, deterministic=10, shuffle=False, validation_size=0.0)

    weights = [np.zeros((4, 3)), np.zeros((1, 5))]

    mat = Arff('../data/perceptron/debug/linsep2nonorigin.arff', label_count=1)

    data = mat[:,:-1]
    labels = mat[:,-1].reshape(-1,1)
    c.fit(data, labels, initial_weights=weights)
    print(c.weights)


def evaluation():
    c = mlp.MLPClassifier(lr=0.1, momentum=0.5, deterministic=10, shuffle=False, validation_size=0.0)
    mat = Arff('../data/perceptron/evaluation/data_banknote_authentication.arff', label_count=1)
    weights = [np.zeros((8, 5)), np.zeros((1, 9))]
    data = mat[:, :-1]
    labels = mat[:, -1].reshape(-1, 1)
    c.fit(data, labels, initial_weights=weights)
    print(c.weights)
    weights_to_csv('evaluation.csv', c.weights)


def weights_to_csv(path, weights):
    with open(path, 'w') as fh:
        for i in range(len(weights)):
            for j in range(weights[i].shape[0]):
                for k in range(weights[i].shape[1]):
                    fh.write('{:4e}\n'.format(weights[i][j][k]))


def iris():
    c = mlp.MLPClassifier(lr=0.1, momentum=0.5, shuffle=True, validation_size=0.2, stop_threshold=40)
    mat = Arff('../data/perceptron/iris.arff')
    whole_data = mat[:, :-1]
    whole_raw_labels = mat[:, -1].reshape(-1, 1)
    whole_labels = np.zeros([whole_raw_labels.shape[0], 3])
    for i in np.arange(whole_raw_labels.shape[0]):
        if whole_raw_labels[i][0] == 0:
            whole_labels[i] = np.array([1.0, 0.0, 0.0])
        elif whole_raw_labels[i][0] == 1.0:
            whole_labels[i] = np.array([0.0, 1.0, 0.0])
        elif whole_raw_labels[i][0] == 2.0:
            whole_labels[i] = np.array([0.0, 0.0, 1.0])
    X_train, X_test, y_train, y_test = train_test_split(whole_data, whole_labels, test_size=0.25, shuffle=True)

    c.fit(X_train, y_train)
    print(c.score(X_test, y_test))
    fig, ax = plt.subplots()
    x_range = range(len(c.train_mse))
    ax.plot(x_range, c.train_mse, marker='o', markersize=2, label='train MSE', color='r')
    ax.plot(x_range, c.val_mse, marker='o', markersize=2, label='val MSE', color='b')
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('MSE', fontsize=14)

    ax2 = ax.twinx()
    ax2.plot(x_range, c.val_accuracy, linestyle='dashed', label='val accuracy', color='g')
    ax2.set_ylabel('Accuracy', fontsize=14)
    fig.legend(loc='lower right', bbox_to_anchor=(0.89, 0.45))
    plt.show()


def vowel_feature_accuracy():
    mat, whole_data, whole_raw_labels, whole_labels = get_vowel_dataset()

    c = mlp.MLPClassifier(lr=0.1, momentum=0.5, shuffle=True, validation_size=0.1, stop_threshold=40)

    '''baseline accuracy, average for 3 times'''
    _, _, _, _, test_accuracy = train_a_few_time(c, whole_data, whole_labels, avg_time=3)
    print('test accuracy for each training is {}, average is {:.4f}'.format(test_accuracy, test_accuracy.mean()))

    '''filtering train or not, accuracy'''
    filter_data = mat[:, 1:-1]
    _, _, _, _, test_accuracy = train_a_few_time(c, filter_data, whole_labels, avg_time=3)
    print('test accuracy for each training is {}, average is {:.4f}'.format(test_accuracy, test_accuracy.mean()))


def vowel_lr():
    # c = mlp.MLPClassifier(lr=0.1, momentum=0.5, shuffle=True, validation_size=0.1, stop_threshold=20)
    mat, whole_data, whole_raw_labels, whole_labels = get_vowel_dataset()
    filter_data = mat[:, 1:-1]

    '''different lr'''
    best_epoch = {}
    train_mse = {}
    val_mse = {}
    test_mse = {}
    test_accuracy = {}
    # all_lr = [0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0]
    all_lr = [0.01, 0.03, 0.1, 0.3, 1.0]
    # all_lr = [0.01, 0.1, 1.0]
    for lr in all_lr:
        c =  mlp.MLPClassifier(lr=lr, momentum=0.5, shuffle=True, validation_size=0.1, stop_threshold=40)
        best_epoch[lr], train_mse[lr], val_mse[lr], test_mse[lr], test_accuracy[lr] = train_a_few_time(c, filter_data, whole_labels, avg_time=3)

    # best_epoch = {0.01: np.array([23, 45, 36]), 0.03: np.array([23, 15, 46]), 0.1: np.array([66, 75, 46]), 0.3: np.array([56, 55, 23]), 1.0: np.array([12, 8, 20])}
    # train_mse = {0.01: np.array([0.35, 0.75, 0.99]), 0.03: np.array([0.23, 0.15, 0.46]), 0.1: np.array([0.18, 0.45, 0.85]), 0.3: np.array([0.56, 0.55, 0.23]), 1.0: np.array([1.28, 1.75, 1.69])}
    # val_mse = {0.01: np.array([0.25, 0.55, 0.69]), 0.03: np.array([0.23, 0.15, 0.46]), 0.1: np.array([0.18, 0.25, 0.35]), 0.3: np.array([0.56, 0.55, 0.23]), 1.0: np.array([0.68, 0.45, 0.45])}
    # test_mse = {0.01: np.array([0.68, 0.99, 0.12]), 0.03: np.array([0.23, 0.15, 0.46]), 0.1: np.array([0.92, 0.85, 0.97]), 0.3: np.array([0.56, 0.55, 0.23]), 1.0: np.array([0.98, 0.95, 0.85])}

    '''plot test accuracy'''
    x_plot = np.arange(len(all_lr)) * 2
    fig, ax = plt.subplots()
    test_accuracy_plot = []
    for lr in all_lr:
        test_accuracy_plot.append(test_accuracy[lr].mean())
    ax.plot(x_plot, test_accuracy_plot)
    ax.set_ylabel('Test Accuracy', fontsize=14)
    ax.set_xlabel('Learning Rate', fontsize=14)
    ax.set_xticks(x_plot)
    ax.set_xticklabels(all_lr)
    plt.show()

    '''plot mse'''

    width = 0.4
    fig, ax = plt.subplots()
    train_mse_plot = []
    val_mse_plot = []
    test_mse_plot = []
    for lr in all_lr:
        train_mse_plot.append(train_mse[lr].mean())
        val_mse_plot.append(val_mse[lr].mean())
        test_mse_plot.append(test_mse[lr].mean())
    rects1 = ax.bar(x_plot - width, train_mse_plot, width=width, label='Train')
    rects2 = ax.bar(x_plot, val_mse_plot, width=width, label='Validation')
    rects3 = ax.bar(x_plot + width, test_mse_plot, width=width, label='Test')

    ax.set_ylabel('Best MSE', fontsize=14)
    ax.set_xlabel('Learning Rate', fontsize=14)
    ax.set_xticks(x_plot)
    ax.set_xticklabels(all_lr)
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)
    ax.set_ylim(top=max((max(train_mse_plot), max(val_mse_plot), max(test_mse_plot))) * 1.1)
    ax.legend(loc='upper center')
    plt.show()

    '''plot epoch'''
    fig, ax = plt.subplots()
    epoch_plot = []
    for lr in all_lr:
        epoch_plot.append(best_epoch[lr].mean())

    rects4 = ax.bar(x_plot, epoch_plot, width=0.35)
    # ax.scatter(all_lr, epoch_plot)
    ax.set_ylabel('Stopping Epoch', fontsize=14)
    ax.set_xlabel('Learning Rate', fontsize=14)
    ax.set_xticks(x_plot)
    ax.set_xticklabels(all_lr)
    ax.set_ylim(top=max(epoch_plot) * 1.1)
    autolabel(rects4, ax)
    plt.show()

def vowel_hidden_node():
    mat, whole_data, whole_raw_labels, whole_labels = get_vowel_dataset()
    filter_data = mat[:, 1:-1]

    '''train different number of hidden nodes'''
    lr = 0.1
    best_epoch = {}
    train_mse = {}
    val_mse = {}
    test_mse = {}
    test_accuracy = {}
    # all_num = [1, 2, 4, 8, 16, 32, 64, 128]
    all_num = [256]
    for num_hid_nodes in all_num:
        c =  mlp.MLPClassifier(lr=lr, momentum=0.5, hidden_layer_widths=[num_hid_nodes], shuffle=True, validation_size=0.1, stop_threshold=40)
        best_epoch[num_hid_nodes], train_mse[num_hid_nodes], val_mse[num_hid_nodes], test_mse[num_hid_nodes], test_accuracy[num_hid_nodes] = train_a_few_time(c, filter_data, whole_labels, avg_time=3)

    '''plot test accuracy'''
    x_plot = np.arange(len(all_num)) * 2
    fig, ax = plt.subplots()
    test_accuracy_plot = []
    for num_hid_nodes in all_num:
        test_accuracy_plot.append(test_accuracy[num_hid_nodes].mean())
    ax.plot(x_plot, test_accuracy_plot)
    ax.set_ylabel('Test Accuracy', fontsize=14)
    ax.set_xlabel('Hidden Node Number', fontsize=14)
    ax.set_xticks(x_plot)
    ax.set_xticklabels(all_num)
    plt.show()

    '''plot mse'''
    width = 0.4
    fig, ax = plt.subplots()
    train_mse_plot = []
    val_mse_plot = []
    test_mse_plot = []
    for num_hid_nodes in all_num:
        train_mse_plot.append(train_mse[num_hid_nodes].mean())
        val_mse_plot.append(val_mse[num_hid_nodes].mean())
        test_mse_plot.append(test_mse[num_hid_nodes].mean())
    rects1 = ax.bar(x_plot - width, train_mse_plot, width=width, label='Train')
    rects2 = ax.bar(x_plot, val_mse_plot, width=width, label='Validation')
    rects3 = ax.bar(x_plot + width, test_mse_plot, width=width, label='Test')

    ax.set_ylabel('Best MSE', fontsize=14)
    ax.set_xlabel('Hidden Node Number', fontsize=14)
    ax.set_xticks(x_plot)
    ax.set_xticklabels(all_num)
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)
    ax.set_ylim(top=max((max(train_mse_plot), max(val_mse_plot), max(test_mse_plot))) * 1.1)
    ax.legend(loc='best')
    plt.show()
    # print(test_accuracy_plot, train_mse_plot, val_mse_plot, test_mse_plot)


def vowel_hidden_node_supplement():

    all_num = [1, 2, 4, 8, 16, 32, 64, 128, 256]


    '''plot test accuracy'''
    x_plot = np.arange(len(all_num)) * 2
    fig, ax = plt.subplots()
    test_accuracy_plot = [0.19,	0.227, 0.485, 0.601, 0.71, 0.767, 0.822, 0.869, 0.773]
    # for num_hid_nodes in all_num:
    #     test_accuracy_plot.append(test_accuracy[num_hid_nodes].mean())
    ax.plot(x_plot, test_accuracy_plot)
    ax.set_ylabel('Test Accuracy', fontsize=14)
    ax.set_xlabel('Hidden Node Number', fontsize=14)
    ax.set_xticks(x_plot)
    ax.set_xticklabels(all_num)
    plt.show()

    '''plot mse'''
    width = 0.4
    fig, ax = plt.subplots()
    train_mse_plot = [1.63,	1.51,	1.0,	0.66,	0.48,	0.36,	0.18,	0.15,	0.31]
    val_mse_plot = [1.49,	1.41,	1.01,	0.57,	0.51,	0.33,	0.17,	0.29,	0.41]
    test_mse_plot = [1.61,	1.54,	1.05,	0.83,	0.61,	0.5,	0.39,	0.3,	0.45]
    # for num_hid_nodes in all_num:
    #     train_mse_plot.append(train_mse[num_hid_nodes].mean())
    #     val_mse_plot.append(val_mse[num_hid_nodes].mean())
    #     test_mse_plot.append(test_mse[num_hid_nodes].mean())
    rects1 = ax.bar(x_plot - width, train_mse_plot, width=width, label='Train')
    rects2 = ax.bar(x_plot, val_mse_plot, width=width, label='Validation')
    rects3 = ax.bar(x_plot + width, test_mse_plot, width=width, label='Test')

    ax.set_ylabel('Best MSE', fontsize=14)
    ax.set_xlabel('Hidden Node Number', fontsize=14)
    ax.set_xticks(x_plot)
    ax.set_xticklabels(all_num)
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)
    ax.set_ylim(top=max((max(train_mse_plot), max(val_mse_plot), max(test_mse_plot))) * 1.1)
    ax.legend(loc='best')
    plt.show()


def vowel_momentum():
    mat, whole_data, whole_raw_labels, whole_labels = get_vowel_dataset()
    filter_data = mat[:, 1:-1]

    '''train different number of hidden nodes'''
    lr = 0.1
    num_hid_nodes = 128
    best_epoch = {}
    train_mse = {}
    val_mse = {}
    test_mse = {}
    test_accuracy = {}
    all_momentums = [0.1, 0.3, 0.5, 0.7, 0.9]
    for m in all_momentums:
        c = mlp.MLPClassifier(lr=lr, momentum=m, hidden_layer_widths=[num_hid_nodes], shuffle=True,
                              validation_size=0.1, stop_threshold=40)
        best_epoch[m], train_mse[m], val_mse[m], test_mse[m], test_accuracy[m] = train_a_few_time(c, filter_data,
                                                                                                       whole_labels,
                                                                                                       avg_time=3)

    '''plot test accuracy'''
    x_plot = np.arange(len(all_momentums)) * 2
    fig, ax = plt.subplots()
    test_accuracy_plot = []
    for m in all_momentums:
        test_accuracy_plot.append(test_accuracy[m].mean())
    ax.plot(x_plot, test_accuracy_plot)
    ax.set_ylabel('Test Accuracy', fontsize=14)
    ax.set_xlabel('Momentum', fontsize=14)
    ax.set_xticks(x_plot)
    ax.set_xticklabels(all_momentums)
    plt.show()

    '''plot epoch'''
    fig, ax = plt.subplots()
    epoch_plot = []
    for m in all_momentums:
        epoch_plot.append(best_epoch[m].mean())

    rects4 = ax.bar(x_plot, epoch_plot, width=0.35)
    ax.set_ylabel('Stopping Epoch', fontsize=14)
    ax.set_xlabel('Learning Rate', fontsize=14)
    ax.set_xticks(x_plot)
    ax.set_xticklabels(all_momentums)
    autolabel(rects4, ax)
    ax.set_ylim(top=max(epoch_plot) * 1.1)
    plt.show()


def vowel_momentum_supplement():

    all_momentums = [0.1, 0.3, 0.5, 0.7, 0.9]
    x_plot = np.arange(len(all_momentums)) * 2

    '''plot epoch'''
    fig, ax = plt.subplots()
    epoch_plot = [196.33, 136.00, 131.07, 91.67, 144.00]
    # for m in all_momentums:
    #     epoch_plot.append(best_epoch[m].mean())

    rects4 = ax.bar(x_plot, epoch_plot, width=0.35)
    ax.set_ylabel('Stopping Epoch', fontsize=14)
    ax.set_xlabel('Momentum', fontsize=14)
    ax.set_xticks(x_plot)
    ax.set_xticklabels(all_momentums)
    autolabel(rects4, ax)
    ax.set_ylim(top=max(epoch_plot) * 1.1)
    plt.show()


def sklearn_performance_iris():

    mat = Arff('../data/perceptron/iris.arff')
    whole_data = mat[:, :-1]
    whole_raw_labels = mat[:, -1].reshape(-1, 1)
    whole_labels = np.zeros([whole_raw_labels.shape[0], 3])
    for i in np.arange(whole_raw_labels.shape[0]):
        if whole_raw_labels[i][0] == 0:
            whole_labels[i] = np.array([1.0, 0.0, 0.0])
        elif whole_raw_labels[i][0] == 1.0:
            whole_labels[i] = np.array([0.0, 1.0, 0.0])
        elif whole_raw_labels[i][0] == 2.0:
            whole_labels[i] = np.array([0.0, 0.0, 1.0])
    X_train, X_test, y_train, y_test = train_test_split(whole_data, whole_labels, test_size=0.25, shuffle=True)

    c = MLPClassifier(hidden_layer_sizes=(8,), learning_rate='constant', learning_rate_init=0.1, momentum=0.1, validation_fraction=0.2,
                      early_stopping=True, n_iter_no_change=40)
    c.fit(X_train, y_train)
    print(c.score(X_test, y_test))
    print(c.n_iter_)


def sklearn_performance_vowel():
    mat, whole_data, whole_raw_labels, whole_labels = get_vowel_dataset()
    filter_data = mat[:, 1:-1]
    X_train, X_test, y_train, y_test = train_test_split(filter_data, whole_labels, test_size=0.25, shuffle=True)

    c =  MLPClassifier(hidden_layer_sizes=(128, ), learning_rate='constant', learning_rate_init=0.1, momentum=0.1, validation_fraction=0.1,
                       early_stopping=True, n_iter_no_change=40)
    c.fit(X_train, y_train)
    print(c.score(X_test, y_test))
    print(c.n_iter_)


def sklearn_grid_search():
    mat, whole_data, whole_raw_labels, whole_labels = get_vowel_dataset()
    filter_data = mat[:, 1:-1]
    X_train, X_test, y_train, y_test = train_test_split(filter_data, whole_labels, test_size=0.5, shuffle=True)

    tuned_parameters = {'hidden_layer_sizes': [(128, ), (64, ), (32, ), (16, ), (8, ), (4, ), (2, ), (1, )],
                         'learning_rate': ['constant'],
                         'learning_rate_init': [0.01, 0.03, 0.1, 0.3, 1.0],
                         'momentum': [0.1, 0.3, 0.5, 0.7, 0.9],
                         'validation_fraction': [0.1],
                         'early_stopping': [True], 'n_iter_no_change': [40]}

    clf = GridSearchCV(MLPClassifier(), tuned_parameters)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print('Best score found on development set:')
    print(clf.best_score_)
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()


def self_grid_search():
    mat, whole_data, whole_raw_labels, whole_labels = get_vowel_dataset()
    filter_data = mat[:, 1:-1]
    X_train, X_test, y_train, y_test = train_test_split(filter_data, whole_labels, test_size=0.5, shuffle=True)

    tuned_parameters = {'hidden_layer_widths': [[128], [64], [32], [16], [8], [4], [2], [1]],
                        'lr': [0.01, 0.03, 0.1, 0.3, 1.0],
                        'momentum': [0.1, 0.3, 0.5, 0.7, 0.9],
                        'validation_size': [0.1],
                        'stop_threshold': [40]}
    # c = mlp.MLPClassifier(lr=lr, momentum=m, hidden_layer_widths=[num_hid_nodes], shuffle=True,
    #                       validation_size=0.1, stop_threshold=40)
    clf = GridSearchCV(mlp.MLPClassifier(), tuned_parameters, n_jobs=4)
    clf.fit(X_train, y_train)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print('Best score found on development set:')
    print(clf.best_score_)
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()


def self_random_search():
    mat, whole_data, whole_raw_labels, whole_labels = get_vowel_dataset()
    filter_data = mat[:, 1:-1]
    X_train, X_test, y_train, y_test = train_test_split(filter_data, whole_labels, test_size=0.5, shuffle=True)

    tuned_parameters = {'hidden_layer_widths': [[128], [64], [32], [16], [8], [4], [2], [1]],
                        'lr': [0.01, 0.03, 0.1, 0.3, 1.0],
                        'momentum': [0.1, 0.3, 0.5, 0.7, 0.9],
                        'validation_size': [0.1],
                        'stop_threshold': [40]}

    clf = RandomizedSearchCV(mlp.MLPClassifier(), tuned_parameters, n_iter=10, n_jobs=4)
    clf.fit(X_train, y_train)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print('Best score found on development set:')
    print(clf.best_score_)
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

def get_vowel_dataset():
    mat = Arff('../data/vowel.arff')
    whole_data = mat[:, :-1]
    whole_raw_labels = mat[:, -1].reshape(-1, 1)
    whole_labels = np.zeros([whole_raw_labels.shape[0], 11])
    # print(whole_raw_labels.dtype)
    for i in range(whole_raw_labels.shape[0]):
        p = int(whole_raw_labels[i][0])
        whole_labels[i][p] = 1.0

    return mat, whole_data, whole_raw_labels, whole_labels

def train_a_few_time(c, X, y, avg_time):
    test_accuracy = np.zeros(avg_time)
    train_mse = np.zeros(avg_time)
    val_mse = np.zeros(avg_time)
    best_epoch = np.zeros(avg_time)
    test_mse = np.zeros(avg_time)

    for i in range(avg_time):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)
        c.fit(X_train, y_train)

        test_accuracy[i] = c.score(X_test, y_test)
        train_mse[i] = c.train_mse[c.best_epoch]
        val_mse[i] = c.val_mse[c.best_epoch]
        best_epoch[i] = c.best_epoch
        test_mse[i] = c._compute_mse(X_test, y_test)

        # print('Trial {}, accuracy: {:.4f}.'.format(i + 1, c.score(X_test, y_test)))
    # accuracy = accuracy / avg_time
    return best_epoch, train_mse, val_mse, test_mse, test_accuracy


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height), fontsize=7,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 2),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def test():
    mat, whole_data, whole_raw_labels, whole_labels = get_vowel_dataset()
    c = mlp.MLPClassifier(lr=0.1, momentum=0.5, shuffle=True, validation_size=0.1, stop_threshold=40)

    c.fit(whole_data, whole_labels)
    be = c.best_epoch
    train_mse = c.train_mse[be]
    val_mse = c.val_mse[be]
    val_accuracy = c.val_accuracy[be]
    test_mse = c._compute_mse(whole_data, whole_labels)



    print(be, train_mse, val_mse, val_accuracy, test_mse)

    c.fit(whole_data, whole_labels)
    print(be, train_mse, val_mse, val_accuracy, test_mse)
    be = c.best_epoch
    train_mse = c.train_mse[be]
    val_mse = c.val_mse[be]
    val_accuracy = c.val_accuracy[be]

    print(be, train_mse, val_mse, val_accuracy, test_mse)

if __name__ == '__main__':
    evaluation()
