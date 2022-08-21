from decisiontree import DTClassifier
import numpy as np
from tools.arff import Arff
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn import tree
import pandas as pd
import graphviz

def test():
    c = DTClassifier()
    X = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 2, 1],
                  [1, 2, 1],
                  [1, 1, 0],
                  [1, 1, 1],
                  [0, 0, 1],
                  [1, 1, 0],
                  [0, 0, 0]], dtype=float)
    y = np.array([[2], [0], [1], [2], [1], [2], [1], [1], [0]], dtype=float)

    # X = np.array([['s', 'h', 'h', 'f'],
    #               ['s', 'h', 'h', 't'],
    #               ['o', 'h', 'h', 'f'],
    #               ['r', 'm', 'h', 'f'],
    #               ['r', 'c', 'n', 'f'],
    #               ['r', 'c', 'n', 't'],
    #               ['o', 'c', 'n', 't'],
    #               ['s', 'm', 'h', 'f'],
    #               ['s', 'c', 'n', 'f'],
    #               ['r', 'm', 'n', 'f'],
    #               ['s', 'm', 'n', 't'],
    #               ['o', 'm', 'h', 't'],
    #               ['o', 'h', 'n', 'f'],
    #               ['r', 'm', 'h', 't']])
    # y = np.array([[0],
    #               [0],
    #               [1],
    #               [1],
    #               [1],
    #               [0],
    #               [1],
    #               [0],
    #               [1],
    #               [1],
    #               [1],
    #               [1],
    #               [1],
    #               [0]], dtype=float)
    # info = c._compute_info(y)
    # X1, y1 = c._one_split(X, y)[0][0.0]
    # split, attr = c._one_split(X1, y1)
    c.fit(X, y)
    yy = c.predict(X)
    accuray = c.score(X, y)
    # split = c._one_split(X, y)
    print(accuray)
    print(c.tree)

def debug_1():
    mat = Arff("../data/dt/lenses.arff")

    counts = []  ## this is so you know how many types for each column

    for i in range(mat.data.shape[1]):
        counts += [mat.unique_value_count(i)]
    data = mat.data[:, 0:-1]
    labels = mat.data[:, -1].reshape(-1, 1)

    DTClass = DTClassifier(counts)
    DTClass.fit(data, labels)
    mat2 = Arff("../data/dt/all_lenses.arff")
    data2 = mat2.data[:, 0:-1]
    labels2 = mat2.data[:, -1].reshape(-1, 1)
    pred = DTClass.predict(data2)
    Acc = DTClass.score(data2, labels2)
    np.savetxt("pred_lenses.csv", pred, delimiter=",")
    print("Accuracy = [{:.2f}]".format(Acc))


def debug_2():
    mat = Arff("../data/dt/lenses.arff")

    counts = []  ## this is so you know how many types for each column

    for i in range(mat.data.shape[1]):
        counts += [mat.unique_value_count(i)]
    data = mat.data[:, 0:-1]
    labels = mat.data[:, -1].reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, shuffle=True)

    DTClass = DTClassifier(counts)
    DTClass.fit(X_train, y_train)
    Acc = DTClass.score(X_test, y_test)
    print("Accuracy = [{:.2f}]".format(Acc))


def evaluation():
    mat = Arff("../data/dt/zoo.arff")

    counts = []  ## this is so you know how many types for each column

    for i in range(mat.data.shape[1]):
        counts += [mat.unique_value_count(i)]
    data = mat.data[:, 0:-1]
    labels = mat.data[:, -1].reshape(-1, 1)
    DTClass = DTClassifier(counts)
    DTClass.fit(data, labels)
    mat2 = Arff("../data/dt/all_zoo.arff")
    data2 = mat2.data[:, 0:-1]
    labels2 = mat2.data[:, -1].reshape(-1, 1)
    pred = DTClass.predict(data2)
    np.savetxt("pred_zoo.csv", pred, delimiter=",")
    Acc = DTClass.score(data2, labels2)
    print("Accuracy = [{:.2f}]".format(Acc))


def evaluation_2():
    mat = Arff("../data/dt/zoo.arff")

    counts = []  ## this is so you know how many types for each column

    for i in range(mat.data.shape[1]):
        counts += [mat.unique_value_count(i)]
    data = mat.data[:, 0:-1]
    labels = mat.data[:, -1].reshape(-1, 1)
    DTClass = DTClassifier(counts)
    DTClass.fit(data, labels)
    mat2 = Arff("../data/dt/all_zoo.arff")
    data2 = mat2.data[:, 0:-1]
    labels2 = mat2.data[:, -1].reshape(-1, 1)
    pred = DTClass.predict(data2)
    np.savetxt("pred_zoo_edit_dataset.csv", pred, delimiter=",")
    Acc = DTClass.score(data2, labels2)
    print("Accuracy = [{:.2f}]".format(Acc))


def cars():
    mat = Arff("../data/dt/cars.arff")

    counts = []  ## this is so you know how many types for each column

    for i in range(mat.data.shape[1]):
        counts += [mat.unique_value_count(i)]
    X = mat.data[:, 0:-1]
    y = mat.data[:, -1].reshape(-1, 1)

    ten_fold = KFold(n_splits=10, shuffle=True)
    ten_fold.get_n_splits(X, y)
    i = 0
    train_acc = np.zeros(10)
    test_acc = np.zeros(10)
    for train_index, test_index in ten_fold.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        c = DTClassifier()
        c.fit(X_train, y_train)
        train_acc[i] = c.score(X_train, y_train)
        test_acc[i] = c.score(X_test, y_test)

        print('Iteration: {}, training accuracy: {:.2f}, test accuracy: {:.2f}.'.format(i+1, train_acc[i], test_acc[i]))
        print(c.tree)
        i += 1
    print('average training accuracy: {:.2f}, average test accuracy: {:.2f}.'.format(train_acc.mean(), test_acc.mean()))

def voting():
    mat = Arff("../data/dt/voting.arff")

    counts = []  ## this is so you know how many types for each column

    for i in range(mat.data.shape[1]):
        counts += [mat.unique_value_count(i)]
    X = mat.data[:, 0:-1]
    y = mat.data[:, -1].reshape(-1, 1)

    ten_fold = KFold(n_splits=10)
    ten_fold.get_n_splits(X, y)
    i = 0
    train_acc = np.zeros(10)
    test_acc = np.zeros(10)
    for train_index, test_index in ten_fold.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        c = DTClassifier()
        c.fit(X_train, y_train)
        train_acc[i] = c.score(X_train, y_train)
        test_acc[i] = c.score(X_test, y_test)

        print(
            'Iteration: {}, training accuracy: {:.2f}, test accuracy: {:.2f}.'.format(i + 1, train_acc[i], test_acc[i]))
        print(c.tree)
        i += 1
    print('average training accuracy: {:.2f}, average test accuracy: {:.2f}.'.format(train_acc.mean(), test_acc.mean()))


def sk_cars():
    mat = Arff("../data/dt/cars.arff")

    counts = []  ## this is so you know how many types for each column

    for i in range(mat.data.shape[1]):
        counts += [mat.unique_value_count(i)]
    X = mat.data[:, 0:-1]
    y = mat.data[:, -1].reshape(-1, 1)

    ten_fold = KFold(n_splits=10, shuffle=True)
    ten_fold.get_n_splits(X, y)
    max_depth_choice = [3, 4, 5, None]
    min_impurity_choice = [0.02, 0.05, 0.08, 0.0]
    for depth in max_depth_choice:
        for impurity_threshold in min_impurity_choice:
            print('max depth:{}, impurity threshold:{}.\n'.format(depth, impurity_threshold))
            i = 0
            train_acc = np.zeros(10)
            test_acc = np.zeros(10)
            for train_index, test_index in ten_fold.split(X):
                # print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                c = DecisionTreeClassifier('entropy', max_depth=depth, min_impurity_decrease=impurity_threshold)
                c.fit(X_train, y_train)
                train_acc[i] = c.score(X_train, y_train)
                test_acc[i] = c.score(X_test, y_test)

                # print(
                #     'Iteration: {}, training accuracy: {:.2f}, test accuracy: {:.2f}.'.format(i + 1, train_acc[i], test_acc[i]))
                i += 1
            print('average training accuracy: {:.2f}, average test accuracy: {:.2f}.\n'.format(train_acc.mean(), test_acc.mean()))


def sk_voting():
    mat = Arff("../data/dt/voting.arff")

    counts = []  ## this is so you know how many types for each column

    for i in range(mat.data.shape[1]):
        counts += [mat.unique_value_count(i)]
    X = mat.data[:, 0:-1]
    y = mat.data[:, -1].reshape(-1, 1)

    imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0.1)
    imp.fit(X)
    X = imp.transform(X)

    ten_fold = KFold(n_splits=10, shuffle=True)
    ten_fold.get_n_splits(X, y)
    max_depth_choice = [8, 12, 14, None]
    min_impurity_choice = [0.02, 0.05, 0.08, 0.0]
    for depth in max_depth_choice:
        for impurity_threshold in min_impurity_choice:
            print('max depth:{}, impurity threshold:{}.\n'.format(depth, impurity_threshold))
            i = 0
            train_acc = np.zeros(10)
            test_acc = np.zeros(10)
            for train_index, test_index in ten_fold.split(X):
                # print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                c = DecisionTreeClassifier('entropy', max_depth=depth, min_impurity_decrease=impurity_threshold)
                c.fit(X_train, y_train)
                train_acc[i] = c.score(X_train, y_train)
                test_acc[i] = c.score(X_test, y_test)

                # print(
                #     'Iteration: {}, training accuracy: {:.2f}, test accuracy: {:.2f}.'.format(i + 1, train_acc[i], test_acc[i]))
                i += 1
            print('average training accuracy: {:.2f}, average test accuracy: {:.2f}.\n'.format(train_acc.mean(), test_acc.mean()))


def tic_tac_toe():
    df = pd.read_csv('../data/dt/tic-tac-toe-endgame.csv')
    X = np.array(df[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9' ]])
    x1 = np.zeros_like(X, dtype=float)
    x2 = np.ones_like(X, dtype=float) * (X == 'o')
    x3 = np.ones_like(X, dtype=float) * (X == 'b') * 2
    X = x1 + x2 + x3

    y = np.array(df['V10']).reshape(-1, 1)
    y = np.ones_like(y, dtype=float) * (y == 'positive')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, shuffle=True, random_state=32)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, shuffle=True, random_state=42)
    max_depth_choice = [5, 7, None]
    min_impurity_choice = [0.02, 0.05, 0.08, 0.0]
    criteria = ['gini', 'entropy']
    md = None
    mi = 0.0
    mc = 'entropy'
    max_test_acc = 0.0
    for depth in max_depth_choice:
        for impurity_threshold in min_impurity_choice:
            for ct in criteria:
                print('max depth:{}, impurity threshold:{}, criteria:{}.\n'.format(depth, impurity_threshold, ct))

                c = DecisionTreeClassifier(criterion=ct, max_depth=depth, min_impurity_decrease=impurity_threshold)

                c.fit(X_train, y_train)
                train_acc = c.score(X_train, y_train)
                test_acc = c.score(X_test, y_test)

                if test_acc > max_test_acc:
                    max_test_acc = test_acc
                    md = depth
                    mi = impurity_threshold
                    mc = ct
                print('training accuracy: {:.3f}, test accuracy: {:.3f}.\n'.format(train_acc, test_acc))


    print(md, mi, mc)
    c = DecisionTreeClassifier(criterion=mc, max_depth=md, min_impurity_decrease=mi)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=42)
    c.fit(X_train, y_train)
    train_acc = c.score(X_train, y_train)
    test_acc = c.score(X_test, y_test)

    print('best hyper parameter:\nmax depth:{}, impurity threshold:{}, criteria:{}.\n'.format(md, mi, mc))
    print('best accuracy:\ntraining accuracy: {:.3f}, test accuracy: {:.3f}.\n'.format(train_acc, test_acc))
    dot_data = tree.export_graphviz(c)
    graph = graphviz.Source(dot_data)
    graph.render('tree.pdf')


def weather():
    mat = Arff("../data/dt/weather.arff")

    counts = []  ## this is so you know how many types for each column

    for i in range(mat.data.shape[1]):
        counts += [mat.unique_value_count(i)]
    X = mat.data[:, 0:-1]
    y = mat.data[:, -1].reshape(-1, 1)

    # print(X, '\n', y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=32)

    max_depth_choice = [2, 3, None]
    min_impurity_choice = [0.02, 0.05, 0.08, 0.0]
    criteria = ['gini', 'entropy']
    md = None
    mi = 0.0
    mc = 'entropy'
    max_test_acc = 0.0
    for depth in max_depth_choice:
        for impurity_threshold in min_impurity_choice:
            for ct in criteria:
                print('max depth:{}, impurity threshold:{}, criteria:{}.\n'.format(depth, impurity_threshold, ct))

                c = DecisionTreeClassifier(criterion=ct, max_depth=depth, min_impurity_decrease=impurity_threshold)

                c.fit(X_train, y_train)
                train_acc = c.score(X_train, y_train)
                test_acc = c.score(X_test, y_test)

                if test_acc > max_test_acc:
                    max_test_acc = test_acc
                    md = depth
                    mi = impurity_threshold
                    mc = ct
                print('training accuracy: {:.3f}, test accuracy: {:.3f}.\n'.format(train_acc, test_acc))

    c = DecisionTreeClassifier(criterion=mc, max_depth=md, min_impurity_decrease=mi)
    c.fit(X_train, y_train)
    train_acc = c.score(X_train, y_train)
    test_acc = c.score(X_test, y_test)

    print('best hyper parameter:\nmax depth:{}, impurity threshold:{}, criteria:{}.\n'.format(md, mi, mc))
    print('best accuracy:\ntraining accuracy: {:.3f}, test accuracy: {:.3f}.\n'.format(train_acc, test_acc))
    dot_data = tree.export_graphviz(c,
                                    feature_names=['outlook', 'temperature', 'humidity', 'windy'],
                                    class_names=['play', 'not play'],
                                    filled=True,
                                    rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render('tree')

if __name__ == '__main__':
    evaluation_2()
