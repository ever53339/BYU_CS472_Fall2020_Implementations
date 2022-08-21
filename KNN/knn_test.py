from KNN.KNN import KNNClassifier
import numpy as np
from tools.arff import Arff
from sklearn.model_selection import train_test_split
import time
from sklearn import neighbors
import matplotlib.pyplot as plt

def homework():
    data = np.array([[0.3, 0.8], [-0.3, 1.6], [0.9, 0.0], [1.0, 1.0]])
    cl = np.array([[0], [1], [1], [0]], dtype=float)
    rl = np.array([[0.6], [-0.3], [0.8], [1.2]])

    data2 = np.array([[0.5, 0.2]])

    # KNN = KNNClassifier(labeltype='classification', weight_type='no_weight', num_neighbour=3)
    # KNN.fit(data, cl)
    # print(KNN.predict(data2))

    KNN = KNNClassifier(labeltype='regression', weight_type='inverse_distance', num_neighbour=3)
    KNN.fit(data, rl)
    print(KNN.predict(data2))

def debug():
    mat = Arff("./data/knn/seismic-bumps_train.arff", label_count=1)
    mat2 = Arff("./data/knn/seismic-bumps_test.arff", label_count=1)
    raw_data = mat.data
    h, w = raw_data.shape
    train_data = raw_data[:, :-1]
    train_labels = raw_data[:, -1].reshape(-1, 1)

    raw_data2 = mat2.data
    h2, w2 = raw_data2.shape
    test_data = raw_data2[:, :-1]
    test_labels = raw_data2[:, -1].reshape(-1, 1)

    KNN = KNNClassifier(weight_type='inverse_distance', num_neighbour=3)
    KNN.fit(train_data, train_labels)
    pred = KNN.predict(test_data)
    score = KNN.score(test_data, test_labels)
    np.savetxt("seismic-bump-prediction.csv", pred, delimiter=',', fmt="%i")
    print("Acc = [{:.4f}]".format(score))


def evaluation():
    mat = Arff("data/knn/diabetes.arff", label_count=1)
    mat2 = Arff("data/knn/diabetes_test.arff", label_count=1)
    raw_data = mat.data
    h, w = raw_data.shape
    train_data = raw_data[:, :-1]
    train_labels = raw_data[:, -1].reshape(-1, 1)

    raw_data2 = mat2.data
    h2, w2 = raw_data2.shape
    test_data = raw_data2[:, :-1]
    test_labels = raw_data2[:, -1].reshape(-1, 1)

    KNN = KNNClassifier(labeltype ='classification', weight_type='inverse_distance', num_neighbour=3)
    KNN.fit(train_data, train_labels)
    pred = KNN.predict(test_data)
    score = KNN.score(test_data, test_labels)
    np.savetxt("diabetes_test_prediction.csv", pred, delimiter=',', fmt="%i")
    print("Acc = [{:.4f}]".format(score))


def telescope():
    mat = Arff("./data/knn/telescope_train.arff", label_count=1)
    mat2 = Arff("./data/knn/telescope_test.arff", label_count=1)
    raw_data = mat.data
    h, w = raw_data.shape
    train_data = raw_data[:, :-1]
    train_labels = raw_data[:, -1].reshape(-1, 1)

    raw_data2 = mat2.data
    h2, w2 = raw_data2.shape
    test_data = raw_data2[:, :-1]
    test_labels = raw_data2[:, -1].reshape(-1, 1)

    'without normalization'
    KNN = KNNClassifier(labeltype='classification', weight_type='no_weight', num_neighbour=3)
    KNN.fit(train_data, train_labels)
    # pred = KNN.predict(test_data)
    score = KNN.score(test_data, test_labels)
    # np.savetxt("diabetes_test_prediction.csv", pred, delimiter=',', fmt="%i")
    print("telescope, without normalization, Acc = [{:.4f}]\n".format(score))

    'with normalization'
    min = np.min(train_data, axis=0)
    max= np.max(train_data, axis=0)

    norm_train_data = normalize(train_data, min,  max)
    norm_test_data = normalize(test_data, min, max)

    KNN.fit(norm_train_data, train_labels)
    score = KNN.score(norm_test_data, test_labels)

    print("telescope, with normalization, Acc = [{:.4f}]\n".format(score))

    for k in range(1, 16):
        KNN = KNNClassifier(labeltype='classification', weight_type='no_weight', num_neighbour=k)
        KNN.fit(norm_train_data, train_labels)
        score = KNN.score(norm_test_data, test_labels)

        print("telescope, with normalization, no weight, {} neighbours, Acc = [{:.4f}]\n".format(k, score))

    for k in range(1, 16):
        KNN = KNNClassifier(labeltype='classification', weight_type='inverse_distance', num_neighbour=k)
        KNN.fit(norm_train_data, train_labels)
        score = KNN.score(norm_test_data, test_labels)

        print("telescope, with normalization, inverse distance, {} neighbours, Acc = [{:.4f}]\n".format(k, score))


def housing():
    mat = Arff("./data/knn/housing_train.arff", label_count=1)
    mat2 = Arff("./data/knn/housing_test.arff", label_count=1)
    raw_data = mat.data
    h, w = raw_data.shape
    train_data = raw_data[:, :-1]
    train_labels = raw_data[:, -1].reshape(-1, 1)

    raw_data2 = mat2.data
    h2, w2 = raw_data2.shape
    test_data = raw_data2[:, :-1]
    test_labels = raw_data2[:, -1].reshape(-1, 1)

    'with normalization'
    min = np.min(train_data, axis=0)
    max = np.max(train_data, axis=0)

    norm_train_data = normalize(train_data, min, max)
    norm_test_data = normalize(test_data, min, max)

    for k in range(1, 16):
        KNN = KNNClassifier(labeltype='regression', weight_type='no_weight', num_neighbour=k)
        KNN.fit(norm_train_data, train_labels)
        score = KNN.score(norm_test_data, test_labels)

        print("housing, with normalization, no weight, {} neighbours, MSE = [{:.4f}]\n".format(k, score))

    for k in range(1, 16):
        KNN = KNNClassifier(labeltype='regression', weight_type='inverse_distance', num_neighbour=k)
        KNN.fit(norm_train_data, train_labels)
        score = KNN.score(norm_test_data, test_labels)

        print("housing, with normalization, inverse distance, {} neighbours, MSE = [{:.4f}]\n".format(k, score))


def credit_a():
    mat = Arff("./data/knn/credit_a.arff", label_count=1)
    raw_data = mat.data
    h, w = raw_data.shape
    data = raw_data[:, :-1]
    labels = raw_data[:, -1].reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, shuffle=True, random_state=32)
    for num_neighbour in range(1, 16):
    # num_neighbour = 5
        KNN = KNNClassifier(columntype=['nominal', 'continuous', 'continuous',
                                        'nominal', 'nominal', 'nominal', 'nominal',
                                        'continuous', 'nominal', 'nominal', 'continuous',
                                        'nominal', 'nominal', 'continuous', 'continuous'],
                            labeltype='classification', weight_type='inverse_distance', num_neighbour=num_neighbour)

        KNN.fit(X_train, y_train)
        score = KNN.score(X_test, y_test)
        print("{} neighbours, Acc = [{:.4f}]".format(num_neighbour, score))
    # num_neighbour = 1
    # KNN = KNNClassifier(columntype=['nominal', 'continuous', 'continuous',
    #                                 'nominal', 'nominal', 'nominal', 'nominal',
    #                                 'continuous', 'nominal', 'nominal', 'continuous',
    #                                 'nominal', 'nominal', 'continuous', 'continuous'],
    #                     labeltype='classification', weight_type='inverse_distance', num_neighbour=num_neighbour)
    #
    # KNN.fit(X_train, y_train)
    # score = KNN.score(X_train, y_train)
    # print("{} neighbours, Acc = [{:.4f}]".format(num_neighbour, score))


def sk_telescope():
    mat = Arff("./data/knn/telescope_train.arff", label_count=1)
    mat2 = Arff("./data/knn/telescope_test.arff", label_count=1)
    raw_data = mat.data
    h, w = raw_data.shape
    train_data = raw_data[:, :-1]
    train_labels = raw_data[:, -1].reshape(-1, 1)

    raw_data2 = mat2.data
    h2, w2 = raw_data2.shape
    test_data = raw_data2[:, :-1]
    test_labels = raw_data2[:, -1].reshape(-1, 1)

    'with normalization'
    min = np.min(train_data, axis=0)
    max = np.max(train_data, axis=0)

    norm_train_data = normalize(train_data, min, max)
    norm_test_data = normalize(test_data, min, max)

    W = ['uniform', 'distance']
    N = [3, 5, 7]
    P = [1, 2]
    for p in P:
        for n in N:
            for w in W:
                c = neighbors.KNeighborsClassifier(n_neighbors=n, weights=w, p=p)
                c.fit(norm_train_data, np.squeeze(train_labels))
                score = c.score(norm_test_data, test_labels)
                print('p: {}， n: {}, w: {}, Acc: {:.4f}\n'.format(p, n, w, score))


def sk_housing():
    mat = Arff("./data/knn/housing_train.arff", label_count=1)
    mat2 = Arff("./data/knn/housing_test.arff", label_count=1)
    raw_data = mat.data
    h, w = raw_data.shape
    train_data = raw_data[:, :-1]
    train_labels = raw_data[:, -1].reshape(-1, 1)

    raw_data2 = mat2.data
    h2, w2 = raw_data2.shape
    test_data = raw_data2[:, :-1]
    test_labels = raw_data2[:, -1].reshape(-1, 1)

    'with normalization'
    min = np.min(train_data, axis=0)
    max = np.max(train_data, axis=0)

    norm_train_data = normalize(train_data, min, max)
    norm_test_data = normalize(test_data, min, max)

    W = ['uniform', 'distance']
    N = [3, 5, 7]
    P = [1, 2]
    for p in P:
        for n in N:
            for w in W:
                c = neighbors.KNeighborsRegressor(n_neighbors=n, weights=w, p=p)
                c.fit(norm_train_data, np.squeeze(train_labels))
                score = c.score(norm_test_data, test_labels)
                print('p: {}， n: {}, w: {}, Acc: {:.4f}\n'.format(p, n, w, score))


def abalone():
    mat = Arff("./data/knn/abalone.arff", label_count=1)
    raw_data = mat.data
    h, w = raw_data.shape
    data = raw_data[:, :-1]
    labels = raw_data[:, -1].reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, shuffle=True, random_state=46)

    # min = np.min(X_train, axis=0)
    # max = np.max(X_train, axis=0)
    #
    # norm_train_data = normalize(X_train, min, max)
    # norm_test_data = normalize(X_test, min, max)

    W = ['uniform', 'distance']
    N = [3, 5, 7]
    P = [1, 2]
    for p in P:
        for n in N:
            for w in W:
                c = neighbors.KNeighborsRegressor(n_neighbors=n, weights=w, p=p)
                c.fit(X_train, np.squeeze(y_train))
                score = c.score(X_test, y_test)
                print('p: {}， n: {}, w: {}, Acc: {:.4f}\n'.format(p, n, w, score))

def plot_telescope():
    k = range(1, 16)
    a1 = [0.8114, 0.8114, 0.8306, 0.8335, 0.8444, 0.8437,
          0.8438, 0.8458, 0.8446, 0.8476, 0.8462, 0.8482, 0.8465, 0.8476, 0.8429]
    a2 = [0.8114, 0.8114, 0.8311, 0.8338, 0.8456, 0.8468, 0.8491,
          0.8512, 0.8485, 0.8513, 0.8528, 0.8531, 0.8516, 0.8530, 0.8507]
    print(np.argmax(a1), np.argmax(a2))
    fig, ax = plt.subplots()
    ax.plot(k, a1, marker='v', label='no weights')
    ax.plot(k, a2, marker='o', label='inverse distance')
    ax.set_xlabel('number of neighbors')
    ax.set_ylabel('test accuracy')
    ax.legend()
    plt.savefig('telescope.png')
    plt.show()


def plot_housing():
    k = range(1, 16)
    a1 = [24.6084, 23.1906, 16.5987, 15.4575, 15.8970, 17.2844, 18.9694,
          17.9299, 20.6589, 22.4415, 23.4989, 24.7812, 24.5133, 23.2192, 24.2782]
    a2 = [24.6084, 22.9626, 16.3612, 13.5890, 12.1196, 10.9597, 10.7363, 10.8165,
          11.3451, 11.5925, 11.5515, 11.7144, 11.6558, 11.6700, 11.8807]

    print(np.argmin(a1), np.argmin(a2))
    fig, ax = plt.subplots()
    ax.plot(k, a1, marker='v', label='no weights')
    ax.plot(k, a2, marker='o', label='inverse distance')
    ax.set_xlabel('number of neighbors')
    ax.set_ylabel('test MSE')
    ax.legend()
    plt.savefig('housing.png')
    plt.show()

def test():
    y = np.array([[1, 1.0], [1, 0], [0, 0.5], [0, -0.2]])
    output = np.array([[0, 0.6], [1, -0.3], [1, 1.2], [0, 0]])
    print(np.sum(np.mean(np.square(y - output), axis=0)))

def normalize(array, min, max):

    return (array - min)/(max - min)



if __name__ == '__main__':
    credit_a()
    # y = np.array([[0], [0], [1], [0], [1], [1], [1], [1]])
    # label_values = set([0, 1])
    # value_cnt = {}
    # column = [0, 1, 0, 2, 2, 0, 2, 2]
    # for j in range(len(column)):
    #     value = column[j]
    #     if not np.isnan(value):
    #         if value not in value_cnt:
    #             value_cnt[value] = {lv: 0 for lv in label_values}
    #             value_cnt[value]['total'] = 1
    #             value_cnt[value][y[j][0]] = 1
    #         else:
    #             value_cnt[value]['total'] += 1
    #             value_cnt[value][y[j][0]] += 1
    # probs = {value: {lv: value_cnt[value][lv] / value_cnt[value]['total'] for lv in label_values} for value in
    #          value_cnt}
    # vdm = {}
    # for v1 in probs:
    #     for v2 in probs:
    #         if v2 != v1:
    #             vdm[(v1, v2)] = 0
    #             for lv in label_values:
    #                 vdm[(v1, v2)] += np.square(probs[v1][lv] - probs[v2][lv])
    # print(vdm)
