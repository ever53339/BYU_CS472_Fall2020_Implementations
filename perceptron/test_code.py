import numpy as np
from tools.arff import Arff
from perceptron import PerceptronClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

def debug_code():

    mat = Arff('../data/perceptron/debug/linsep2nonorigin.arff')
    np_mat = mat.data
    data = mat[:,:-1]
    labels = mat[:,-1].reshape(-1,1)

    clf = Perceptron(tol=1e-3, eta0=0.1)
    clf.fit(data, labels)
    print(clf.score(data, labels))
    print(clf.coef_)
    print(clf.intercept_)

    # print(labels)
    P2Class = PerceptronClassifier(lr=0.1,shuffle=False, deterministic=10)
    # training_x, training_y, test_x, test_y = P2Class.split_data(data, labels, training_proportion=1)
    # print(training_x)
    # print(training_y)
    # print(test_x)
    # print(test_y)
    P2Class.fit(data, labels, initial_weights=np.array([0, 0, 0]))
    Accuracy = P2Class.score(data,labels)
    print("Accuray = [{:.2f}]".format(Accuracy))
    print("Final Weights =",P2Class.get_weights())

    clf = Perceptron(tol=1e-3)
    clf.fit(data, labels)
    print(clf.score(data, labels))
    print(clf.coef_)
    print(clf.intercept_)

    # print(P2Class.weights)
    #
    # print(P2Class.predict(data))
    # print(labels)
    # print(P2Class.score(data, labels))

def vote_code():
    mat = Arff('../data/perceptron/vote.arff')
    data = mat[:, :-1]
    print(data.shape)
    labels = mat[:, -1].reshape(-1, 1)

    P2Class = PerceptronClassifier(lr=0.1, shuffle=True)
    clf = Perceptron(tol=1e-3)
    parameters = [{} for i in range(5)]
    accuracy_variation = []
    sl_p  = [{} for i in range(5)]
    for i in range(5):
        training_X, training_y, test_X, test_y = P2Class.split_data(data, labels, training_proportion=0.7)
        # print(training_X.shape, training_y.shape, test_X.shape, test_y.shape)

        P2Class.fit(training_X, training_y)

        parameters[i]['training_accuracy'] = P2Class.score(training_X, training_y)
        parameters[i]['test_accuracy'] = P2Class.score(test_X, test_y)
        parameters[i]['num_epochs'] = P2Class.epoch
        parameters[i]['final weights'] = P2Class.get_weights()
        accuracy_variation.append(P2Class.accuracy_variation)

        clf.fit(training_X, training_y)
        sl_p[i]['training_accuracy'] = clf.score(training_X, training_y)
        sl_p[i]['test_accuracy'] = clf.score(test_X, test_y)
        sl_p[i]['final weights'] = np.append(clf.coef_, clf.intercept_)
    print(parameters)
    print(sl_p)
    avg_parameters = {}
    for key in parameters[0]:
        total = parameters[0][key]
        for i in range(1, 5):
            total += parameters[i][key]

        avg_parameters[key] = total/5
    print(avg_parameters)

    max_len = 0
    for ac in accuracy_variation:
        if len(ac) > max_len:
            max_len = len(ac)
    # print(max_len)

    avg_accuracy_variation = np.zeros(max_len)

    for i in range(max_len):
        cnt = 0
        for ac in accuracy_variation:
            if len(ac) > i:
                avg_accuracy_variation[i] += ac[i]
                cnt += 1
        avg_accuracy_variation[i] /= cnt

    print(avg_accuracy_variation)
    # print(P2Class.accuracy_variation)

    # clf = Perceptron(tol=1e-3)
    # clf.fit(data, labels)
    # print(clf.score(data, labels))

def iris_code():
    mat = Arff('../data/perceptron/iris.arff')
    whole_data = mat[:, :-1]
    print(whole_data.shape)
    whole_raw_labels = mat[:, -1].reshape(-1, 1)
    whole_labels = np.zeros([whole_raw_labels.shape[0], 3])
    for i in np.arange(whole_raw_labels.shape[0]):
        if whole_raw_labels[i][0] == 0:
            whole_labels[i] = np.array([1.0, 0.0, 0.0])
        elif whole_raw_labels[i][0] == 1.0:
            whole_labels[i] = np.array([0.0, 1.0, 0.0])
        elif whole_raw_labels[i][0] == 2.0:
            whole_labels[i] = np.array([0.0, 0.0, 1.0])
    P2Class_1 = PerceptronClassifier(lr=0.1, shuffle=True)
    data, labels, test_data, test_labels = P2Class_1.split_data(whole_data, whole_labels, training_proportion=0.7)
    print(data.shape, labels.shape, test_data.shape, test_labels.shape)


    def training(data, labels, test_data, test_labels):

        P2Class = PerceptronClassifier(lr=0.02, shuffle=True)
        P2Class.fit(data, labels)
        training_accuracy = P2Class.score(data, labels)
        test_accuracy = P2Class.score(test_data, test_labels)
        print('training_accuracy:{}, test_accuracy{}'.format(training_accuracy, test_accuracy))
        weights = P2Class.get_weights()
        return weights

    w1 = training(data, np.expand_dims(labels[:, 0], axis=1), test_data, np.expand_dims(test_labels[:, 0], axis=1))
    w2 = training(data, np.expand_dims(labels[:, 1], axis=1), test_data, np.expand_dims(test_labels[:, 1], axis=1))
    w3 = training(data, np.expand_dims(labels[:, 2], axis=1), test_data, np.expand_dims(test_labels[:, 2], axis=1))

    w = np.array([w1, w2, w3])

    def compute_net(weights, data):
        net = np.zeros([data.shape[0], weights.shape[0]])
        data_with_bias = np.append(data, np.ones([data.shape[0], 1]), axis=1)
        for i in np.arange(net.shape[0]):
            for j in np.arange((net.shape[1])):
                net[i][j] = np.dot(weights[j], data_with_bias[i])
        return net

    training_net = compute_net(w, data)
    test_net = compute_net(w, test_data)
    print(training_net.shape, test_net.shape)

    def net_to_output(net):
        output = np.zeros(net.shape)
        for i in np.arange(net.shape[0]):
            output[i][net[i].argmax()] = 1.0
            # if net[i].argmax() == 0:
            #     output[i] = np.array([1.0, 0.0, 0.0])
            # elif net[i]. argmax() == 1:
            #     output[i] = np.array([0.0, 1.0, 0.0])
            # elif net[i].argmax() == 2:
            #     output[i] = np.array([0.0, 0.0, 1.0])
        return output

    training_output = net_to_output(training_net)
    test_output = net_to_output(test_net)

    print(training_output.shape, test_output.shape)

    def build_confusion_matrix(output, labels):
        matrix = np.zeros([labels.shape[1], labels.shape[1]]).astype(int)

        for i in np.arange(labels.shape[0]):
            j = labels[i].argmax()
            k = output[i].argmax()
            matrix[j][k] += 1

        return matrix

    training_matrix = build_confusion_matrix(training_output, labels)
    test_matrix = build_confusion_matrix(test_output, test_labels)

    print(training_matrix)
    print(test_matrix)
    # print(training_net)
    # print(test_net)
    # print(training_net.shape)
    # print(test_net.shape)
    # P2Class_1 = PerceptronClassifier(lr=0.1, shuffle=True)
    # data, labels, test_data, test_labels = P2Class_1.split_data(whole_data, whole_labels, training_proportion=0.7)
    # print(data.shape, labels.shape, test_data.shape, test_labels.shape)
    # P2Class_1 = PerceptronClassifier(lr=0.1, shuffle=True)
    # P2Class_1.fit(data, np.expand_dims(labels[:,0], axis=1))
    # print(P2Class_1.score(data, np.expand_dims(labels[:,0], axis=1)))
    # print()
    # print(np.expand_dims(labels[:, 1], axis=1))
    # P2Class_2 = PerceptronClassifier(lr=0.1, shuffle=True)
    # P2Class_2.fit(data, np.expand_dims(labels[:, 1], axis=1))
    # print(P2Class_2.score(data, np.expand_dims(labels[:, 1], axis=1)))
    #
    # P2Class_3 = PerceptronClassifier(lr=0.1, shuffle=True)
    # P2Class_3.fit(data, np.expand_dims(labels[:, 2], axis=1))
    # print(P2Class_3.score(data, np.expand_dims(labels[:, 2], axis=1)))


    # clf_1 = Perceptron(tol=1e-3)
    # clf_1.fit(data, np.expand_dims(labels[:,0], axis=1))
    # print(clf_1.score(data, np.expand_dims(labels[:,0], axis=1)))
    #
    # clf_2 = Perceptron(tol=1e-3)
    # clf_2.fit(data, np.expand_dims(labels[:, 1], axis=1))
    # print(clf_2.score(data, np.expand_dims(labels[:, 1], axis=1)))
    #
    # clf_3 = Perceptron(tol=1e-3)
    # clf_3.fit(data, np.expand_dims(labels[:, 2], axis=1))
    # print(clf_3.score(data, np.expand_dims(labels[:, 2], axis=1)))

def iris_sklearn():
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

    # P2Class_1 = PerceptronClassifier(lr=0.1, shuffle=True)
    # data, labels, test_data, test_labels = P2Class_1.split_data(whole_data, whole_labels, training_proportion=0.7)
    # clf = Perceptron(tol=1e-3)
    data, test_data, labels, test_labels = train_test_split(whole_data, whole_labels, test_size=0.3)

    def training(data, labels, test_data, test_labels):

        clf = Perceptron(tol=1e-3)
        clf.fit(data, labels)
        training_accuracy = clf.score(data, labels)
        test_accuracy = clf.score(test_data, test_labels)
        print('training_accuracy:{}, test_accuracy{}'.format(training_accuracy, test_accuracy))
        weights = np.append(clf.coef_, clf.intercept_)
        return weights



    w1 = training(data, np.expand_dims(labels[:, 0], axis=1), test_data, np.expand_dims(test_labels[:, 0], axis=1))
    w2 = training(data, np.expand_dims(labels[:, 1], axis=1), test_data, np.expand_dims(test_labels[:, 1], axis=1))
    w3 = training(data, np.expand_dims(labels[:, 2], axis=1), test_data, np.expand_dims(test_labels[:, 2], axis=1))

    w = np.array([w1, w2, w3])

    def compute_net(weights, data):
        net = np.zeros([data.shape[0], weights.shape[0]])
        data_with_bias = np.append(data, np.ones([data.shape[0], 1]), axis=1)
        for i in np.arange(net.shape[0]):
            for j in np.arange((net.shape[1])):
                net[i][j] = np.dot(weights[j], data_with_bias[i])
        return net

    training_net = compute_net(w, data)
    test_net = compute_net(w, test_data)


    def net_to_output(net):
        output = np.zeros(net.shape)
        for i in np.arange(net.shape[0]):
            output[i][net[i].argmax()] = 1.0
            # if net[i].argmax() == 0:
            #     output[i] = np.array([1.0, 0.0, 0.0])
            # elif net[i]. argmax() == 1:
            #     output[i] = np.array([0.0, 1.0, 0.0])
            # elif net[i].argmax() == 2:
            #     output[i] = np.array([0.0, 0.0, 1.0])
        return output

    training_output = net_to_output(training_net)
    test_output = net_to_output(test_net)

    def build_confusion_matrix(output, labels):
        matrix = np.ones([labels.shape[1], labels.shape[1]]).astype(int)

        for i in np.arange(labels.shape[0]):
            j = labels[i].argmax()
            k = output[i].argmax()
            matrix[j][k] += 1

        return matrix

    training_matrix = build_confusion_matrix(training_output, labels)
    test_matrix = build_confusion_matrix(test_output, test_labels)

    print(training_matrix)
    print(test_matrix)

def vote_sklearn():
    mat = Arff('../data/perceptron/vote.arff')
    data = mat[:, :-1]
    labels = mat[:, -1].reshape(-1, 1)
    clf = Perceptron(tol=1e-3)
    print(data)
    print(labels)

    parameters = [{} for i in range(5)]
    # accuracy_variation = []
    for i in range(5):
        training_X, test_X, training_y, test_y = train_test_split(data, labels, test_size=0.3)

        clf.fit(training_X, training_y)

        parameters[i]['training_accuracy'] = clf.score(training_X, training_y)
        parameters[i]['test_accuracy'] = clf.score(test_X, test_y)
        parameters[i]['num_epochs'] = clf.n_iter_
        parameters[i]['final weights'] = np.append(clf.coef_, clf.intercept_)
        # accuracy_variation.append(P2Class.accuracy_variation)
    print(parameters)
    sl_avg_parameters = {}
    for key in parameters[0]:
        total = parameters[0][key]
        for i in range(1, 5):
            total += parameters[i][key]

        sl_avg_parameters[key] = total / 5
    print(sl_avg_parameters)


if __name__ == '__main__':
    iris_code()
