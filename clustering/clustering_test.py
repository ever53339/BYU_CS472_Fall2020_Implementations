import numpy as np
from clustering.HAC import HACClustering, compute_silhouette_score
from clustering.Kmeans import  KMEANSClustering
from tools.arff import Arff
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabaz_score, silhouette_samples
from numpy import genfromtxt
import matplotlib.pyplot as plt


def meta_data_test():
    X = np.array([[1, 1],
                  [2, 1],
                  [3, 2],
                  [4, 5],
                  [5, 5]])

    c = HACClustering(2, link_type='complete')
    c.fit(X)


def debug_hac():
    mat = Arff("../data/clustering/abalone.arff", label_count=0)  ## label_count = 0 because clustering is unsupervised.

    raw_data = mat.data
    data = raw_data



    'with normalization'
    min = np.min(data, axis=0)
    max = np.max(data, axis=0)

    norm_data = normalize(data, min, max)
    c = HACClustering(5, link_type='complete')
    c.fit(norm_data)
    c.save_clusters('debug_hac_complete.txt')

    c = HACClustering(5, link_type='single')
    c.fit(norm_data)
    c.save_clusters('debug_hac_single.txt')

def evaluation():
    mat = Arff("../data/clustering/seismic-bumps_train.arff", label_count=0)  ## label_count = 0 because clustering is unsupervised.

    raw_data = mat.data
    # data = raw_data[:, :-1]
    data = raw_data

    'with normalization'
    min = np.min(data, axis=0)
    max = np.max(data, axis=0)

    norm_data = normalize(data, min, max)
    c = HACClustering(5, link_type='complete')
    c.fit(norm_data)
    c.save_clusters('evaluation_hac_complete.txt')

    c = HACClustering(5, link_type='single')
    c.fit(norm_data)
    c.save_clusters('evaluation_hac_single.txt')

    c = KMEANSClustering(5, debug=True)
    c.fit(norm_data)
    c.save_clusters('evaluation_kmeans.txt')

def iris_hac():
    mat = Arff("../data/clustering/iris.arff", label_count=0)  ## label_count = 0 because clustering is unsupervised.

    raw_data = mat.data
    data = raw_data[:,:-1]

    'with normalization'
    min = np.min(data, axis=0)
    max = np.max(data, axis=0)

    norm_data = normalize(data, min, max)
    for k in range(2, 8):
        c = HACClustering(k, link_type='complete')
        c.fit(norm_data)
        c.save_clusters('nolabel_hac_iris_complete_{}.txt'.format(k))

        c = HACClustering(k, link_type='single')
        c.fit(norm_data)
        c.save_clusters('nolabel_hac_iris_single_{}.txt'.format(k))


def iris_hac_2():
    mat = Arff("../data/clustering/iris.arff", label_count=0)  ## label_count = 0 because clustering is unsupervised.

    raw_data = mat.data
    data = raw_data

    'with normalization'
    min = np.min(data, axis=0)
    max = np.max(data, axis=0)

    norm_data = normalize(data, min, max)
    for k in range(2, 8):
        c = HACClustering(k, link_type='complete')
        c.fit(norm_data)
        c.save_clusters('hac_iris_complete_{}.txt'.format(k))

        c = HACClustering(k, link_type='single')
        c.fit(norm_data)
        c.save_clusters('hac_iris_single_{}.txt'.format(k))


def debug_kmeans():
    mat = Arff("../data/clustering/abalone.arff", label_count=0)  ## label_count = 0 because clustering is unsupervised.

    raw_data = mat.data
    data = raw_data

    'with normalization'
    min = np.min(data, axis=0)
    max = np.max(data, axis=0)

    norm_data = normalize(data, min, max)
    c = KMEANSClustering(5, debug=True)
    c.fit(norm_data)
    c.save_clusters('debug_kmeans.txt')


def iris_kmeans():
    mat = Arff("../data/clustering/iris.arff", label_count=0)  ## label_count = 0 because clustering is unsupervised.

    raw_data = mat.data
    data = raw_data[:, :-1]

    'with normalization'
    min = np.min(data, axis=0)
    max = np.max(data, axis=0)

    norm_data = normalize(data, min, max)
    for k in range(2, 8):
        c = KMEANSClustering(k, debug=False)
        c.fit(norm_data)
        c.save_clusters('nolabel_kmeans_iris_{}.txt'.format(k))


def iris_kmeans_2():
    mat = Arff("../data/clustering/iris.arff", label_count=0)  ## label_count = 0 because clustering is unsupervised.

    raw_data = mat.data
    data = raw_data

    'with normalization'
    min = np.min(data, axis=0)
    max = np.max(data, axis=0)

    norm_data = normalize(data, min, max)
    for k in range(2, 8):
        c = KMEANSClustering(k, debug=False)
        c.fit(norm_data)
        c.save_clusters('kmeans_iris_{}.txt'.format(k))


def iris_kmeans_3():
    mat = Arff("../data/clustering/iris.arff", label_count=0)  ## label_count = 0 because clustering is unsupervised.

    raw_data = mat.data
    data = raw_data

    'with normalization'
    min = np.min(data, axis=0)
    max = np.max(data, axis=0)

    norm_data = normalize(data, min, max)
    for i in range(5):
        c = KMEANSClustering(4, debug=False)
        c.fit(norm_data)
        c.save_clusters('kmeans_iris_4_time_{}.txt'.format(i+1))


def iris_sk_hac_complete():
    mat = Arff("../data/clustering/iris.arff", label_count=0)  ## label_count = 0 because clustering is unsupervised.

    raw_data = mat.data
    data = raw_data

    'with normalization'
    min = np.min(data, axis=0)
    max = np.max(data, axis=0)

    norm_data = normalize(data, min, max)
    sil_score = {}
    ss = {}
    ch_score = {}

    for k in range(2, 8):
        c = AgglomerativeClustering(n_clusters=k, linkage='complete')
        c.fit(norm_data)

        sil_score[k] = silhouette_score(norm_data, c.labels_, metric='euclidean', sample_size=None)
        ss[k] = compute_silhouette_score(norm_data, c.labels_)
        ch_score[k] = calinski_harabaz_score(norm_data, c.labels_)
    print('sk-complete link, sk silhouette score: ', sil_score)
    print('sk-complete link, own silhouette score: ', ss)
    print('sk-complete link, sk calinski_harabaz_score: ', ch_score)


def iris_sk_hac_single():
    mat = Arff("../data/clustering/iris.arff", label_count=0)  ## label_count = 0 because clustering is unsupervised.

    raw_data = mat.data
    data = raw_data

    'with normalization'
    min = np.min(data, axis=0)
    max = np.max(data, axis=0)

    norm_data = normalize(data, min, max)
    sil_score = {}
    ss = {}
    ch_score = {}

    for k in range(2, 8):
        c = AgglomerativeClustering(n_clusters=k, linkage='single')
        c.fit(norm_data)

        sil_score[k] = silhouette_score(norm_data, c.labels_, metric='euclidean', sample_size=None)
        ss[k] = compute_silhouette_score(norm_data, c.labels_)
        ch_score[k] = calinski_harabaz_score(norm_data, c.labels_)
    print('sk-single link, sk silhouette score: ', sil_score)
    print('sk-single link, own silhouette score: ', ss)
    print('sk-single link, sk calinski_harabaz_score: ', ch_score)


def iris_sk_kmeans():
    mat = Arff("../data/clustering/iris.arff", label_count=0)  ## label_count = 0 because clustering is unsupervised.

    raw_data = mat.data
    data = raw_data

    'with normalization'
    min = np.min(data, axis=0)
    max = np.max(data, axis=0)

    norm_data = normalize(data, min, max)
    sil_score = {}
    ss = {}
    ch_score = {}
    for k in range(2, 8):
        c = KMeans(n_clusters=k)
        c.fit(norm_data)

        sil_score[k] = silhouette_score(norm_data, c.labels_, metric='euclidean', sample_size=None)
        ss[k] = compute_silhouette_score(norm_data, c.labels_)
        ch_score[k] = calinski_harabaz_score(norm_data, c.labels_)
    print('sk-kmeans, sk silhouette score: ', sil_score)
    print('sk-kmeans, own silhouette score: ',ss)
    print('sk-kmeans, sk calinski_harabaz_score: ', ch_score)


def other_dataset_sk_kmeans():

    data = genfromtxt('../data/clustering/buddymove_holidayiq.csv', delimiter=',')[1:, 1:]

    'with normalization'
    min = np.min(data, axis=0)
    max = np.max(data, axis=0)

    norm_data = normalize(data, min, max)
    print('\nother dataset sk kmeans results:')
    for init in ['k-means++', 'random']:
        for max_iter in [50, 100, 150]:
            for k in range(2, 8):
                c = KMeans(n_clusters=k, init=init, max_iter=max_iter)
                c.fit(norm_data)

                sil_score = silhouette_score(norm_data, c.labels_, metric='euclidean', sample_size=None)
                ss = compute_silhouette_score(norm_data, c.labels_)

                assert np.abs(sil_score - ss) < 1e-4

                print('init : {}, max_iter: {}, n_clusters: {}, silhouette score: {}'.format(init, max_iter, k, ss))


def other_dataset_sk_hac():
    data = genfromtxt('../data/clustering/buddymove_holidayiq.csv', delimiter=',')[1:, 1:]

    'with normalization'
    min = np.min(data, axis=0)
    max = np.max(data, axis=0)

    norm_data = normalize(data, min, max)
    print('\nother dataset sk hac results:')
    for affinity in ['euclidean', 'manhattan']:
        for linkage in ['complete', 'average', 'single']:
            for k in range(2, 8):
                c = AgglomerativeClustering(n_clusters=k, affinity=affinity, linkage=linkage)
                c.fit(norm_data)

                sil_score = silhouette_score(norm_data, c.labels_, metric='euclidean', sample_size=None)
                ss = compute_silhouette_score(norm_data, c.labels_)

                assert np.abs(sil_score - ss) < 1e-4

                print('affinity : {}, linkage: {}, n_clusters: {}, silhouette score: {}'.format(affinity, linkage, k, ss))


def plot_1():
    x = range(2, 8)
    y1 = [25.7462, 7.1537, 6.2517, 4.7586, 4.1143, 3.6657]

    y2 = [12.1437, 11.9008, 11.0010, 10.8369, 10.5036, 10.3903]

    y3 = [12.1437, 7.1386, 5.5417, 4.6035, 4.7287, 3.7285]

    fig, ax = plt.subplots()
    ax.plot(x, y1, label='hac-complete', marker='o')
    ax.plot(x, y2, label='hac-single', marker='^')
    ax.plot(x, y3, label='kmeans', marker='.', color='r')

    ax.set_xlabel('number of clusters')
    ax.set_ylabel('Total SSE')

    ax.legend()
    fig.savefig('nolabel_iris.png')
    plt.show()


def plot_2():
    x = range(2, 8)
    y1 = [18.3937, 10.8347, 9.9327, 5.4397, 4.3133, 3.7136]

    y2 = [18.3937, 7.8175, 7.5020, 7.2591, 6.7360, 6.5526]

    y3 = [18.3937, 17.4643, 6.6203, 5.1035, 6.1329, 3.8244]

    fig, ax = plt.subplots()
    ax.plot(x, y1, label='hac-complete', marker='o')
    ax.plot(x, y2, label='hac-single', marker='^')
    ax.plot(x, y3, label='kmeans', marker='.', color='r')

    ax.set_xlabel('number of clusters')
    ax.set_ylabel('Total SSE')

    ax.legend()
    fig.savefig('label_iris.png')
    plt.show()


def normalize(array, min, max):

    return (array - min)/(max - min)


if __name__ == '__main__':
    evaluation()
