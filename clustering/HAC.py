import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

class HACClustering(BaseEstimator, ClusterMixin):

    def __init__(self,k=3,link_type='single'): ## add parameters here
        """
        Args:
            k = how many final clusters to have
            link_type = single or complete. when combining two clusters use complete link or single link
        """
        assert link_type in ['single', 'complete']
        self.link_type = link_type
        self.k = k

    def fit(self, X, y=None):
        """ Fit the data; In this lab this will make the K clusters :D
        Args:
            X (array-like): A 2D numpy array with the training data
            y (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        num_instances = X.shape[0]
        assert self.k <= num_instances

        self.clusters = [[i] for i in X]

        while self.k < len(self.clusters):
            self._merge_once()

        self.centroids = [np.mean(c, axis=0) for c in self.clusters]
        self.SSEs = [np.square(self.clusters[i] - self.centroids[i]).sum() for i in range(len(self.clusters))]
        self.total_SSE = np.sum(self.SSEs)

        return self

    def _merge_once(self):
        distances = self._build_distance_matrix()
        idx = np.argmin(distances)
        r = idx // len(self.clusters)
        c = idx % len(self.clusters)
        self.clusters[r].extend(self.clusters.pop(c))

    def _build_distance_matrix(self):
        n = len(self.clusters)
        distances = [[] for i in range(n)]

        for i in range(n):
            for j in range(n):
                if i >= j:
                    distances[i].append(np.inf)
                else:
                    distances[i].append(self._compute_distance(i, j))

        return np.array(distances)

    def _compute_distance(self, i, j):
        final_d = np.inf if self.link_type == 'single' else -np.inf

        for ii in self.clusters[i]:
            for ij in self.clusters[j]:
                d = np.square(ii - ij).sum()

                if self.link_type == 'single':
                    if d < final_d:
                        final_d = d
                else:
                    if d > final_d:
                        final_d = d
        return final_d

    def save_clusters(self, filename):
        """
            f = open(filename,"w+") 
            Used for grading.
            write("{:d}\n".format(k))
            write("{:.4f}\n\n".format(total SSE))
            for each cluster and centroid:
                write(np.array2string(centroid,precision=4,separator=","))
                write("\n")
                write("{:d}\n".format(size of cluster))
                write("{:.4f}\n\n".format(SSE of cluster))
            f.close()
        """
        with open(filename, 'w+') as fh:
            fh.write("{:d}\n".format(self.k))
            fh.write("{:.4f}\n\n".format(self.total_SSE))

            for i in range(len(self.clusters)):
                fh.write(np.array2string(self.centroids[i], precision=4, separator=","))
                fh.write("\n")
                fh.write("{:d}\n".format(len(self.clusters[i])))
                fh.write("{:.4f}\n\n".format(self.SSEs[i]))


def compute_silhouette_score(data, labels):
    d = []

    for i in range(data.shape[0]):
        d.append(np.sqrt(np.sum(np.square(data - data[i]), axis=1)))

    d = np.stack(d)

    s = []
    label_set = set(labels)
    for i in range(data.shape[0]):
        b_values = []

        for l in label_set:
            if l == labels[i]:
                if (l == labels).sum() == 1:
                    a = 0
                else:
                    mean_d = (d[i] * (l == labels)).sum() / ((l == labels).sum() - 1)
                    a = mean_d
            else:
                mean_d = (d[i] * (l == labels)).sum() / (l == labels).sum()
                b_values.append(mean_d)

        b = np.min(b_values)

        if a == 0:
            s.append(0)
        else:
            s.append((b - a) / max(a, b))

    return np.mean(s)
