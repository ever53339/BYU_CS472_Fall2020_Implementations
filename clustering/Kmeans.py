import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

class KMEANSClustering(BaseEstimator,ClusterMixin):

    def __init__(self, k=3, debug=False): ## add parameters here
        """
        Args:
            k = how many final clusters to have
            debug = if debug is true use the first k instances as the initial centroids otherwise choose random points as the initial centroids.
        """
        self.k = k
        self.debug = debug

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

        rerun = True
        while rerun:
            if self.debug:
                self.clusters = [[X[i]] for i in range(self.k)]
                # idxes = np.array([i if i < self.k else None for i in range(num_instances)])
            else:
                centroid_idxes = np.random.choice(num_instances, self.k, replace=False)
                self.clusters = [[X[i]] for i in centroid_idxes]
                # idxes = np.array([centroid_idxes.tolist().index(i) if i in centroid_idxes else None for i in range(num_instances)])

            self.centroids = [np.mean(c, axis=0) for c in self.clusters]

            converge = False
            while not converge:
                converge = self._update_clustering(X)

            self.SSEs = [np.square(self.clusters[i] - self.centroids[i]).sum() for i in range(len(self.clusters))]
            self.total_SSE = np.sum(self.SSEs)

            if [] not in self.clusters:
                rerun = False

        return self

    def _update_clustering(self, X):
        converge = False

        distance_matrix = np.stack([np.square(X - self.centroids[i]).sum(axis=1) for i in range(len(self.centroids))])
        new_idxes = np.argmin(distance_matrix, axis=0)

        new_clusters = [[] for i in range(self.k)]
        for i in range(X.shape[0]):
            new_clusters[new_idxes[i]].append(X[i])

        self.clusters = new_clusters
        new_centroids = [np.mean(c, axis=0) for c in self.clusters]

        if (np.stack(new_centroids) == np.stack(self.centroids)).all():
            converge = True
        self.centroids = new_centroids

        return converge

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
