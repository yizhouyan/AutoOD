import numpy as np
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
# from k_means_constrained import KMeansConstrained
# from imblearn.under_sampling import RepeatedEditedNearestNeighbours, AllKNN
# from imblearn.combine import SMOTEENN, SMOTETomek
from utils import time_this
from multiprocessing import Pool
from functools import partial


class SVMMP:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    inlier_index: np.ndarray
    outlier_index: np.ndarray
    method: str
    model: list
    # kmeans_model: type(KMeans) or type(KMeansConstrained)

    def __init__(self, n_partition=2):
        self.n_partition = n_partition

    # @time_this
    def _partial_fit(self, partition: np.ndarray, cluster_label=None, preprocessing=None):
        index = np.concatenate([partition, self.outlier_index])
        if preprocessing is None:
            svc = SVC(gamma='auto', probability=True, random_state=0)
            # print(self.X_train[index].shape, self.y_train[index].shape,
            #       partition.shape, self.outlier_index.shape)
            svc.fit(self.X_train[index], self.y_train[index])
            if isinstance(cluster_label, int):
                return cluster_label, svc
            return svc
        # if preprocessing=='AllKNN':
        #     model = AllKNN()
        # elif preprocessing=='RENN':
        #     model = RepeatedEditedNearestNeighbours()
        # elif preprocessing== 'SMOTEENN':
        #     model = SMOTEENN()
        # elif preprocessing=='SMOTETomek':
        #     model = SMOTETomek()
        # else:
        #     raise ValueError
        X_resampled, y_resampled = model.fit_resample(self.X_train[index], self.y_train[index])
        print(self.X_train[index].shape, self.y_train[index].shape, X_resampled.shape, y_resampled.shape)
        svc = SVC(gamma='auto', probability=True, random_state=0)
        svc.fit(X_resampled, y_resampled)
        return svc

    # @time_this
    def _partial_predict(self, pid: int):
        return self.model[pid].predict(self.X_test)

    # @time_this
    def _partial_predict_proba(self, pid: int):
        return self.model[pid].predict_proba(self.X_test)[:, 1]

    @time_this
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, method='random', preprocessing=None, set_partition=False):
        assert method in ['random', 'kmeans', 'kmeans-constrained']
        self.X_train = X_train
        self.y_train = y_train
        self.method = method
        counts = np.bincount(self.y_train)
        self.inlier_index = np.where(y_train == np.argmax(counts))[0]
        self.outlier_index = np.where(y_train == np.argmin(counts))[0]
        print(f'outlier({np.argmin(counts)}) count: {len(self.outlier_index)}, inlier count: {len(self.inlier_index)}')
        if set_partition:
            self.n_partition = np.ceil(len(self.inlier_index) / len(self.outlier_index)).astype(np.int64)
            self.n_partition = min(self.n_partition, 12)
        print(f'n_partition: {self.n_partition}')

        if method=='random':
            args = [partition for partition in np.array_split(self.inlier_index, self.n_partition, axis=0)]
            with Pool(self.n_partition) as pool:
                self.model = pool.map(partial(self._partial_fit, preprocessing=preprocessing), args)
        elif method=='kmeans':
            kmeans = KMeans(n_clusters=self.n_partition, random_state=0)
            kmeans.fit(X_train[self.inlier_index])
            self.kmeans_model = kmeans
            cluster_labels, counts = np.unique(kmeans.labels_, return_counts=True)
            args = [(self.inlier_index[np.where(kmeans.labels_ == cluster_label)], i)
                    for i, cluster_label in enumerate(cluster_labels)]
            with Pool(self.n_partition) as pool:
                self.model = pool.starmap(partial(self._partial_fit, preprocessing=preprocessing), args)
                self.model.sort(key=lambda tup: tup[0])
        # elif method=='kmeans-constrained':
        #     size_min = int(X_train[self.inlier_index].shape[0] / self.n_partition * 0.95)
        #     kmeans = KMeansConstrained(n_clusters=self.n_partition, random_state=0,
        #                                size_min=size_min, n_jobs=self.n_partition)
        #     kmeans.fit(X_train[self.inlier_index])
        #     self.kmeans_model = kmeans
        #     cluster_labels, counts = np.unique(kmeans.labels_, return_counts=True)
        #     args = [(self.inlier_index[np.where(kmeans.labels_ == cluster_label)], i)
        #             for i, cluster_label in enumerate(cluster_labels)]
        #     pool = Pool(self.n_partition)
        #     self.model = pool.starmap(partial(self._partial_fit, preprocessing=preprocessing), args)
        #     self.model.sort(key=lambda tup: tup[0])

    @time_this
    def predict(self, X_test: np.ndarray, ensemble_method='Aggregate'):
        self.X_test = X_test
        if self.method in ['kmeans']:
            cluster_label = self.kmeans_model.predict(X_test)
            svm_model = [self.model[i][1] for i in cluster_label]
            predictions = [j.predict(X_test[i].reshape(1, -1)) for i, j in enumerate(svm_model)]
            return np.array(predictions)
        # if self.method == 'kmeans-constrained':
        #     cluster_label = self.kmeans_model.predict(X_test, size_min=None)
        #     svm_model = [self.model[i][1] for i in cluster_label]
        #     predictions = [j.predict(X_test[i].reshape(1, -1)) for i, j in enumerate(svm_model)]
        #     return np.array(predictions)

        assert ensemble_method in ['Mean Proba', 'Aggregate']
        with Pool(self.n_partition) as pool:
            if ensemble_method == 'Mean Proba':
                predictions = np.array(pool.map(self._partial_predict_proba, [i for i in range(self.n_partition)])).T
                predictions = np.mean(predictions, axis=1)
                predictions = [i for i in predictions > 0.5]
                return np.array(predictions)
            if ensemble_method == 'Aggregate':
                predictions = np.array(pool.map(self._partial_predict, [i for i in range(self.n_partition)])).T
                predictions = np.mean(predictions, axis=1)
                predictions = [i for i in predictions > 0.5]
                return np.array(predictions)
        return None

    def predict_proba(self, X_test:np.ndarray):
        self.X_test = X_test
        with Pool(self.n_partition) as pool:
            predictions = np.array(pool.map(self._partial_predict_proba, [i for i in range(self.n_partition)])).T
            predictions = np.mean(predictions, axis=1)
            return np.array(predictions)



    def _get_boundary(self):
        lr = LogisticRegression(penalty='l2', solver='liblinear', random_state=0)
        lr.fit(self.X_train, self.y_train)
        w, b = lr.coef_, lr.intercept_

