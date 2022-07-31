import pandas as pd
from scipy.io import arff
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
import multiprocessing
import numpy as np
import scipy as sp
from utils import time_this
from sklearn import metrics
from snorkel.labeling import MajorityLabelVoter
from snorkel.labeling import LabelModel
from SVMMP import SVMMP
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler


def load_dataset(filename):
    with open(filename, 'r') as f:
        data, meta = arff.loadarff(f)
    data = pd.DataFrame(data)
    X = data.drop(columns=['id', 'outlier'])
    # Map dataframe to encode values and put values into a numpy array
    y = data["outlier"].map(lambda x: 1 if x == b'yes' else 0).values
    return X, y


def run_knn(X, y, k=60):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)
    knn_dists = neigh.kneighbors(X)[0][:, -1]
    return k, knn_dists


def run_lof(X, y, k=60):
    clf = LocalOutlierFactor(n_neighbors=k)
    clf.fit(X)
    lof_scores = -clf.negative_outlier_factor_
    return k, lof_scores


def run_isolation_forest(X, y, max_features=1.0):
    # training the model
    clf = IsolationForest(random_state=42, max_features=max_features)
    clf.fit(X)
    # predictions
    sklearn_score_anomalies = clf.decision_function(X)
    if_scores = [-1 * s + 0.5 for s in sklearn_score_anomalies]
    return max_features, if_scores


def mahalanobis(x):
    """Compute the Mahalanobis Distance between each row of x and the data
    """
    x_minus_mu = x - np.mean(x)
    cov = np.cov(x.T)
    inv_covmat = sp.linalg.inv(cov)
    results = []
    x_minus_mu = np.array(x_minus_mu)
    for i in range(np.shape(x)[0]):
        cur_data = x_minus_mu[i, :]
        results.append(np.dot(np.dot(x_minus_mu[i, :], inv_covmat), x_minus_mu[i, :].T))
    return np.array(results)


def run_mahalanobis(X, y):
    # training the model
    dist = mahalanobis(x=X)
    return dist


def get_predictions_scores(scores, num_outliers=400, method_name='LOF'):
    threshold = np.sort(scores)[::-1][num_outliers]
    # threshold, max_f1 = get_best_f1_score(y, lof_scores)
    predictions = np.array(scores > threshold)
    predictions = np.array([int(i) for i in predictions])
    #     print('F1 for {} : {}'.format(method_name, metrics.f1_score(y, predictions)))
    return predictions, scores


all_results = []
all_scores = []


@time_this
def majority_vote(L, y):
    majority_model = MajorityLabelVoter()
    preds_train = majority_model.predict_proba(L=L)
    threshold = 0.5
    predictions = np.full((len(y)), 0)
    predictions[preds_train[:, 1] > threshold] = 1


# f1_score = metrics.f1_score(y, predictions)
# print(f"F1 for MV:{f1_score}")
# return f1_score

@time_this
def snorkel(L, y):
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L, seed=123, lr=0.001, n_epochs=1000)
    probs_train = label_model.predict_proba(L=L)
    threshold = 0.5
    predictions = np.full((len(y)), 0)
    predictions[probs_train[:, 1] > threshold] = 1


# f1_score = metrics.f1_score(y, predictions)
# print(f"F1 for MV:{f1_score}")
# return f1_score


@time_this
def autood_preprocessing(X, y, lof_krange, knn_krange, if_range, mahalanobis_N_range, if_N_range, N_range):
    all_results = []
    all_scores = []

    with multiprocessing.Pool(8) as pool:
        temp_lof_results = dict()
        unique_lof_ks = list(set(lof_krange))

        processes = [pool.apply_async(run_lof, args=(X, y, k,)) for k in unique_lof_ks]
        for p in processes:
            k, lof_scores = p.get()
            temp_lof_results[k] = lof_scores

        for i in range(len(lof_krange)):
            lof_predictions, lof_scores = get_predictions_scores(temp_lof_results[lof_krange[i]],
                                                                 num_outliers=N_range[i], method_name='LOF')
            all_results.append(lof_predictions)
            all_scores.append(lof_scores)

        temp_knn_results = dict()
        unique_knn_ks = list(set(knn_krange))

        processes = [pool.apply_async(run_knn, args=(X, y, k,)) for k in unique_knn_ks]
        for p in processes:
            k, knn_scores = p.get()
            temp_knn_results[k] = knn_scores

        for i in range(len(knn_krange)):
            knn_predictions, knn_scores = get_predictions_scores(temp_knn_results[knn_krange[i]],
                                                                 num_outliers=N_range[i], method_name='KNN')
            all_results.append(knn_predictions)
            all_scores.append(knn_scores)

        temp_if_results = dict()
        unique_if_features = list(set(if_range))

        processes = [pool.apply_async(run_isolation_forest, args=(X, y, k,)) for k in unique_if_features]
        for p in processes:
            k, if_scores = p.get()
            temp_if_results[k] = if_scores

        for i in range(len(if_range)):
            if_predictions, if_scores = get_predictions_scores(temp_if_results[if_range[i]], num_outliers=if_N_range[i],
                                                               method_name='IF')
            all_results.append(if_predictions)
            all_scores.append(if_scores)

        mahalanobis_scores = run_mahalanobis(X, y)
        for i in range(len(mahalanobis_N_range)):
            mahalanobis_predictions, mahalanobis_scores = get_predictions_scores(mahalanobis_scores,
                                                                                 num_outliers=mahalanobis_N_range[i],
                                                                                 method_name='mahala')
            all_results.append(mahalanobis_predictions)
            all_scores.append(mahalanobis_scores)

        L = np.stack(all_results).T
        scores = np.stack(all_scores).T
        return L, scores


@time_this
def autood(L, scores, X, y, max_iteration=10):
    # prepare scores for training
    prediction_result_list = []
    classifier_result_list = []
    prediction_list = []
    cur_f1_scores = []
    prediction_high_conf_outliers = np.array([])
    prediction_high_conf_inliers = np.array([])
    prediction_classifier_disagree = np.array([])
    index_range = np.array([[0, 60], [60, 120], [120, 150], [150, 156]])
    coef_index_range = np.array([[0, 10], [10, 20], [20, 25], [25, 26]])
    coef_remain_index = range(156)
    scores_for_training_indexes = []
    for i in range(len(index_range)):
        start = index_range[i][0]
        temp_range = coef_index_range[i][1] - coef_index_range[i][0]
        scores_for_training_indexes = scores_for_training_indexes + list(range(start, start + temp_range))
    scores_for_training = scores[:, np.array(scores_for_training_indexes)]

    # first iteration
    high_confidence_threshold = 0.99
    low_confidence_threshold = 0.01
    max_iter = 200
    remain_params_tracking = np.array(range(0, np.max(coef_index_range)))
    training_data_F1 = []
    two_prediction_corr = []

    min_max_diff = []
    N_size = 6

    last_training_data_indexes = []
    counter = 0

    for i_range in range(0, max_iteration):
        print("##################################################################")
        print('Iteration = {}, L shape = {}'.format(i_range, np.shape(L)))
        num_methods = np.shape(L)[1]

        agree_outlier_indexes = np.sum(L, axis=1) == np.shape(L)[1]
        # print('All agree, Number of outliers = {}'.format(sum(agree_outlier_indexes)))
        agree_inlier_indexes = np.sum(L, axis=1) == 0
        # print('All agree, Number of inliers = {}'.format(sum(agree_inlier_indexes)))

        disagree_indexes = np.where(np.logical_or(np.sum(L, axis=1) == 0, np.sum(L, axis=1) == num_methods) == 0)[0]

        all_inlier_indexes = np.setdiff1d(np.where(agree_inlier_indexes)[0], prediction_high_conf_outliers)
        if len(prediction_high_conf_inliers) > 0:
            all_inlier_indexes = np.intersect1d(
                np.setdiff1d(np.where(agree_inlier_indexes)[0], prediction_high_conf_outliers),
                prediction_high_conf_inliers)
        # print('num of inliers = {}'.format(np.shape(all_inlier_indexes)[0]))

        all_outlier_indexes = np.union1d(np.where(agree_outlier_indexes)[0], prediction_high_conf_outliers)
        # print('num of outliers = {}'.format(np.shape(all_outlier_indexes)[0]))
        all_inlier_indexes = np.setdiff1d(all_inlier_indexes, prediction_classifier_disagree)

        self_agree_index_list = []
        if ((len(all_outlier_indexes) == 0) or (len(all_inlier_indexes) / len(all_outlier_indexes) > 1000)):
            for i in range(0, len(index_range)):
                if (index_range[i, 1] - index_range[i, 0] <= 6):
                    continue
                temp_index = disagree_indexes[np.where(
                    np.sum(L[disagree_indexes][:, index_range[i, 0]: index_range[i, 1]], axis=1) == (
                                index_range[i, 1] - index_range[i, 0]))[0]]
                self_agree_index_list = np.union1d(self_agree_index_list, temp_index)
            self_agree_index_list = [int(i) for i in self_agree_index_list]
        #     self_agree_index_list = np.random.RandomState(1).permutation(self_agree_index_list)[:500]
        all_outlier_indexes = np.union1d(all_outlier_indexes, self_agree_index_list)
        all_outlier_indexes = np.setdiff1d(all_outlier_indexes, prediction_classifier_disagree)
        # print('num of outliers = {}'.format(np.shape(all_outlier_indexes)[0]))

        data_indexes = np.concatenate((all_inlier_indexes, all_outlier_indexes), axis=0)
        data_indexes = np.array([int(i) for i in data_indexes])
        labels = np.concatenate((np.zeros(len(all_inlier_indexes)), np.ones(len(all_outlier_indexes))), axis=0)
        transformer = RobustScaler().fit(scores_for_training)
        scores_transformed = transformer.transform(scores_for_training)
        training_data = scores_transformed[data_indexes]
        # print('Training data shape: ', np.shape(training_data))
        # training_data_F1.append(metrics.f1_score(y[data_indexes], labels))
        # print('Training data F-1', metrics.f1_score(y[data_indexes], labels))

        clf = LogisticRegression(random_state=0, penalty='l2', max_iter=max_iter).fit(training_data, labels)
        clf_predictions = clf.predict(scores_transformed)
        clf_predict_proba = clf.predict_proba(scores_transformed)[:, 1]
        # print("F-1 score from LR:",metrics.f1_score(y, np.array([int(i) for i in clf_predict_proba > 0.5])))

        transformer = RobustScaler().fit(X)
        X_transformed = transformer.transform(X)
        X_training_data = X_transformed[data_indexes]

        clf_X = SVC(gamma='auto', probability=True, random_state=0)
        clf_X.fit(X_training_data, labels)
        clf_predictions_X = clf_X.predict(X_transformed)
        clf_predict_proba_X = clf_X.predict_proba(X_transformed)[:,1]

        # clf_X = SVMMP(n_partition=4)
        # clf_X.fit(X_training_data, labels.astype(int), set_partition=False)
        # clf_predictions_X = clf_X.predict(X, 'Mean Proba')
        # clf_predict_proba_X = clf_X.predict_proba(X_transformed)
        #
        # print("F-1 score from SVM:",metrics.f1_score(y, np.array([int(i) for i in clf_predict_proba_X > 0.5])))

        agreed_outlier_indexes = np.where(np.sum(L, axis=1) == np.shape(L)[1])[0]
        agreed_inlier_indexes = np.where(np.sum(L, axis=1) == 0)[0]

        prediction_result_list.append(clf_predict_proba)
        classifier_result_list.append(clf_predict_proba_X)

        prediction_list.append(np.array([int(i) for i in clf_predictions]))

        prediction_high_conf_outliers = np.intersect1d(
            np.where(prediction_result_list[-1] > high_confidence_threshold)[0],
            np.where(classifier_result_list[-1] > high_confidence_threshold)[0])
        # print('length of prediction_high_conf_outliers:' , len(prediction_high_conf_outliers))
        prediction_high_conf_inliers = np.intersect1d(
            np.where(prediction_result_list[-1] < low_confidence_threshold)[0],
            np.where(classifier_result_list[-1] < low_confidence_threshold)[0])
        # print('length of prediction high conf inliers: ', len(prediction_high_conf_inliers))

        temp_prediction = np.array([int(i) for i in prediction_result_list[-1] > 0.5])
        temp_classifier = np.array([int(i) for i in classifier_result_list[-1] > 0.5])
        prediction_classifier_disagree = np.where(temp_prediction != temp_classifier)[0]

        two_prediction_corr.append(np.corrcoef(clf_predict_proba, clf_predict_proba_X)[0, 1])

        if np.max(coef_index_range) >= 2:
            if (len(prediction_high_conf_outliers) > 0 and len(prediction_high_conf_inliers) > 0):
                new_data_indexes = np.concatenate((prediction_high_conf_outliers, prediction_high_conf_inliers), axis=0)
                new_data_indexes = np.array([int(i) for i in new_data_indexes])
                new_labels = np.concatenate(
                    (np.ones(len(prediction_high_conf_outliers)), np.zeros(len(prediction_high_conf_inliers))), axis=0)
                clf_prune_2 = LogisticRegression(random_state=0, penalty='l2', max_iter=max_iter).fit(
                    scores_transformed[new_data_indexes], new_labels)
                combined_coef = clf_prune_2.coef_[0]
            else:
                combined_coef = clf.coef_[0]

            if (np.max(coef_index_range) >= 2):
                if (len(set(combined_coef)) > 1):
                    cur_clf_coef = combined_coef
                    cutoff = max(max(0, np.mean(combined_coef) - np.std(combined_coef)), min(combined_coef))

                    remain_indexes_after_cond = (
                                cur_clf_coef > cutoff)  # np.logical_and(cur_clf_coef > cutoff, abs(cur_clf_coef) > 0.01) # #
                    remain_params_tracking = remain_params_tracking[remain_indexes_after_cond]

                    remain_indexes_after_cond_expanded = []
                    for i in range(0, len(coef_index_range)):  #
                        s_e_range = coef_index_range[i, 1] - coef_index_range[i, 0]
                        s1, e1 = coef_index_range[i, 0], coef_index_range[i, 1]
                        s2, e2 = index_range[i, 0], index_range[i, 1]
                        saved_indexes = np.where(cur_clf_coef[s1:e1] > cutoff)[0]
                        for j in range(N_size):
                            remain_indexes_after_cond_expanded.extend(np.array(saved_indexes) + j * s_e_range + s2)

                    new_coef_index_range_seq = []
                    for i in range(0, len(coef_index_range)):  #
                        s, e = coef_index_range[i, 0], coef_index_range[i, 1]
                        new_coef_index_range_seq.append(sum((remain_indexes_after_cond)[s:e]))

                    coef_index_range = []
                    index_range = []
                    cur_sum = 0
                    for i in range(0, len(new_coef_index_range_seq)):
                        coef_index_range.append([cur_sum, cur_sum + new_coef_index_range_seq[i]])
                        index_range.append([cur_sum * 6, 6 * (cur_sum + new_coef_index_range_seq[i])])
                        cur_sum += new_coef_index_range_seq[i]

                    coef_index_range = np.array(coef_index_range)
                    index_range = np.array(index_range)

                    L = L[:, remain_indexes_after_cond_expanded]
                    scores_for_training = scores_for_training[:, remain_indexes_after_cond]
        if ((len(last_training_data_indexes) == len(data_indexes)) and
                (sum(last_training_data_indexes == data_indexes) == len(data_indexes)) and
                (np.max(coef_index_range) < 2)):
            counter = counter + 1
        else:
            counter = 0
        if (counter > 3):
            break
        last_training_data_indexes = data_indexes
