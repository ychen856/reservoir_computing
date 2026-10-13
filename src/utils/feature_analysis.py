import numpy as np
from sklearn.decomposition import PCA


def pairwise_feature_distances(X):

    M = X.shape[0]

    D = np.zeros((M, M))

    for i in range(M):
        for j in range(M):

            D[i, j] = np.linalg.norm(
                X[i] - X[j]
            )

    return D

def compute_class_statistics(X, y, M):

    centroids = []
    spreads = []

    for m in range(M):

        X_m = X[y == m]

        centroid = np.mean(
            X_m,
            axis=0
        )

        distances = np.linalg.norm(
            X_m - centroid,
            axis=1
        )

        spread = np.mean(distances)

        centroids.append(centroid)
        spreads.append(spread)

    return (
        np.asarray(centroids),
        np.asarray(spreads)
    )


def centroid_distance_matrix(centroids):

    M = len(centroids)

    D = np.zeros((M, M))

    for i in range(M):
        for j in range(M):

            D[i, j] = np.linalg.norm(
                centroids[i] - centroids[j]
            )

    return D

def project_reservoir_features_pca(X):

    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X)

    explained = pca.explained_variance_ratio_

    return X_pca, pca, explained

