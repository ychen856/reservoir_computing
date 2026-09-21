import numpy as np
from sklearn.decomposition import PCA

def extract_temporal_voltage_feature(
    sol,
    K,
    t_start,
    t_end
):

    # K uniformly spaced temporal sampling points
    sample_times = np.linspace(
        t_start,
        t_end,
        K
    )

    # Reservoir voltages
    V1 = np.interp(
        sample_times,
        sol.t,
        sol.y[0]
    )

    V2 = np.interp(
        sample_times,
        sol.t,
        sol.y[1]
    )

    # Interleave V1 and V2:
    #
    # [V1(t1), V2(t1),
    #  V1(t2), V2(t2),
    #  ...]
    feature = np.column_stack(
        [V1, V2]
    ).reshape(-1)

    return sample_times, feature


def extract_temporal_voltage_feature(
    sol,
    K,
    t_start,
    t_end
):

    sample_times = np.linspace(
        t_start,
        t_end,
        K
    )

    V1 = np.interp(
        sample_times,
        sol.t,
        sol.y[0]
    )

    V2 = np.interp(
        sample_times,
        sol.t,
        sol.y[1]
    )

    feature = np.column_stack(
        [V1, V2]
    ).reshape(-1)

    return sample_times, feature


def build_mpsk_feature_matrix(
    sols,
    K,
    t_start,
    t_end
):

    features = []

    sample_times = None

    for sol in sols:

        sample_times, feature = (
            extract_temporal_voltage_feature(
                sol=sol,
                K=K,
                t_start=t_start,
                t_end=t_end
            )
        )

        features.append(feature)

    X = np.asarray(features)

    return sample_times, X

def build_repeated_noise_dataset(
    sols_by_class,
    K,
    t_start,
    t_end
):

    X = []
    y = []

    sample_times = None

    for m, class_sols in enumerate(sols_by_class):

        for sol in class_sols:

            sample_times, feature = (
                extract_temporal_voltage_feature(
                    sol=sol,
                    K=K,
                    t_start=t_start,
                    t_end=t_end
                )
            )

            X.append(feature)
            y.append(m)

    X = np.asarray(X)
    y = np.asarray(y)

    return sample_times, X, y


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

