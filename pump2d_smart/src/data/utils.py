"""Data Preprocessing Utility Functions for Pump Simulation Data.

Provides the space-filling Kennard-Stone algorithm to divide training and validation sets.
"""

import numpy as np


def normalize_data(X: np.ndarray) -> np.ndarray:
    """Normalizes the feature matrix X to have zero mean and unit variance.

    Args:
        X: Feature matrix of shape (N, D).

    Returns:
        Normalized feature matrix of the same shape. (N,D)
        Means (D,)
        Standard deviations (D,)
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (
        (X - mean) / np.where(std == 0, 1, std),
        mean,
        std,
    )  # Avoid division by zero for constant features


def kennard_stone_split(
    X: np.ndarray, train_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Applies the space-filling Kennard-Stone algorithm to split X into train/test sets.
    Before running the algorithm, the data is normalized to have zero mean and unit variance.

    Selects samples sequentially to span the parameter space as uniformly as possible.

    Args:
        X: Feature coordinate matrix of shape (N, D).
        train_size: Target size of the training set.

    Returns:
        Tuple of [selected_indices, remaining_indices] as numpy arrays.
    """
    n_samples = X.shape[0]
    if train_size >= n_samples:
        return np.arange(n_samples), np.empty(0, dtype=np.int64)
    if train_size <= 0:
        return np.empty(0, dtype=np.int64), np.arange(n_samples)

    X_norm, _means, _std = normalize_data(X)

    # Compute pairwise Euclidean distance matrix
    dist_matrix = np.linalg.norm(X_norm[:, None, :] - X_norm[None, :, :], axis=-1)

    # Step 1: Find the two points furthest apart in the feature space
    i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    selected = [i, j]
    remaining = list(set(range(n_samples)) - {i, j})

    # min_distances tracks the minimum distance of each remaining point to the selected set
    min_distances = np.minimum(dist_matrix[remaining, i], dist_matrix[remaining, j])

    # Step 2 & 3: Iteratively select the point maximizing the minimum distance to the training set
    while len(selected) < train_size:
        idx_in_remaining = np.argmax(min_distances)
        new_point = remaining[idx_in_remaining]

        selected.append(new_point)
        remaining.pop(idx_in_remaining)

        if len(remaining) == 0:
            break

        # Update min_distances by comparing with the newly added point
        dist_to_new = dist_matrix[remaining, new_point]
        min_distances = np.minimum(
            np.delete(min_distances, idx_in_remaining), dist_to_new
        )

    selected_indices = np.array(selected, dtype=np.int64)
    remaining_indices = np.array(remaining, dtype=np.int64)
    return selected_indices, remaining_indices
