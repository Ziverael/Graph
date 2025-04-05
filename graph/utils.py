import numpy as np


def get_combinations(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    return np.column_stack([np.repeat(xs, len(ys)), np.tile(ys, len(xs))])


def get_unique_combinations(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Treat pairs (x, y) and (y, x) as the same pair."""
    A, B = np.meshgrid(xs, ys, indexing="ij")
    pairs = np.stack([A.ravel(), B.ravel()], axis=1)
    pairs = np.sort(pairs, axis=1)
    return np.unique(pairs, axis=0)


def is_link_in_ring_lattice(i: np.ndarray, j: np.ndarray, k: int, n: int) -> np.ndarray:
    diff = np.mod(np.abs(i - j), n - 1 - k / 2)
    return (0 < diff) & (diff <= k / 2)


def is_rightmost_neighbor(i: np.ndarray, j: np.ndarray, k: int, n: int) -> np.ndarray:
    diff = np.mod(i - j, n - 1 - k / 2)
    return (0 < diff) & (diff <= k / 2)
