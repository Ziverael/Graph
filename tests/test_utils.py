from graph import utils
import numpy as np
import pytest


@pytest.mark.parametrize(
    ("xs", "ys", "expected"),
    [
        pytest.param(
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            np.array(
                [[1, 4], [1, 5], [1, 6], [2, 4], [2, 5], [2, 6], [3, 4], [3, 5], [3, 6]]
            ),
            id="regular-case",
        ),
        pytest.param(
            np.array([1, 2, 3]),
            np.array([1]),
            np.array([[1, 1], [2, 1], [3, 1]]),
            id="single-element",
        ),
        pytest.param(
            np.array([1, 2, 3]),
            np.array([]),
            np.array([]).reshape(0, 2),
            id="empty-array",
        ),
    ],
)
def test_get_combinations(xs, ys, expected):
    # when
    result = utils.get_combinations(xs, ys)

    # then
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    ("xs", "ys", "expected"),
    [
        pytest.param(np.array([1]), np.array([1]), np.array([[1, 1]]), id="one-pair"),
        pytest.param(
            np.array([1]),
            np.array([2, 3, 4]),
            np.array([[1, 2], [1, 3], [1, 4]]),
            id="single-element",
        ),
        pytest.param(
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            np.array([[1, 1], [1, 2], [1, 3], [2, 2], [2, 3], [3, 3]]),
            id="multiple-elements",
        ),
        pytest.param(
            np.array([1, 2, 3]),
            np.array([]),
            np.array([]).reshape(0, 2),
            id="empty-array",
        ),
    ],
)
def test_get_unique_combinations(xs, ys, expected):
    # when
    result = utils.get_unique_combinations(xs, ys)

    # then
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    ("i", "j", "k", "n", "expected"),
    [
        pytest.param(np.array([1]), np.array([10]), 2, 16, np.array([False])),
        pytest.param(np.array([10]), np.array([1]), 2, 16, np.array([False])),
        pytest.param(
            np.array([1, 1]), np.array([2, 10]), 1, 10, np.array([False, True])
        ),
        pytest.param(
            np.array([2, 10]), np.array([1, 1]), 1, 10, np.array([False, True])
        ),
        pytest.param(
            np.array([1, 2, 3, 4, 5, 1]),
            np.array([2, 3, 4, 5, 6, 5]),
            4,
            6,
            np.array([True, True, True, True, True, True]),
        ),
        pytest.param(
            np.array([1, 1, 1, 1, 1, 1]),
            np.array([1, 2, 3, 4, 5, 6]),
            2,
            6,
            np.array([False, True, False, False, False, True]),
        ),
    ],
)
def test_is_link_in_ring_lattice(i, j, k, n, expected):
    # when
    result = utils.is_link_in_ring_lattice(i, j, k, n)
    print(i, j)
    print(result)
    # then
    assert np.array_equal(result, expected)
