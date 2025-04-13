import pytest
from graph import base


@pytest.fixture
def empty_graph():
    return base.Graph()


@pytest.fixture
def empty_flexible_graph():
    return base.FlexibleGraph()
