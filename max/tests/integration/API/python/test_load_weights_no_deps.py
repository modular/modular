# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test to make sure we get a proper error message when we try to load a weight file from a ."""

from os import getenv
from pathlib import Path

import pytest
from max.graph.weights import GGUFWeights, PytorchWeights


@pytest.fixture
def graph_testdata() -> Path:
    """Returns the path to the Modular .derived directory."""
    path = getenv("GRAPH_TESTDATA")
    assert path is not None
    return Path(path)


def test_load_gguf(graph_testdata: Path) -> None:
    """Tests adding an external weight to a graph."""

    with pytest.raises(ImportError) as info:
        weights = GGUFWeights(graph_testdata / "example_data.gguf")
    # Also test the error message, we want to make sure we hit our custom error message
    assert str(info.value) == "Unable to load gguf file, gguf not installed"


def test_load_pytorch(graph_testdata: Path) -> None:
    """Tests adding an external weight to a graph."""

    with pytest.raises(ImportError) as info:
        weights = PytorchWeights(graph_testdata / "example_data.pt")
    assert str(info.value) == "Unable to load pytorch file, torch not installed"
