# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Test to make sure we get a proper error message when we try to load a weight file from a ."""

from os import getenv
from pathlib import Path

import pytest
from max.graph.weights import GGUFWeights


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
