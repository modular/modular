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

from pathlib import Path

import pytest
from max.graph.weights import WeightsFormat, weights_format


def test_weights_format__raises_with_no_weights_path() -> None:
    with pytest.raises(ValueError):
        weights_format([])


def test_weights_format__raises_with_bad_weights_path() -> None:
    with pytest.raises(ValueError):
        weights_format([Path("this_is_a_random_weight_path_without_extension")])


def test_weights_format__raises_with_conflicting_weights_path() -> None:
    with pytest.raises(ValueError):
        weights_format(
            [
                Path("this_is_a_random_weight_path_without_extension"),
                Path("this_is_a_gguf_file.gguf"),
            ]
        )


def test_weights_format__correct_weights_format() -> None:
    assert weights_format([Path("model_a.gguf")]) == WeightsFormat.gguf
    assert (
        weights_format(
            [Path("model_b.safetensors"), Path("model_c.safetensors")]
        )
        == WeightsFormat.safetensors
    )
