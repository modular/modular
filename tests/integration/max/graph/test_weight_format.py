# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
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
