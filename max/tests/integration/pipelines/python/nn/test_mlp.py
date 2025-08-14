# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import pytest
import torch
from max.dtype import DType
from shared_mlp_impl import compare_mlp_outputs


@pytest.mark.parametrize("use_subgraphs", [True, False])
def test_mlp(use_subgraphs: bool) -> None:
    compare_mlp_outputs(
        1024,
        1024,
        "silu",
        torch.float32,
        DType.float32,
        use_subgraphs=use_subgraphs,
    )
    compare_mlp_outputs(
        2048,
        1024,
        "gelu",
        torch.float32,
        DType.float32,
        use_subgraphs=use_subgraphs,
    )
    compare_mlp_outputs(
        1024,
        512,
        "gelu_tanh",
        torch.float32,
        DType.float32,
        use_subgraphs=use_subgraphs,
    )
    compare_mlp_outputs(
        256,
        1024,
        "tanh",
        torch.float32,
        DType.float32,
        use_subgraphs=use_subgraphs,
    )
    compare_mlp_outputs(
        2048,
        1024,
        "gelu",
        torch.float32,
        DType.float32,
        has_bias=True,
        use_subgraphs=use_subgraphs,
    )

    # TODO(MODELS-506): Investigate high atol on very few elements at index (0, _) when using bias.
    compare_mlp_outputs(
        256,
        1024,
        "tanh",
        torch.float32,
        DType.float32,
        has_bias=True,
        use_subgraphs=use_subgraphs,
    )
    compare_mlp_outputs(
        1024,
        1024,
        "silu",
        torch.float32,
        DType.float32,
        has_bias=True,
        use_subgraphs=use_subgraphs,
    )
    compare_mlp_outputs(
        1024,
        512,
        "gelu_tanh",
        torch.float32,
        DType.float32,
        has_bias=True,
        use_subgraphs=use_subgraphs,
    )
    compare_mlp_outputs(
        1024,
        2048,
        "gelu",
        torch.float32,
        DType.float32,
        has_bias=True,
        use_subgraphs=use_subgraphs,
    )

    # TODO: Investigate why the following tests fail
    # compare_mlp_outputs(4096, 2048, "relu", TORCH_DTYPE, DTYPE)
    # compare_mlp_outputs(2048, 4096, "sigmoid", TORCH_DTYPE, DTYPE)
