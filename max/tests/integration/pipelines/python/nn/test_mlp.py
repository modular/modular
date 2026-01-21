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

from __future__ import annotations

import pytest
import torch
from max.dtype import DType
from shared_mlp_impl import compare_gated_mlp_outputs, compare_mlp_outputs


@pytest.mark.parametrize("use_subgraphs", [True, False])
@pytest.mark.parametrize(
    "hidden_dim,dim,activation,has_bias",
    [
        (1024, 1024, "silu", False),
        (2048, 1024, "gelu", False),
        (1024, 512, "gelu_tanh", False),
        (256, 1024, "tanh", False),
        (2048, 1024, "gelu", True),
        # TODO(MODELS-506): Investigate high atol on very few elements at index (0, _) when using bias.
        (256, 1024, "tanh", True),
        (1024, 1024, "silu", True),
        (1024, 512, "gelu_tanh", True),
        (1024, 2048, "gelu", True),
    ],
)
def test_mlp(
    hidden_dim: int,
    dim: int,
    activation: str,
    has_bias: bool,
    use_subgraphs: bool,
) -> None:
    compare_mlp_outputs(
        hidden_dim,
        dim,
        activation,
        torch.float32,
        DType.float32,
        has_bias=has_bias,
        use_subgraphs=use_subgraphs,
    )


# TODO: Investigate why the following tests fail
# compare_mlp_outputs(4096, 2048, "relu", TORCH_DTYPE, DTYPE)
# compare_mlp_outputs(2048, 4096, "sigmoid", TORCH_DTYPE, DTYPE)


@pytest.mark.parametrize("use_subgraphs", [True, False])
@pytest.mark.parametrize(
    "in_features,hidden_features,out_features,activation,has_bias",
    [
        (256, None, None, "silu", False),
        (512, 1024, None, "silu", False),
        (1024, None, 512, "gelu", False),
        (256, 512, 256, "gelu_tanh", False),
        (512, None, None, "tanh", False),
        (1024, 2048, 1024, "silu", True),
        (256, None, None, "gelu", True),
        (512, 1024, 512, "gelu_tanh", True),
    ],
)
def test_gated_mlp(
    in_features: int,
    hidden_features: int | None,
    out_features: int | None,
    activation: str,
    has_bias: bool,
    use_subgraphs: bool,
) -> None:
    compare_gated_mlp_outputs(
        in_features,
        hidden_features,
        out_features,
        activation,
        torch.float32,
        DType.float32,
        has_bias=has_bias,
        use_subgraphs=use_subgraphs,
    )
