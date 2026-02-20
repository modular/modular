# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
from shared_mlp_impl import compare_mlp_outputs


@pytest.mark.parametrize("use_subgraphs", [True, False])
def test_mlp_gpu(use_subgraphs: bool) -> None:
    compare_mlp_outputs(
        2048,
        1024,
        "gelu",
        torch.float32,
        DType.float32,
        use_subgraphs=use_subgraphs,
        n_gpus=1,
    )


@pytest.mark.parametrize("n_gpus", [2, 4])
def test_mlp_distributed(n_gpus: int) -> None:
    compare_mlp_outputs(
        1024,
        1024,
        "gelu",
        torch.float32,
        DType.float32,
        use_subgraphs=False,
        n_gpus=n_gpus,
    )

    compare_mlp_outputs(
        14336,
        4096,
        "silu",
        torch.float32,
        DType.float32,
        use_subgraphs=False,
        n_gpus=n_gpus,
    )

    compare_mlp_outputs(
        14336,
        4096,
        "silu",
        torch.float32,
        DType.float32,
        use_subgraphs=False,
        n_gpus=n_gpus,
        enable_matmul_allreduce=True,
        seq_len=4096,
    )
