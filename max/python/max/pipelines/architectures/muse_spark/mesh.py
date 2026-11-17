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

"""Muse Spark's device mesh and its expert-parallel exchange.

Attention heads, KV heads and the vocabulary split over the one ``tp``
axis; every layer's experts split whole over the same devices.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from max.driver import Device
from max.dtype import DType
from max.nn.ep_config import EPConfig, calculate_ep_max_tokens_per_rank
from max.tensor.sharding import DeviceMesh

if TYPE_CHECKING:
    from .model_config import MuseSparkTextConfig


def device_mesh(
    config: MuseSparkTextConfig, devices: Sequence[Device], ep_size: int
) -> DeviceMesh:
    """The one-axis mesh over ``devices``.

    It first checks that the model splits evenly over them.
    """
    n = len(devices)
    # Only NVFP4 experts split over the devices; BF16 ones are replicated.
    expert_parallel = config.quant_config is not None
    if n > 1:
        splits = [
            ("attention heads", config.num_attention_heads),
            ("KV heads", config.num_key_value_heads),
        ]
        if expert_parallel:
            splits.append(("experts", config.num_local_experts))
        for name, count in splits:
            if count % n:
                raise ValueError(
                    f"Muse Spark splits its {count} {name} evenly over the"
                    f" devices; {n} devices do not divide them."
                )
        if expert_parallel and ep_size != n:
            raise ValueError(
                f"Muse Spark on {n} devices runs its experts expert-parallel"
                f" over all of them: set --ep-size {n} (got {ep_size})."
            )
    return DeviceMesh(tuple(devices), (n,), ("tp",))


def ep_config(
    config: MuseSparkTextConfig,
    n_devices: int,
    max_batch_input_tokens: int,
) -> EPConfig:
    """The NVFP4 layers' token exchange.

    It moves NVFP4 tokens of ``moe_hidden_size``, each device sending its own
    rows.
    """
    quant_config = config.quant_config
    assert quant_config is not None and quant_config.is_nvfp4
    return EPConfig(
        dispatch_dtype=DType.uint8,
        dispatch_quant_config=quant_config,
        combine_dtype=DType.bfloat16,
        hidden_size=config.moe_hidden_size,
        top_k=config.num_experts_per_tok,
        n_experts=config.num_local_experts,
        max_tokens_per_rank=calculate_ep_max_tokens_per_rank(
            max_batch_input_tokens=max_batch_input_tokens,
            ep_size=n_devices,
            data_parallel_degree=1,
        ),
        n_gpus_per_node=n_devices,
        n_nodes=1,
    )


def bf16_dispatch(ep: EPConfig) -> EPConfig:
    """``ep`` carrying BF16 tokens, for the BF16 layers' experts.

    Both formats share one exchange, so EP starts from this config: its
    messages are the larger, and the buffers are sized by the first start.
    """
    return replace(
        ep, dispatch_dtype=DType.bfloat16, dispatch_quant_config=None
    )
