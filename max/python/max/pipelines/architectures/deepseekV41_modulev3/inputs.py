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
"""Model inputs of the DeepSeek-V4.1-Flash ModuleV3 pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from max import tree
from max.driver import Buffer
from max.pipelines.lib import ModelInputs
from transformers import PretrainedConfig

from .interface import V41Config


def v41_config(
    huggingface_config: Mapping[str, object] | PretrainedConfig,
) -> V41Config:
    """Reads the interface config off a HF config object or its ``to_dict()``."""
    if not isinstance(huggingface_config, Mapping):
        huggingface_config = huggingface_config.to_dict()
    return V41Config.from_hf(huggingface_config)


@dataclass
class DeepseekV41Inputs(ModelInputs):
    """Inputs for one DeepSeek-V4.1-Flash forward, in compile ABI order.

    Under data parallelism (``data_parallel_splits`` set) the CPU DP split
    pair follows ``input_row_offsets`` and precedes the KV leaves, matching
    ``DeepseekV41.forward``'s variadic order.

    The extras after the KV leaves carry the per-forward Engram and sliding-window
    inputs, in :func:`~.interface.extra_input_names` order.
    """

    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer = field(kw_only=True)

    extras: tuple[Buffer, ...] = field(kw_only=True)
    """The forward inputs after the KV leaves, in ``interface.extra_input_names`` order."""

    ep_inputs: tuple[Buffer, ...] = field(default=(), kw_only=True)
    """The EP comm buffers trailing the extras, in
    ``EPBatchManager.input_types`` order; empty without expert
    parallelism. The batch processor binds them from the model's
    ``EPCommInitializer``."""

    data_parallel_splits: Buffer | None = field(default=None, kw_only=True)
    """CPU int64 ``[dp + 1]`` request-index split boundaries."""

    input_row_offsets_i64: Buffer | None = field(default=None, kw_only=True)
    """CPU int64 replica of ``input_row_offsets``, the split's row bounds."""

    signal_buffers: list[Buffer] = field(default_factory=list, kw_only=True)
    """The pipeline's signal buffers, which a DSpark pairing's graph takes
    after its other inputs. The trunk's own compiled callable appends its
    own, so :attr:`buffers` leaves them out."""

    def __post_init__(self) -> None:
        if (self.data_parallel_splits is None) != (
            self.input_row_offsets_i64 is None
        ):
            raise ValueError(
                "the data-parallel split pair must be set together: "
                "data_parallel_splits and input_row_offsets_i64"
            )

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        """Flat graph inputs in compile ABI order.

        The signal buffers that trail them at TP > 1 are appended by the
        compiled callable itself.
        """
        assert self.kv_cache_inputs is not None
        dp_pair: tuple[Buffer, ...] = ()
        # ``__post_init__`` rejected a half-set pair; this narrows the types.
        if (
            self.data_parallel_splits is not None
            and self.input_row_offsets_i64 is not None
        ):
            dp_pair = (self.data_parallel_splits, self.input_row_offsets_i64)
        return (
            self.tokens,
            self.return_n_logits,
            self.input_row_offsets,
            *dp_pair,
            *tree.leaves(self.kv_cache_inputs),
            *self.extras,
            *self.ep_inputs,
        )
