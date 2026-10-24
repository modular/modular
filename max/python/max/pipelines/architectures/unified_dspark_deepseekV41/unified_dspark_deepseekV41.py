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
"""Speculative decoding driver for DeepSeek-V4.1-Flash with DSpark drafting."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from max.graph import BufferType, DeviceRef, TensorType, TensorValue
from max.nn.layer import Module
from max.pipelines.speculative.block_driver import BlockDriver
from max.pipelines.speculative.config import SpeculativeConfig
from max.pipelines.speculative.spec_input_types import SpecDecodeInputTypeSpec

from ..deepseekV41.dspark_drafter import DeepseekV41WithDSpark
from ..deepseekV41.kv_leaves import V41KVLeafSpec
from ..deepseekV41.model_config import DeepseekV41Config
from .spec_adapters import (
    DeepseekV41BlockTarget,
    DSparkDeepseekV41Proposer,
    TrunkTaps,
)

__all__ = ["UnifiedDSparkDeepseekV41"]


class _SignatureOnly(Module):
    """Stands in for the model in a driver built only for its signature."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("a signature-only driver builds no forward")


class UnifiedDSparkDeepseekV41(BlockDriver[TrunkTaps, TensorValue]):
    """Verifies drafts with the target model and drafts blocks with DSpark.

    Target and draft are one :class:`DeepseekV41WithDSpark`; ``draft`` names
    its drafter so the driver's module tree has both roles, but the weights
    are loaded and registered through ``target`` alone. On more than one
    device both roles run on the target's per-device copies (:meth:`shard`):
    tensor-parallel, or data-parallel with expert-parallel MoE when the
    config's data-parallel degree is the device count.

    The model takes graph values at construction (its Engram tables are graph
    inputs), so the graph's signature has to exist before it does: built with
    ``model=None``, this driver only answers :meth:`input_types` (and
    :meth:`decode_inputs`), from the same spec a model-carrying one uses.

    ``trailing_input_types`` are this model's own graph inputs, appended
    after the driver's standard inputs, in order: the Engram lookback window
    (when the config has Engram), then the pinned Engram tables, one
    weight/scale pair per device and Engram layer.
    """

    def __init__(
        self,
        config: DeepseekV41Config,
        speculative_config: SpeculativeConfig,
        specs: Sequence[V41KVLeafSpec],
        *,
        model: DeepseekV41WithDSpark | None,
        trailing_input_types: Sequence[TensorType | BufferType] = (),
        enable_structured_output: bool = False,
    ) -> None:
        if speculative_config.num_speculative_tokens is not None:
            config.validate_num_speculative_tokens(
                speculative_config.num_speculative_tokens
            )
        verifier = DeepseekV41BlockTarget(model, config, specs)
        drafter = DSparkDeepseekV41Proposer(model, config)
        super().__init__(
            verifier.verify,
            drafter,
            target_model=model if model is not None else _SignatureOnly(),
            draft_model=model.dspark if model is not None else _SignatureOnly(),
            input_spec=SpecDecodeInputTypeSpec(
                devices=config.devices,
                # Tensor parallelism needs only signal buffers; data
                # parallelism also needs the host offsets and replica splits
                # of a distributed signature.
                distributed=config.data_parallel_degree > 1,
                data_parallel_degree=config.data_parallel_degree,
                include_signal_buffers=len(config.devices) > 1,
                enable_vision=True,
                vision_hidden_size=config.hidden_size,
            ),
            speculative_config=speculative_config,
            enable_structured_output=enable_structured_output,
            vocab_size=config.vocab_size,
            ep_input_types=verifier.ep_input_types(),
            trailing_input_types=trailing_input_types,
        )
        self.config = config
        self._verifier = verifier
        self._drafter = drafter

    def shard(
        self,
        devices: Sequence[DeviceRef],
        engram_host_tables: Sequence[Any],
    ) -> list[DeepseekV41WithDSpark]:
        """Runs both roles on per-device copies of :attr:`target`.

        Call it once, after the weights are loaded into :attr:`target`, which
        stays the module the weights registry is built from. Returns the
        per-device copies, one per device.
        """
        assert isinstance(self.target, DeepseekV41WithDSpark)
        replicas = (
            self.target.data_parallel_replicas(devices, engram_host_tables)
            if self.config.data_parallel_degree > 1
            else self.target.tensor_parallel_replicas(
                devices, engram_host_tables
            )
        )
        self._verifier.replicas = replicas
        self._drafter.replicas = replicas
        return replicas
