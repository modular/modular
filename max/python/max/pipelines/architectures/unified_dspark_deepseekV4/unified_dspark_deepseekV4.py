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
"""DeepSeek-V4 + its DSpark stages on the block driver."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from max.dtype import DType
from max.graph import BufferType, DeviceRef, Dim, TensorType, TensorValue, Value
from max.nn.kv_cache import KVCacheParamInterface
from max.pipelines.speculative.block_driver import BlockDriver
from max.pipelines.speculative.config import SpeculativeConfig
from max.pipelines.speculative.spec_input_types import SpecDecodeInputTypeSpec
from typing_extensions import override

from ..deepseekV4.deepseekV4 import DeepseekV4
from ..deepseekV4.model_config import DeepseekV4Config
from .spec_adapters import (
    DeepseekV4BlockTarget,
    DSparkDeepseekV4Proposer,
    TrunkHidden,
)

__all__ = ["UnifiedDSparkDeepseekV4"]


class UnifiedDSparkDeepseekV4(BlockDriver[TrunkHidden, TensorValue]):
    """Verify through the trunk, draft a block through the DSpark stages.

    Target and draft are one :class:`DeepseekV4`; ``draft`` names its
    ``mtp`` stages so the driver's module tree has both roles, but the
    weights are loaded and registered through ``target`` alone. On more than
    one device both roles run tensor-parallel on the target's per-device
    copies (:meth:`shard`).
    """

    target: DeepseekV4

    def __init__(
        self,
        config: DeepseekV4Config,
        speculative_config: SpeculativeConfig,
        *,
        enable_structured_output: bool = False,
    ) -> None:
        model = DeepseekV4(config)
        verifier = DeepseekV4BlockTarget(model, config)
        drafter = DSparkDeepseekV4Proposer(model, config)
        super().__init__(
            verifier,
            drafter,
            target_model=model,
            draft_model=model.mtp,
            input_spec=SpecDecodeInputTypeSpec(
                devices=config.devices,
                # Tensor-parallel only: the replicas' collectives need signal
                # buffers, the driver nothing else of a distributed signature.
                distributed=False,
                include_signal_buffers=len(config.devices) > 1,
            ),
            speculative_config=speculative_config,
            enable_structured_output=enable_structured_output,
        )
        self.config = config
        self._verifier = verifier
        self._drafter = drafter

    @override
    @property
    def has_trailing_inputs(self) -> bool:
        return True

    @override
    def input_types(
        self, kv_params: KVCacheParamInterface | None = None
    ) -> tuple[TensorType | BufferType, ...]:
        """The canonical signature, then one host ``uint8[windows_r{ratio}]``
        per compression ratio whose length is the verify forward's window
        count (:func:`~..deepseekV4.layers.ragged.window_count` of the merged
        lengths)."""
        return (
            *super().input_types(kv_params),
            *(
                TensorType(DType.uint8, [f"windows_r{r}"], DeviceRef.CPU())
                for r in self.config.window_ratios
            ),
        )

    def verify_windows(self, trailing: Sequence[Value[Any]]) -> dict[int, Dim]:
        """The window counts :meth:`input_types` appended, per ratio."""
        return {
            r: v.tensor.shape[0]
            for r, v in zip(self.config.window_ratios, trailing, strict=True)
        }

    def shard(self, devices: Sequence[DeviceRef]) -> None:
        """Runs both roles on per-device copies of :attr:`target`.

        Call it once, after the weights are loaded into :attr:`target`, which
        stays the module the weights registry is built from.
        """
        replicas = self.target.tensor_parallel_replicas(devices)
        self._verifier.replicas = replicas
        self._drafter.replicas = replicas
