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

"""Backend selection for expert-parallel graph operations."""

from __future__ import annotations

from typing import Protocol

from max.graph import BufferValue, Dim, TensorValue
from max.nn.ep_config import NVSHMEM_EP_BACKEND, EPConfig


class EPBackend(Protocol):
    """The graph operations an alternate EP backend must provide."""

    def init(
        self, config: EPConfig, gpus: list[TensorValue]
    ) -> list[TensorValue]:
        """Initializes the backend and returns its pointers and ranks."""
        ...

    def dispatch(
        self,
        config: EPConfig,
        input_tokens: list[TensorValue],
        topk_ids: list[TensorValue],
        router_weights: list[TensorValue],
        input_scales: list[TensorValue] | None,
        atomic_counters: list[BufferValue],
        send_buf_ptrs: TensorValue,
    ) -> list[tuple[TensorValue, ...]]:
        """Dispatches each device's tokens and returns its expert inputs."""
        ...

    def combine(
        self,
        config: EPConfig,
        input_tokens: list[TensorValue],
        atomic_counters: list[BufferValue],
        send_buf_ptrs: TensorValue,
        num_tokens_per_device: list[Dim],
    ) -> list[TensorValue]:
        """Combines each device's expert outputs."""
        ...


_BACKENDS: dict[str, EPBackend] = {}


def register_ep_backend(name: str, backend: EPBackend) -> None:
    """Registers an alternate EP backend under ``name``."""
    _BACKENDS[name] = backend


def backend_for(config: EPConfig) -> EPBackend | None:
    """Returns the alternate backend ``config`` names, or ``None`` for NVSHMEM.

    Raises:
        ValueError: If no backend is registered under ``config.backend``.
    """
    if config.backend == NVSHMEM_EP_BACKEND:
        return None
    try:
        return _BACKENDS[config.backend]
    except KeyError:
        registered = ", ".join(sorted(_BACKENDS)) or "none"
        raise ValueError(
            f"unknown EP backend {config.backend!r}; registered backends:"
            f" {registered}"
        ) from None
