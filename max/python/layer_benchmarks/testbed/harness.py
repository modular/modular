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

"""Abstract base class for layer test harnesses."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, cast

import numpy as np
import torch
from max.driver import Accelerator, Buffer, DLPackArray
from max.engine import InferenceSession, Model
from max.graph import Graph
from max.interfaces import RequestID, TextGenerationContext, TokenBuffer
from max.kv_cache import PagedKVCacheManager
from max.pipelines.core import TextContext
from typing_extensions import TypeVar

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

StaticParamsT = TypeVar("StaticParamsT")
DynamicParamsT = TypeVar("DynamicParamsT")
ContextT = TypeVar("ContextT")

_T = TypeVar("_T", bound="DataclassInstance")


def dict_to_dataclass(cls: type[_T], d: dict[str, Any]) -> _T:
    """Convert a dict to a dataclass, validating required fields.

    Raises:
        TypeError: If required fields (no default) are missing from the dict,
            or if the dict contains keys not defined on the dataclass.
    """
    fields = dataclasses.fields(cls)
    known = {f.name for f in fields}
    extra = set(d) - known
    if extra:
        raise TypeError(
            f"{cls.__name__} got unexpected fields: {sorted(extra)}"
        )
    required = {
        f.name
        for f in fields
        if f.default is dataclasses.MISSING
        and f.default_factory is dataclasses.MISSING
    }
    missing = required - set(d)
    if missing:
        raise TypeError(
            f"{cls.__name__} missing required fields: {sorted(missing)}"
        )
    return cls(**{k: v for k, v in d.items() if k in known})


class CompiledLayerBundle(NamedTuple):
    """Result of compiling a layer graph.

    Attributes:
        compiled_model: The compiled model from session.load().
        device: The GPU accelerator device.
        session: The InferenceSession used for compilation.
    """

    compiled_model: Model
    device: Accelerator
    session: InferenceSession


class LayerTestHarness(ABC, Generic[StaticParamsT, DynamicParamsT, ContextT]):
    """Abstract base class for layer test harnesses.

    A harness knows how to construct a specific layer type, build its
    computation graph, prepare inputs for execution, and provide a
    torch reference implementation for correctness testing.

    Subclasses implement the layer-specific logic; the LayerTestRunner
    provides layer-agnostic benchmark/profile/IR-dump/correctness modes.

    Type parameters:
        StaticParamsT: Dataclass defining model-config-level parameters.
        DynamicParamsT: Dataclass defining per-shape parameters.
        ContextT: Type of cleanup context returned by prepare_inputs.

    """

    @staticmethod
    @abstractmethod
    def static_params_type() -> type[StaticParamsT]:
        """Return the dataclass type for static params."""
        ...

    @staticmethod
    @abstractmethod
    def dynamic_params_type() -> type[DynamicParamsT]:
        """Return the dataclass type for dynamic params."""
        ...

    def __init__(
        self,
        static_params: StaticParamsT,
        session: InferenceSession,
        device: Accelerator,
    ) -> None:
        self.static_params = static_params
        self.session = session
        self.device = device

    @property
    def num_devices(self) -> int:
        """Number of GPU devices required. Override for multi-GPU harnesses."""
        return 1

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this harness (e.g. 'attention_with_rope')."""
        ...

    @abstractmethod
    def build_and_compile(self) -> CompiledLayerBundle:
        """Build the layer graph and compile it.

        Returns:
            A CompiledLayerBundle containing the compiled model.
        """
        ...

    @abstractmethod
    def prepare_inputs(
        self, bundle: CompiledLayerBundle, dynamic_params: DynamicParamsT
    ) -> tuple[list[Buffer], ContextT]:
        """Prepare inputs for a single benchmark shape.

        Args:
            bundle: The compiled layer bundle.
            dynamic_params: Per-shape parameters (e.g. batch_size, seq_len).

        Returns:
            A tuple of (execute_args, context) where execute_args is the
            list passed to compiled_model.execute() and context is passed
            to cleanup_inputs().
        """
        ...

    @abstractmethod
    def cleanup_inputs(
        self, bundle: CompiledLayerBundle, context: ContextT
    ) -> None:
        """Clean up resources allocated by prepare_inputs.

        Override this for layers that allocate resources (e.g. KV cache
        claims) that must be released between shapes.
        """
        ...

    @abstractmethod
    def build_graph(self) -> tuple[Graph, dict[str, DLPackArray]]:
        """Build the layer graph without compiling.

        Returns:
            A tuple of (graph, weights_registry) that can be passed to
            session.load() or serialized via str(graph) for IR dump.
        """
        ...

    @abstractmethod
    def cuda_graph_eligible(self, dynamic_params: DynamicParamsT) -> bool:
        """Whether this shape should use CUDA graph capture/replay."""
        ...

    # ------------------------------------------------------------------ #
    # Correctness support
    # ------------------------------------------------------------------ #

    @abstractmethod
    def torch_reference_layer(self, device: str = "cuda") -> torch.nn.Module:
        """Return a torch reference module for correctness comparison."""
        ...

    @abstractmethod
    def prepare_torch_inputs(
        self,
        execute_args: list[Buffer],
        dynamic_params: DynamicParamsT,
        device: str = "cuda",
    ) -> list[torch.Tensor]:
        """Prepare torch inputs matching the MAX inputs for a given shape.

        Args:
            execute_args: The execute args returned by prepare_inputs,
                so the torch reference receives identical input data.
            dynamic_params: Per-shape parameters (e.g. batch_size, seq_len).
            device: Torch device string.
        """
        ...

    def postprocess_torch_output(self, output: object) -> torch.Tensor:
        """Extract the comparable tensor from the torch module's output.

        Override for modules that return tuples or need shape adjustment.
        Default: return output as-is (works for modules returning a single tensor).
        """
        assert isinstance(output, torch.Tensor)
        return output


# ------------------------------------------------------------------ #
# Shared helpers for TP attention harnesses
# ------------------------------------------------------------------ #


def prepare_tp_attention_inputs(
    bundle: CompiledLayerBundle,
    dynamic_params: Mapping[str, int],
    *,
    kv_manager: PagedKVCacheManager,
    hidden_size: int,
    max_seq_len: int,
    signal_buffers: list[Buffer],
    tp_degree: int,
) -> tuple[list[Buffer], list[TextGenerationContext]]:
    """Prepare inputs for a TP attention benchmark shape.

    Allocates KV cache claims, builds input tensors, and assembles
    per-device KV args + signal buffers into a flat execute_args list.

    Args:
        bundle: The compiled layer bundle.
        dynamic_params: Per-shape params (batch_size, seq_len, ctx_len).
        kv_manager: The KV cache manager.
        hidden_size: Model hidden dimension.
        max_seq_len: Maximum sequence length for the model.
        signal_buffers: Cross-device signal buffers.
        tp_degree: Tensor parallel degree.

    Returns:
        (execute_args, batch) where batch is the list of TextGenerationContext
        objects for cleanup.
    """
    batch_size = dynamic_params["batch_size"]
    seq_len = dynamic_params["seq_len"]
    ctx_len = dynamic_params.get("ctx_len", 0)

    device = bundle.device

    total_len = ctx_len + seq_len

    batch: list[TextGenerationContext] = []
    for _ in range(batch_size):
        ctx = TextContext(
            request_id=RequestID(),
            max_length=max(total_len, max_seq_len),
            tokens=TokenBuffer(np.empty(total_len, dtype=np.int64)),
        )
        kv_manager.claim(ctx.request_id, replica_idx=0)
        kv_manager.alloc(ctx, replica_idx=0)
        if ctx_len > 0:
            ctx.tokens.skip_processing(ctx_len)
        batch.append(ctx)

    kv_runtime = kv_manager.runtime_inputs(
        cast(list[list[TextGenerationContext]], [batch])
    )

    total_tokens = batch_size * seq_len
    input_tensor = Buffer.from_dlpack(
        torch.randn(total_tokens, hidden_size, dtype=torch.bfloat16)
    ).to(device)
    row_offsets = Buffer.from_numpy(
        np.array(
            [i * seq_len for i in range(batch_size + 1)],
            dtype=np.uint32,
        )
    ).to(device)

    kv_args: list[Buffer] = []
    for dev_idx in range(tp_degree):
        dev_runtime = kv_runtime.inputs[dev_idx]
        dev = Accelerator(id=dev_idx)
        assert dev_runtime.attention_dispatch_metadata is not None
        kv_args.extend(
            [
                dev_runtime.blocks.to(dev),
                dev_runtime.cache_lengths.to(dev),
                dev_runtime.lookup_table.to(dev),
                dev_runtime.max_lengths,
                dev_runtime.attention_dispatch_metadata,
            ]
        )

    execute_args: list[Buffer] = [
        input_tensor,
        row_offsets,
        *kv_args,
        *signal_buffers,
    ]

    return execute_args, batch


def cleanup_tp_attention_inputs(
    kv_manager: PagedKVCacheManager,
    batch: list[TextGenerationContext],
) -> None:
    """Release KV cache claims allocated by prepare_tp_attention_inputs."""
    for ctx in batch:
        kv_manager.release(ctx.request_id, replica_idx=0)
