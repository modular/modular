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
"""Pipeline executor abstract base class for model execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from max.engine import InferenceSession

from .interfaces.tensor_struct import TensorStruct
from .model_manifest import ModelManifest
from .pipeline_runtime_config import PipelineRuntimeConfig

ContextT = TypeVar("ContextT")
"""Type variable for the context type.

Deliberately unbound. Context lifecycle concerns (request IDs, status
tracking) belong to the serving layer, not the executor. This allows
the same executor to be used in serving, batch offline, notebook, and
test contexts without imposing unnecessary constraints.
"""

InputsT = TypeVar("InputsT", bound=TensorStruct)
"""Type variable for the inputs type.

Bound to :class:`TensorStruct`, a frozen-dataclass base that enforces
all fields are :class:`~max.experimental.tensor.Tensor`,
:class:`~max.driver.Buffer`, or ``Optional`` variants thereof at
class-definition time.

Concrete subclasses should define a ``@dataclass(frozen=True)``
inheriting from :class:`TensorStruct` with named fields for each
input tensor. Optional features use ``Tensor | None`` fields and
are detected at runtime via ``is not None``.

The struct exposes ``.to(device)`` for bulk device transfer, so the
caller retains full control over where inputs reside before passing
them to :meth:`PipelineExecutor.execute`.
"""

OutputsT = TypeVar("OutputsT", bound=TensorStruct)
"""Type variable for the outputs type.

Bound to :class:`TensorStruct`, a frozen-dataclass base that enforces
all fields are :class:`~max.experimental.tensor.Tensor`,
:class:`~max.driver.Buffer`, or ``Optional`` variants thereof at
class-definition time.

Concrete subclasses should define a ``@dataclass(frozen=True)``
inheriting from :class:`TensorStruct` with named fields for each
output tensor (for example, ``result.images``).

The struct exposes ``.to(device)`` for bulk device transfer, so the
caller retains full control over moving outputs to host or another
device after execution.
"""


class PipelineExecutor(ABC, Generic[ContextT, InputsT, OutputsT]):
    """Defines the minimal interface for preparing inputs and executing a compiled model graph.

    ``PipelineExecutor`` deliberately separates input preparation from
    execution into two distinct steps. This decoupling gives the upstream
    caller complete control over device placement and host/device data
    transfer between the two phases. For example, a caller may:

    - Call :meth:`prepare_inputs` on CPU, inspect or log the resulting
      tensors, transfer them to a specific GPU, then call :meth:`execute`.
    - Prepare inputs for multiple batches concurrently, then execute them
      in a specific order determined by a scheduler.
    - Transfer outputs back to host immediately, or keep them on device
      for further downstream processing.

    The executor itself has no opinion about where tensors live. It
    produces and consumes tensors; the caller decides when and where to
    move them.

    The three type parameters are:

    - ``ContextT`` -- The context type representing a single request or
      work item in a batch. Deliberately unbound so that serving, batch,
      and test callers can use their own context types without inheriting
      request lifecycle protocols.
    - ``InputsT`` -- The inputs type, a frozen dataclass inheriting from
      :class:`TensorStruct`.  Produced by :meth:`prepare_inputs` and
      consumed by :meth:`execute` with no intermediate transformation
      required. Optional features use ``Tensor | None`` fields.
    - ``OutputsT`` -- The outputs type, a frozen dataclass inheriting from
      :class:`TensorStruct`.  Returned by :meth:`execute` with named
      field access (for example, ``result.images``).

    Both ``InputsT`` and ``OutputsT`` are bound to :class:`TensorStruct`,
    which enforces at class-definition time that every field is
    ``Tensor``, ``Buffer``, or ``Optional[Tensor | Buffer]``.  The
    struct exposes ``.to(device)`` for caller-controlled bulk device
    transfer.

    Concrete subclasses own all graph construction, compilation, and
    weight loading internally. The constructor receives a
    :class:`~max.pipelines.lib.model_manifest.ModelManifest` for weight
    access, an :class:`~max.engine.InferenceSession` for compilation,
    and a :class:`~max.pipelines.lib.pipeline_runtime_config.PipelineRuntimeConfig`
    for runtime settings.

    Args:
        manifest: The model manifest providing weight access and model
            configuration for all components in the pipeline.
        session: The inference session used to compile and load model
            graphs.
        runtime_config: Model-agnostic runtime settings controlling
            batching, scheduling, and execution behavior.
    """

    @abstractmethod
    def __init__(
        self,
        manifest: ModelManifest,
        session: InferenceSession,
        runtime_config: PipelineRuntimeConfig,
    ) -> None: ...

    @abstractmethod
    def prepare_inputs(self, contexts: list[ContextT]) -> InputsT:
        """Converts a batch of contexts into a structured tensor container ready for graph execution.

        Each context in the batch represents a single request or work item.
        The implementation is responsible for collating, tokenizing, or
        otherwise transforming the batch into the tensor format expected by
        the compiled graph.

        The returned struct may contain tensors on any device. The caller
        is responsible for transferring them to the appropriate device
        before passing them to :meth:`execute`, using ``.to(device)``
        on the returned :class:`TensorStruct`.

        Args:
            contexts: A list of context objects representing the batch of
                requests to prepare inputs for.

        Returns:
            A :class:`TensorStruct` containing the prepared graph inputs
            for the batch.
        """
        ...

    @abstractmethod
    def execute(self, inputs: InputsT) -> OutputsT:
        """Runs the compiled model graph on the provided inputs.

        The inputs should be the :class:`TensorStruct` produced by
        :meth:`prepare_inputs`, passed through without transformation.
        The caller may have transferred them to a different device between
        preparation and execution.

        The returned struct may contain device-resident tensors. The
        caller is responsible for any host transfer needed for
        post-processing, using ``.to(device)`` on the returned
        :class:`TensorStruct`.

        Args:
            inputs: The prepared graph inputs, as returned by
                :meth:`prepare_inputs`.

        Returns:
            A :class:`TensorStruct` containing the model outputs.
        """
        ...
