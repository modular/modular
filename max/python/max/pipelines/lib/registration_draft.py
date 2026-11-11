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
"""Builds registered draft components for the shared speculative pipeline."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, Protocol, TypeVar

from max import functional as F
from max.graph.weights import WeightData
from max.nn import Module as ModuleV3
from max.nn.legacy.layer import Module as GraphApiModule
from max.nn.outputs import ReturnHiddenStates
from max.pipelines.speculative import (
    DraftAliases,
    SpeculativeConfig,
    SpeculativeMethod,
    proposers,
    validate_draft_state_dict,
)
from transformers import AutoConfig

from .graph_specs import GraphRole
from .interfaces.pipeline_model import _cast_to_parameter_dtypes
from .module_registration import construct_module, device_mesh
from .registration_graph_api import load_graph_api_module
from .utils import parse_state_dict_from_weights
from .weight_transform import CheckpointRole, WeightEntry, WeightTransform

if TYPE_CHECKING:
    from max.pipelines.speculative.block_driver import BlockProposer
    from max.pipelines.speculative.driver import SequentialProposer

    from .arch_lookup import SupportedArchitecture
    from .interfaces.pipeline_model import ModelLoadContext

__all__ = [
    "DraftProposerFactory",
    "DraftRegistration",
    "RegisteredDraftBuilder",
    "TargetHiddenStates",
    "default_target_hidden_states",
]

_ModuleT = TypeVar("_ModuleT")
_ConfigT = TypeVar("_ConfigT")
_ModuleT_contra = TypeVar("_ModuleT_contra", contravariant=True)
_ConfigT_contra = TypeVar("_ConfigT_contra", contravariant=True)


class DraftProposerFactory(Protocol[_ModuleT_contra, _ConfigT_contra]):
    """Returns the proposer that drives a registered draft."""

    def __call__(
        self,
        module: _ModuleT_contra,
        config: _ConfigT_contra,
        *,
        speculative: SpeculativeConfig,
        takes_distributed_inputs: bool,
        target_text_config: AutoConfig,
    ) -> SequentialProposer[Any, Any, Any, Any] | BlockProposer[Any, Any]:
        """Returns the proposer for ``module``, built from ``config``.

        Args:
            module: The constructed draft module.
            config: The config its registration finalized.
            speculative: The speculative config.
            takes_distributed_inputs: Whether the target's verify takes the
                distributed signature, as every DeepSeek-family target's
                does, so the proposer must too.
            target_text_config: The target's text config, whose logit and
                embedding scales a draft that projects through the
                target's head applies.
        """
        ...


@dataclass(frozen=True)
class TargetHiddenStates:
    """The hidden states a draft reads off its target."""

    mode: ReturnHiddenStates
    """What the target returns."""
    layer_ids: tuple[int, ...] = ()
    """The target layers it captures, as MAX indices, for
    :attr:`~max.nn.legacy.transformer.ReturnHiddenStates.SELECTED_LAYERS`."""


def default_target_hidden_states(
    speculative: SpeculativeConfig, draft_config: object
) -> TargetHiddenStates:
    """Returns the target hidden states the method's own helpers select.

    A draft whose config names ``target_layer_ids`` reads those layers;
    any other reads the target's last normalized layer.
    """
    return TargetHiddenStates(
        mode=proposers.target_hidden_states(speculative, draft_config),
        layer_ids=proposers.capture_layers(speculative, draft_config),
    )


@dataclass(frozen=True)
class DraftRegistration(Generic[_ModuleT, _ConfigT]):
    """The construction and proposer contract of a draft.

    The config factory only finalizes configuration. The framework loads the
    checkpoint, constructs the registered module and drives it through its
    proposer with the method's driver. Other drafts retain their existing
    builders until migrated.
    """

    method: SpeculativeMethod
    configure: Callable[[ModelLoadContext, Mapping[str, WeightData]], _ConfigT]
    """Finalizes the draft's config from its checkpoint, before the module
    is constructed."""
    create_proposer: DraftProposerFactory[_ModuleT, _ConfigT] | None = None
    """Returns the draft's proposer, for a draft the method's own proposer
    can't drive. It is a fallback: a target adapter's proposer, when the
    target's architecture has one, runs instead. ``None`` runs the method's
    own, as :func:`~max.pipelines.speculative.proposers.sequential_proposer`
    and :func:`~max.pipelines.speculative.proposers.block_proposer` pick
    it."""
    target_hidden_states: Callable[
        [SpeculativeConfig, _ConfigT], TargetHiddenStates
    ] = default_target_hidden_states
    """Returns the hidden states the target returns for the draft, and the
    layers it captures for them."""
    shared_modules: tuple[Literal["embed_tokens", "lm_head"], ...] = ()
    optional_weights: frozenset[str] = frozenset()
    drafter_export: Callable[[ModelLoadContext, _ConfigT], object] | None = None
    """Describes what the graph bakes in of the draft, for an export's
    sidecar."""

    strict_weights: bool = True
    """Rejects unexpected owned checkpoint weights as well as missing ones.

    Missing required weights are always rejected. Shared target modules and
    absent optional weights remain valid. Set this to ``False`` only when
    the draft intentionally accepts unused checkpoint weights; those keys
    are logged as warnings instead.
    """

    def bind_shared_modules(self, target: object, draft: object) -> None:
        """Aliases declared target modules onto the draft before tracing.

        Raises:
            ValueError: If either component lacks a declared module.
        """
        bindings: list[tuple[str, object]] = []
        for name in self.shared_modules:
            source = getattr(target, name, None)
            destination = getattr(draft, name, None)
            if not isinstance(
                source, (GraphApiModule, ModuleV3)
            ) or not isinstance(destination, (GraphApiModule, ModuleV3)):
                raise ValueError(
                    f"target and draft must expose shared module {name!r}"
                )
            bindings.append((name, source))
        for slot, source in bindings:
            setattr(draft, slot, source)


@dataclass
class RegisteredDraftBuilder:
    """Constructs a registered draft without a pipeline model.

    A graph-API draft loads its checkpoint as a registered decoder does. A
    ModuleV3 draft holds no weights: it compiles with those
    :meth:`checkpoint_weights` returns.
    """

    architecture: SupportedArchitecture
    _weights: dict[str, Any] = field(default_factory=dict, init=False)

    def build_module(
        self, context: ModelLoadContext
    ) -> tuple[GraphApiModule | ModuleV3[..., Any], object]:
        """Loads the draft checkpoint and constructs its registered module.

        Raises:
            ValueError: If the requested method differs from the
                registration, or the checkpoint lacks a weight the draft
                owns, or strict loading finds unexpected owned weights.
        """
        registration = self.architecture.draft
        assert registration is not None
        pipeline_config = context.pipeline_config
        speculative = pipeline_config.speculative
        if (
            speculative is None
            or speculative.speculative_method != registration.method
        ):
            raise ValueError(
                f"{self.architecture.name} requires speculative method "
                f"{registration.method!r}"
            )
        module_cls = self.architecture.modules[GraphRole.DRAFT]
        checkpoint = pipeline_config.draft_model or pipeline_config.model
        state_dict = parse_state_dict_from_weights(
            pipeline_config,
            context.weights,
            context.adapter,
            hf_config=checkpoint.huggingface_config,
            model_config=checkpoint,
        )
        config = registration.configure(context, state_dict)
        shared = tuple(f"{name}." for name in registration.shared_modules)
        owned = {
            name: value
            for name, value in state_dict.items()
            if not name.startswith(shared)
        }
        module: GraphApiModule | ModuleV3[..., Any]
        if issubclass(module_cls, GraphApiModule):
            # validate_draft_state_dict checks what the draft owns: shared
            # weights load with the target.
            module = load_graph_api_module(
                module_cls, config, owned, strict=False
            )
            names = list(module.raw_state_dict())
        else:
            assert context.kv_params is not None
            devices = [ref.to_device() for ref in context.kv_params.devices]
            with F.lazy():
                module = construct_module(
                    module_cls,
                    config,
                    device_mesh(
                        devices, pipeline_config.model.data_parallel_degree
                    ),
                )
            names = [name for name, _ in module.parameters]
            self._weights = _cast_to_parameter_dtypes(module, owned)
        if registration.strict_weights:
            unexpected = set(owned) - set(names)
            if unexpected:
                raise ValueError(
                    f"Draft state_dict has unexpected weights: {sorted(unexpected)}"
                )
        validate_draft_state_dict(
            (
                name
                for name in names
                if name not in registration.optional_weights
                or name in state_dict
            ),
            state_dict.keys(),
            DraftAliases(always=shared),
        )
        return module, config

    def checkpoint_weights(
        self, context: ModelLoadContext, config: object
    ) -> dict[str, Any]:
        """Returns the weights a ModuleV3 draft compiles with.

        They are the checkpoint's, named as the module names them, without
        the shared ones its target holds.
        """
        del context, config
        weights, self._weights = self._weights, {}
        return weights

    def weight_entries(
        self, context: ModelLoadContext, config: object
    ) -> list[WeightEntry]:
        """Returns the entry that names the draft's weights.

        Empty unless its checkpoint loads through rules, which then name the
        draft's own weights and leave out those it shares with its target.
        """
        del config
        if not isinstance(context.adapter, WeightTransform):
            return []
        return [WeightEntry(CheckpointRole.DRAFT, context.adapter)]
