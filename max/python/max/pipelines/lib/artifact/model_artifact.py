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
"""The top-level object exported to and loaded from a ``.m`` artifact."""

from __future__ import annotations

import types
import typing
from dataclasses import dataclass, fields

from .codec import collect_hf_refs
from .types import ArtifactComponent, ArtifactError, HuggingFaceRef


@dataclass(frozen=True)
class ModelArtifact:
    """Serializable serve-able pipeline."""

    model: ArtifactComponent | None = None
    tokenizer: ArtifactComponent | None = None
    format_spec: ArtifactComponent | None = None
    engine_config: ArtifactComponent | None = None
    kv_cache: ArtifactComponent | None = None
    batch_processor: ArtifactComponent | None = None
    weight_loader: ArtifactComponent | None = None

    @classmethod
    def component_slots(cls) -> tuple[str, ...]:
        """Returns the names of the component slots."""
        return tuple(f.name for f in fields(cls))

    @classmethod
    def component_types(cls) -> dict[str, type[ArtifactComponent] | None]:
        """Returns the class each slot is deserialized with.

        Returns:
            A mapping from slot name to component class, or to ``None`` for a
            slot still annotated with the :class:`ArtifactComponent` protocol.

        Raises:
            ArtifactError: If a slot's annotation names more than one class or
                a class that isn't an :class:`ArtifactComponent`.
        """
        hints = typing.get_type_hints(cls)
        result: dict[str, type[ArtifactComponent] | None] = {}
        for slot in cls.component_slots():
            hint = hints[slot]
            if typing.get_origin(hint) in (typing.Union, types.UnionType):
                members = [
                    a for a in typing.get_args(hint) if a is not type(None)
                ]
            else:
                members = [hint]
            if len(members) != 1:
                raise ArtifactError(
                    f"{cls.__name__}.{slot}: a component slot must name exactly one"
                    f" class, got {hint}"
                )
            (component_type,) = members
            if component_type is ArtifactComponent:
                result[slot] = None
            elif isinstance(component_type, type) and issubclass(
                component_type, ArtifactComponent
            ):
                result[slot] = component_type
            else:
                raise ArtifactError(
                    f"{cls.__name__}.{slot}: {component_type} is not an"
                    " ArtifactComponent"
                )
        return result

    def components(self) -> dict[str, ArtifactComponent]:
        """Returns the set component slots, keyed by slot name."""
        components: dict[str, ArtifactComponent] = {}
        for slot in self.component_slots():
            component = getattr(self, slot)
            if component is not None:
                assert isinstance(component, ArtifactComponent)
                components[slot] = component
        return components

    def model_sources(self) -> tuple[HuggingFaceRef, ...]:
        """Returns the unique Hugging Face references in the components.

        Serializes every component, so prefer reading
        :attr:`ArtifactMetadata.model_sources` from an exported artifact.
        """
        return collect_hf_refs(
            c.__serialize__() for c in self.components().values()
        )
