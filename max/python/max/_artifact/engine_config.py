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
"""The ``engine_config`` component.

The settings a pipeline's graphs were exported for.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from max.driver import DeviceSpec
from typing_extensions import Self

from .value_types import (
    ArtifactError,
    ArtifactType,
    decode_field,
    decode_object,
    decode_optional,
    expect,
    unknown_fields,
)


@dataclass(frozen=True)
class EngineSamplingConfig:
    """The optional inputs the sampling graphs were built with.

    Args:
        needs_bitmask_constraints: Whether the graphs take a structured-output
            bitmask.
        enable_variable_logits: Whether the sampler takes a variable number
            of logits per request.
        enable_penalties: Whether the sampler takes frequency, presence and
            repetition penalties.
    """

    needs_bitmask_constraints: bool
    enable_variable_logits: bool
    enable_penalties: bool


@dataclass(frozen=True)
class EngineSpeculativeConfig:
    """The speculative-decoding settings a fused graph was built with.

    Args:
        speculative_method: The speculative-decoding method, such as ``mtp``.
        num_speculative_tokens: The number of tokens drafted per step.
        draft_proposal: How the draft proposes tokens: ``argmax``, or
            ``sampled``, which adds the draft's probabilities to the graph.
        reasoning_parser: The reasoning parser, if any. With one, a block
            draft's graph takes whether each request is still thinking.
    """

    speculative_method: str
    num_speculative_tokens: int
    draft_proposal: str
    reasoning_parser: str | None


@dataclass(frozen=True)
class EngineConfig:
    """The settings a pipeline artifact's graphs were exported for.

    An engine serving the artifact runs with them. ``max export`` copies
    each value from the pipeline's resolved configuration rather than
    serializing that configuration, so the artifact's layout doesn't change
    when the configuration's does.

    Args:
        max_batch_size: The most requests in one batch, per data-parallel
            replica.
        max_length: The most tokens one request holds, prompt and generated
            tokens together.
        max_batch_input_tokens: The most input tokens one step processes, per
            data-parallel replica.
        max_batch_total_tokens: The most context tokens one batch holds, or
            ``None`` for no limit.
        devices: The devices the graphs run on, in the order they use them.
        data_parallel_degree: How many replicas of the model the devices
            hold.
        ep_size: The number of devices the experts are split across, or 1
            without expert parallelism.
        sampling: The sampling graphs' optional inputs.
        speculative: The speculative-decoding settings, or ``None`` for a
            pipeline that doesn't decode speculatively.
    """

    max_batch_size: int
    max_length: int
    max_batch_input_tokens: int
    max_batch_total_tokens: int | None
    devices: tuple[DeviceSpec, ...]
    data_parallel_degree: int
    ep_size: int
    sampling: EngineSamplingConfig
    speculative: EngineSpeculativeConfig | None

    def __serialize__(self) -> Mapping[str, ArtifactType]:
        """Returns the settings as a JSON object."""
        speculative = self.speculative
        return {
            "max_batch_size": self.max_batch_size,
            "max_length": self.max_length,
            "max_batch_input_tokens": self.max_batch_input_tokens,
            "max_batch_total_tokens": self.max_batch_total_tokens,
            "devices": [
                {"type": device.device_type, "id": device.id}
                for device in self.devices
            ],
            "data_parallel_degree": self.data_parallel_degree,
            "ep_size": self.ep_size,
            "sampling": {
                "needs_bitmask_constraints": (
                    self.sampling.needs_bitmask_constraints
                ),
                "enable_variable_logits": self.sampling.enable_variable_logits,
                "enable_penalties": self.sampling.enable_penalties,
            },
            "speculative": None
            if speculative is None
            else {
                "speculative_method": speculative.speculative_method,
                "num_speculative_tokens": speculative.num_speculative_tokens,
                "draft_proposal": speculative.draft_proposal,
                "reasoning_parser": speculative.reasoning_parser,
            },
        }

    @classmethod
    def __deserialize__(cls, value: Mapping[str, ArtifactType]) -> Self:
        """Rebuilds the settings written by :meth:`__serialize__`.

        Raises:
            ArtifactError: If a field is missing, unknown or malformed.
        """
        if unknown := unknown_fields(value, cls):
            raise ArtifactError(f"unknown fields {unknown}")
        return cls(
            max_batch_size=decode_field(value, "max_batch_size", _int),
            max_length=decode_field(value, "max_length", _int),
            max_batch_input_tokens=decode_field(
                value, "max_batch_input_tokens", _int
            ),
            max_batch_total_tokens=decode_field(
                value, "max_batch_total_tokens", decode_optional(_int)
            ),
            devices=decode_field(value, "devices", _devices),
            data_parallel_degree=decode_field(
                value, "data_parallel_degree", _int
            ),
            ep_size=decode_field(value, "ep_size", _int),
            sampling=decode_field(value, "sampling", _sampling),
            speculative=decode_field(
                value, "speculative", decode_optional(_speculative)
            ),
        )


def _int(value: ArtifactType) -> int:
    return expect(value, int)


def _str(value: ArtifactType) -> str:
    return expect(value, str)


def _devices(value: ArtifactType) -> tuple[DeviceSpec, ...]:
    return tuple(_device(device) for device in expect(value, list))


def _device(value: ArtifactType) -> DeviceSpec:
    obj = expect(value, dict)
    if unknown := sorted(set(obj) - {"type", "id"}):
        raise ValueError(f"unknown device fields {unknown}")
    device_id = expect(obj["id"], int)
    match expect(obj["type"], str):
        case "cpu":
            return DeviceSpec(device_id, "cpu")
        case "gpu":
            return DeviceSpec(device_id, "gpu")
        case other:
            raise ValueError(f"unknown device type {other!r}")


def _sampling(value: ArtifactType) -> EngineSamplingConfig:
    obj = decode_object(value, EngineSamplingConfig)
    return EngineSamplingConfig(
        needs_bitmask_constraints=expect(
            obj["needs_bitmask_constraints"], bool
        ),
        enable_variable_logits=expect(obj["enable_variable_logits"], bool),
        enable_penalties=expect(obj["enable_penalties"], bool),
    )


def _speculative(value: ArtifactType) -> EngineSpeculativeConfig:
    obj = decode_object(value, EngineSpeculativeConfig)
    return EngineSpeculativeConfig(
        speculative_method=expect(obj["speculative_method"], str),
        num_speculative_tokens=expect(obj["num_speculative_tokens"], int),
        draft_proposal=expect(obj["draft_proposal"], str),
        reasoning_parser=decode_optional(_str)(obj["reasoning_parser"]),
    )
