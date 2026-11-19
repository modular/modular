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
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

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

AcceptanceMode: TypeAlias = Literal["greedy", "stochastic"]
"""How a fused speculative graph accepts drafts."""

DraftProposal: TypeAlias = Literal["argmax", "sampled"]
"""How a speculative draft proposes its tokens."""

Proposer: TypeAlias = Literal["block", "sequential"]
"""How a speculative draft proposes: a block of tokens in one forward, or
one token per forward."""

HostInputMode: TypeAlias = Literal["shape", "staged", "pinned"]
"""How a replayed device graph gets a host input it reads every step.

A replay runs no host code, so a host input keeps the value it had at
capture unless the graph reads it through a stable buffer:

- ``shape``: the value follows from the captured shape (rows, query width
  and aligned cache length), as MAX's KV manager stages its host scalars from
  the aligned batch characteristics, so every replay of a key reads what
  capture recorded.
- ``staged``: the graph copies it to the device from one page-locked buffer
  the serving engine rewrites before every launch and replay.
- ``pinned``: the serving engine stages one value for every step while
  capture is on, a bound that holds for every row, as MAX's overlap pipeline
  pins the fused sampler's batch bounds at capture.
"""

CaptureCacheBound: TypeAlias = Literal["aligned", "bucketed"]
"""The host cache bound a captured step bakes in: its cache length aligned
up to a probed one, or the power-of-two bucket of the bound and its query
rows, which the capture key carries."""


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
class EngineParallelConfig:
    """How the graphs split their work across the devices.

    The devices are listed replica-major: replica ``r`` runs on devices
    ``r * tp`` to ``(r + 1) * tp - 1``. Expert, decode-context and
    prefill-context parallelism are carved from the same devices as MAX's
    configuration defines them; nothing here is inferred.

    Args:
        dp: The data-parallel attention replicas.
        tp: The devices each replica's attention splits its heads across.
        ep: The devices the experts are split across, or 1 without expert
            parallelism.
        dcp: The ranks of each replica that split a request's KV cache by
            token position (decode context parallelism), or 1.
        pcp: The ranks of each replica that split a prefill step's query
            rows, prefill context parallelism. Always 1: PCP is not supported
            on the artifact path, whose warmup does not reserve PCP's
            whole-context scratch.
        dcp_interleave: The consecutive token positions a decode context
            rank owns before the next rank's turn; 1 without it.

    Raises:
        ArtifactError: If the degrees don't fit together.
    """

    dp: int
    tp: int
    ep: int
    dcp: int
    pcp: int
    dcp_interleave: int

    def __post_init__(self) -> None:
        if min(self.dp, self.tp, self.ep, self.dcp, self.pcp) < 1:
            raise ArtifactError(
                f"every parallel degree must be positive: {self}"
            )
        if self.tp % self.dcp:
            raise ArtifactError(
                f"the decode context degree {self.dcp} must divide the"
                f" tensor-parallel degree {self.tp}"
            )
        if self.pcp != 1:
            raise ArtifactError(
                "PCP is not supported on the artifact path: the artifact"
                " warmup does not reserve PCP's whole-context scratch, so pcp"
                f" must be 1, got {self.pcp}"
            )
        if self.dcp_interleave < 1 or (
            self.dcp == 1 and self.dcp_interleave != 1
        ):
            raise ArtifactError(
                f"dcp_interleave {self.dcp_interleave} needs a decode context"
                " degree above 1"
            )
        if self.num_devices % self.ep:
            raise ArtifactError(
                f"the expert-parallel degree {self.ep} must divide the"
                f" {self.num_devices} devices"
            )

    @property
    def num_devices(self) -> int:
        """The devices across every replica: ``dp * tp``."""
        return self.dp * self.tp


@dataclass(frozen=True)
class CapturedHostInput:
    """How a replayed device graph gets one host input it reads every step.

    Args:
        mode: How a replay gets the input.
        value: The value a ``pinned`` input holds for every step; ``None``
            for any other mode.

    Raises:
        ArtifactError: If a ``pinned`` input has no value, or another mode
            has one.
    """

    mode: HostInputMode
    value: int | float | None = None

    def __post_init__(self) -> None:
        if (self.mode == "pinned") != (self.value is not None):
            needs = "needs a" if self.mode == "pinned" else "takes no"
            raise ArtifactError(f"a {self.mode} host input {needs} value")


@dataclass(frozen=True)
class EngineCaptureConfig:
    """The device-graph capture the graphs support.

    Args:
        max_batch_size: The most rows per replica a captured decode step
            holds. Every size from 1 to it is captured at every verify
            width. 0 records no capture shapes, and an engine serving the
            artifact runs every step eagerly; which steps it can capture it
            reads off the graphs' inputs, not from this record.
        prefill_max_tokens: The most tokens per replica a captured prefill
            step holds, or 0 for no prefill capture. The engine derives the
            token grid and row buckets from it.
        host_inputs: How a replay gets each host input the serving graphs
            read every step, keyed ``<group kind>.<field>``. A serving engine
            refuses to capture a graph that reads a host input listed here
            under no mode.
        cache_bound: The host cache bound a replayed step keeps, for graphs
            whose kernels size their launches from it. ``aligned``: the
            step's cache length aligned up to a probed one, as MAX Serve's
            capture runner stages it. ``bucketed``: the next power of two
            over the bound plus the step's query rows, at least 256, less
            those rows and capped at the context limit, with the bucket in
            the capture key, so a replay keeps the launch arm eager takes
            at the bound.
    """

    max_batch_size: int
    prefill_max_tokens: int
    host_inputs: Mapping[str, CapturedHostInput] = field(default_factory=dict)
    cache_bound: CaptureCacheBound = "aligned"

    def __post_init__(self) -> None:
        if self.max_batch_size < 0 or self.prefill_max_tokens < 0:
            raise ArtifactError(f"capture limits must not be negative: {self}")
        if self.prefill_max_tokens and not self.max_batch_size:
            raise ArtifactError(
                "prefill capture needs an architecture that captures its"
                " steps, but max_batch_size is 0"
            )
        for name in self.host_inputs:
            kind, _, leaf = name.partition(".")
            if not kind or not leaf or "." in leaf:
                raise ArtifactError(
                    f"capture.host_inputs key {name!r} is not"
                    " <group kind>.<field>"
                )


@dataclass(frozen=True)
class EngineSpeculativeConfig:
    """The speculative-decoding settings a fused graph was built with.

    Args:
        speculative_method: The speculative-decoding method, such as ``mtp``,
            for logs only: what a serving engine needs of the method is in
            the other fields.
        num_speculative_tokens: The number of tokens drafted per step.
        draft_proposal: How the draft proposes tokens: ``argmax``, or
            ``sampled``, which adds the draft's probabilities to the graph.
        acceptance_mode: How the graph accepts drafts.
        reasoning_parser: The reasoning parser, if any. With one, a block
            draft's graph takes whether each request is still thinking.
        draft_query_len: The query rows one draft forward reads per request:
            1 for a sequential proposer, the drafted tokens for a block one.
        proposer: Whether the draft proposes a block of tokens in one
            forward or one token per forward.
        verify_widths: Every number of drafts a decode step may verify,
            ascending, from the pipeline's width policy, with the mixed
            step's width when it verifies any. 0 is among them when a step
            can skip its drafter.
        mixed_verify_width: The most drafts a mixed prefill and verify
            step verifies per decode row: 0 when mixed steps verify none.
            ``None`` when the graph takes no prefill rows on a verify step.
        verify_takes_prefill_rows: Whether the verify commits a prefill row
            that rides a verify step.
        verify_commits_long_rows: Whether the verify commits a row longer
            than one token and its drafts in every recurrent leaf.
        successor_replay: Whether an engine replays a cached prompt
            row's boundary token for the draft head.
    """

    speculative_method: str
    num_speculative_tokens: int
    draft_proposal: DraftProposal
    acceptance_mode: AcceptanceMode
    reasoning_parser: str | None
    draft_query_len: int
    proposer: Proposer
    verify_widths: tuple[int, ...]
    mixed_verify_width: int | None
    verify_takes_prefill_rows: bool = False
    verify_commits_long_rows: bool = False
    successor_replay: bool = False

    def __post_init__(self) -> None:
        widths = self.verify_widths
        if not widths or list(widths) != sorted(set(widths)) or widths[0] < 0:
            raise ArtifactError(
                "verify_widths must be distinct widths, ascending, with one"
                f" above 0, got {list(widths)}"
            )
        if widths[-1] < 1:
            raise ArtifactError(
                f"verify_widths must hold a width above 0, got {list(widths)}"
            )
        mixed = self.mixed_verify_width
        if mixed is not None and not 0 <= mixed <= self.num_speculative_tokens:
            raise ArtifactError(
                f"mixed_verify_width {mixed} must be within 0 and the"
                f" {self.num_speculative_tokens} drafted tokens"
            )
        if mixed is not None and not self.verify_takes_prefill_rows:
            raise ArtifactError(
                "mixed_verify_width needs a graph that takes prefill rows on"
                " a verify step"
            )
        if widths[-1] > self.num_speculative_tokens:
            raise ArtifactError(
                f"verify width {widths[-1]} exceeds the"
                f" {self.num_speculative_tokens} drafted tokens"
            )
        if self.draft_query_len < 1:
            raise ArtifactError(
                f"draft_query_len must be positive, got {self.draft_query_len}"
            )


@dataclass(frozen=True)
class EnginePrefillContextParallel:
    """When a prefill step splits its rows across a DCP group's ranks.

    The graphs take the prefill context parallel (PCP) path for a step whose
    rows average more than ``min_rows_per_request`` per request, number at
    least ``min_step_tokens``, and sum to at most ``max_batch_context``
    page-aligned context tokens per replica. Every other step, decode and
    verify among them, takes the decode context parallel path.

    Args:
        min_step_tokens: The fewest rows a PCP step has.
        min_rows_per_request: The rows per request a PCP step averages more
            than.
        max_batch_context: The most page-aligned summed context a PCP step
            has on one replica.
    """

    min_step_tokens: int
    min_rows_per_request: int
    max_batch_context: int


@dataclass(frozen=True)
class SpecGraphFacts:
    """The verify facts of a built fused speculative graph.

    Args:
        verify_takes_prefill_rows: Whether the verify commits a prefill row
            that rides a verify step.
        verify_commits_long_rows: Whether the verify commits a row longer
            than one token and its drafts in every recurrent leaf.
        successor_replay: Whether an engine replays a cached prompt
            row's boundary token for the draft head.
    """

    verify_takes_prefill_rows: bool
    verify_commits_long_rows: bool
    successor_replay: bool


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
        max_batch_input_tokens: The compiled input-token capacity of one
            step, per data-parallel replica.
        prefill_token_budget: The prefill tokens the scheduler gives one
            step, per replica, excluding draft rows, or ``None`` to derive
            it from ``max_batch_input_tokens``.
        max_batch_total_tokens: The most context tokens one batch holds, or
            ``None`` for no limit.
        devices: The devices the graphs run on, replica-major.
        parallel: How the graphs split their work across the devices.
        capture: The device-graph capture the graphs support.
        sampling: The sampling graphs' optional inputs.
        speculative: The speculative-decoding settings, or ``None`` for a
            pipeline that doesn't decode speculatively.
        prefix_caching: Whether the architecture serves prefix caching:
            ``False`` where MAX Serve turns it off for the architecture, and
            an engine then serves without it.
        prefill_context_parallel: When a prefill step takes the PCP path,
            or ``None`` for graphs without one. Recorded exactly when
            ``parallel.pcp`` is above 1.

    Raises:
        ArtifactError: If the devices are not ``dp * tp``, the capture
            exceeds the batch size, the prefill budget or prefill capture
            exceeds what a step can take, or the PCP record and
            ``parallel.pcp`` disagree.
    """

    max_batch_size: int
    max_length: int
    max_batch_input_tokens: int
    prefill_token_budget: int | None
    max_batch_total_tokens: int | None
    devices: tuple[DeviceSpec, ...]
    parallel: EngineParallelConfig
    capture: EngineCaptureConfig
    sampling: EngineSamplingConfig
    speculative: EngineSpeculativeConfig | None
    prefix_caching: bool
    prefill_context_parallel: EnginePrefillContextParallel | None = None

    def __post_init__(self) -> None:
        if len(self.devices) != self.parallel.num_devices:
            raise ArtifactError(
                f"{len(self.devices)} devices for dp {self.parallel.dp} x tp"
                f" {self.parallel.tp}"
            )
        budget = self.prefill_token_budget
        if budget is not None and not 0 < budget <= self.max_batch_input_tokens:
            raise ArtifactError(
                f"prefill token budget {budget} must be positive and no"
                " larger than the compiled input capacity"
                f" {self.max_batch_input_tokens}"
            )
        if self.capture.max_batch_size > self.max_batch_size:
            raise ArtifactError(
                f"capture of {self.capture.max_batch_size} rows exceeds the"
                f" batch size {self.max_batch_size}"
            )
        if self.capture.prefill_max_tokens > self.prefill_tokens:
            raise ArtifactError(
                f"prefill capture of {self.capture.prefill_max_tokens} tokens"
                f" exceeds the prefill token budget {self.prefill_tokens}"
            )
        if (self.prefill_context_parallel is not None) != (
            self.parallel.pcp > 1
        ):
            raise ArtifactError(
                "a prefill_context_parallel record comes with parallel pcp"
                f" above 1 and only with it: pcp {self.parallel.pcp}, record"
                f" {self.prefill_context_parallel}"
            )

    @property
    def prefill_tokens(self) -> int:
        """The prefill tokens one step takes per replica.

        The prefill token budget, or the compiled input capacity without one.
        """
        if self.prefill_token_budget is None:
            return self.max_batch_input_tokens
        return self.prefill_token_budget

    def __serialize__(self) -> Mapping[str, ArtifactType]:
        """Returns the settings as a JSON object."""
        speculative = self.speculative
        parallel = self.parallel
        return {
            "max_batch_size": self.max_batch_size,
            "max_length": self.max_length,
            "max_batch_input_tokens": self.max_batch_input_tokens,
            "prefill_token_budget": self.prefill_token_budget,
            "max_batch_total_tokens": self.max_batch_total_tokens,
            "devices": [
                {"type": device.device_type, "id": device.id}
                for device in self.devices
            ],
            "parallel": {
                "dp": parallel.dp,
                "tp": parallel.tp,
                "ep": parallel.ep,
                "dcp": parallel.dcp,
                "pcp": parallel.pcp,
                "dcp_interleave": parallel.dcp_interleave,
            },
            "capture": {
                "max_batch_size": self.capture.max_batch_size,
                "prefill_max_tokens": self.capture.prefill_max_tokens,
                "host_inputs": {
                    name: {"mode": how.mode, "value": how.value}
                    for name, how in sorted(self.capture.host_inputs.items())
                },
                "cache_bound": self.capture.cache_bound,
            },
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
                "acceptance_mode": speculative.acceptance_mode,
                "reasoning_parser": speculative.reasoning_parser,
                "draft_query_len": speculative.draft_query_len,
                "proposer": speculative.proposer,
                "verify_widths": list(speculative.verify_widths),
                "mixed_verify_width": speculative.mixed_verify_width,
                "verify_takes_prefill_rows": (
                    speculative.verify_takes_prefill_rows
                ),
                "verify_commits_long_rows": (
                    speculative.verify_commits_long_rows
                ),
                "successor_replay": speculative.successor_replay,
            },
            "prefill_context_parallel": None
            if (pcp := self.prefill_context_parallel) is None
            else {
                "min_step_tokens": pcp.min_step_tokens,
                "min_rows_per_request": pcp.min_rows_per_request,
                "max_batch_context": pcp.max_batch_context,
            },
            "prefix_caching": self.prefix_caching,
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
            prefill_token_budget=decode_field(
                value, "prefill_token_budget", decode_optional(_int)
            ),
            max_batch_total_tokens=decode_field(
                value, "max_batch_total_tokens", decode_optional(_int)
            ),
            devices=decode_field(value, "devices", _devices),
            parallel=decode_field(value, "parallel", _parallel),
            capture=decode_field(value, "capture", _capture),
            sampling=decode_field(value, "sampling", _sampling),
            speculative=decode_field(
                value, "speculative", decode_optional(_speculative)
            ),
            prefill_context_parallel=decode_field(
                value,
                "prefill_context_parallel",
                decode_optional(_prefill_context_parallel),
            ),
            prefix_caching=decode_field(value, "prefix_caching", _bool),
        )


def _int(value: ArtifactType) -> int:
    return expect(value, int)


def _str(value: ArtifactType) -> str:
    return expect(value, str)


def _bool(value: ArtifactType) -> bool:
    return expect(value, bool)


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


def _parallel(value: ArtifactType) -> EngineParallelConfig:
    obj = decode_object(value, EngineParallelConfig)
    return EngineParallelConfig(
        dp=_int(obj["dp"]),
        tp=_int(obj["tp"]),
        ep=_int(obj["ep"]),
        dcp=_int(obj["dcp"]),
        pcp=_int(obj["pcp"]),
        dcp_interleave=_int(obj["dcp_interleave"]),
    )


def _capture(value: ArtifactType) -> EngineCaptureConfig:
    obj = decode_object(value, EngineCaptureConfig)
    return EngineCaptureConfig(
        max_batch_size=_int(obj["max_batch_size"]),
        prefill_max_tokens=_int(obj["prefill_max_tokens"]),
        host_inputs={
            name: _host_input(how)
            for name, how in expect(obj["host_inputs"], dict).items()
        },
        cache_bound=_cache_bound(obj["cache_bound"]),
    )


def _cache_bound(value: ArtifactType) -> CaptureCacheBound:
    match expect(value, str):
        case "aligned":
            return "aligned"
        case "bucketed":
            return "bucketed"
        case other:
            raise ValueError(f"unknown capture cache bound {other!r}")


def _host_input(value: ArtifactType) -> CapturedHostInput:
    obj = decode_object(value, CapturedHostInput)
    mode = expect(obj["mode"], str)
    if mode not in ("shape", "staged", "pinned"):
        raise ValueError(f"unknown host input mode {mode!r}")
    pinned = obj["value"]
    if pinned is not None and (
        isinstance(pinned, bool) or not isinstance(pinned, int | float)
    ):
        raise ValueError(f"host input value {pinned!r} is not a number")
    return CapturedHostInput(mode=mode, value=pinned)  # type: ignore[arg-type]


def _sampling(value: ArtifactType) -> EngineSamplingConfig:
    obj = decode_object(value, EngineSamplingConfig)
    return EngineSamplingConfig(
        needs_bitmask_constraints=_bool(obj["needs_bitmask_constraints"]),
        enable_variable_logits=_bool(obj["enable_variable_logits"]),
        enable_penalties=_bool(obj["enable_penalties"]),
    )


def _acceptance_mode(value: ArtifactType) -> AcceptanceMode:
    match expect(value, str):
        case "greedy":
            return "greedy"
        case "stochastic":
            return "stochastic"
        case other:
            raise ValueError(f"unknown acceptance mode {other!r}")


def _draft_proposal(value: ArtifactType) -> DraftProposal:
    match expect(value, str):
        case "argmax":
            return "argmax"
        case "sampled":
            return "sampled"
        case other:
            raise ValueError(f"unknown draft proposal {other!r}")


def _proposer(value: ArtifactType) -> Proposer:
    match expect(value, str):
        case "block":
            return "block"
        case "sequential":
            return "sequential"
        case other:
            raise ValueError(f"unknown proposer {other!r}")


def _speculative(value: ArtifactType) -> EngineSpeculativeConfig:
    obj = decode_object(value, EngineSpeculativeConfig)
    return EngineSpeculativeConfig(
        speculative_method=_str(obj["speculative_method"]),
        num_speculative_tokens=_int(obj["num_speculative_tokens"]),
        draft_proposal=_draft_proposal(obj["draft_proposal"]),
        acceptance_mode=_acceptance_mode(obj["acceptance_mode"]),
        reasoning_parser=decode_optional(_str)(obj["reasoning_parser"]),
        draft_query_len=_int(obj["draft_query_len"]),
        proposer=_proposer(obj["proposer"]),
        verify_widths=tuple(
            _int(width) for width in expect(obj["verify_widths"], list)
        ),
        mixed_verify_width=decode_optional(_int)(obj["mixed_verify_width"]),
        verify_takes_prefill_rows=_bool(obj["verify_takes_prefill_rows"]),
        verify_commits_long_rows=_bool(obj["verify_commits_long_rows"]),
        successor_replay=_bool(obj["successor_replay"]),
    )


def _prefill_context_parallel(
    value: ArtifactType,
) -> EnginePrefillContextParallel:
    obj = decode_object(value, EnginePrefillContextParallel)
    return EnginePrefillContextParallel(
        min_step_tokens=expect(obj["min_step_tokens"], int),
        min_rows_per_request=expect(obj["min_rows_per_request"], int),
        max_batch_context=expect(obj["max_batch_context"], int),
    )
