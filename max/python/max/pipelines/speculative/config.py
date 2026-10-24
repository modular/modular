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
"""Configuration types for MAX speculative decoding.

Exposes :class:`SpeculativeConfig`, which controls the speculative decoding
method, the number of draft tokens per step, and how the target verifies
drafts.
"""

from __future__ import annotations

from typing import Literal

from max.config import ConfigFileModel
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from .adaptive_width import parse_adaptive_widths
from .depth_schedule import DepthScheduleEntry, normalize_depth_schedule

__all__ = [
    "MAGIC_DRAFT_TOKEN_ID",
    "SpeculativeConfig",
    "SpeculativeMethod",
    "VerifyWidthRange",
]

MAGIC_DRAFT_TOKEN_ID = 42
"""Sentinel draft-token id for prefill / dummy-draft graph-capture steps.

A row of ``draft_tokens`` whose every position equals this value means "no
real draft prediction to verify": the unified DFlash graphs detect it to zero
out acceptance during prefill, and the overlap pipeline writes it when seeding
draft slots before any real draft exists. Defined here so the graph side
(``architectures``) and the runtime side (``lib``) agree on a single value.
"""

SpeculativeMethod = Literal["eagle", "mtp", "dflash", "dflash2", "dspark"]
"""The supported methods for speculative decoding."""

_ONE_TOKEN_PER_STEP: tuple[SpeculativeMethod, ...] = ("eagle", "mtp")
"""Methods that draft one token per step, and so default to a width of 2."""


class VerifyWidthRange(BaseModel):
    """One inclusive decode-batch-size range and the drafts to verify in it."""

    batch_start: int = Field(
        description="First decode batch size in the range, inclusive."
    )
    """First decode batch size this range covers, inclusive."""

    batch_end: int = Field(
        description="Last decode batch size in the range, inclusive."
    )
    """Last decode batch size this range covers, inclusive."""

    num_tokens: int = Field(
        description=(
            "How many of the carried drafts the target verifies at these "
            "batch sizes."
        )
    )
    """How many of the carried drafts the target verifies in this range."""


class SpeculativeConfig(ConfigFileModel):
    """Configures speculative decoding for a pipeline.

    Speculative decoding accelerates token generation by having a small
    draft step propose several candidate tokens that the larger target
    verifies in one forward pass. This class selects the method
    (:attr:`speculative_method`), how many tokens to draft per step
    (:attr:`num_speculative_tokens`), and the knobs that decide how the
    target verifies them (:attr:`synthetic_acceptance_rate`,
    :attr:`use_greedy_acceptance`).

    The CLI surfaces these fields as ``--speculative-method``,
    ``--num-speculative-tokens``,
    ``--num-speculative-tokens-per-batch-size``,
    ``--adaptive-speculative-widths``,
    ``--adaptive-speculative-min-batch-size``,
    ``--num-speculative-tokens-mixed-batch``, and
    ``--synthetic-acceptance-rate``.
    Construct the config directly when configuring a pipeline
    programmatically:

    .. code-block:: python

        from max.pipelines.speculative import SpeculativeConfig

        spec = SpeculativeConfig(
            speculative_method="eagle",
            num_speculative_tokens=3,
        )

    Instances are immutable. Assigning a field after construction raises.
    """

    model_config = ConfigDict(frozen=True)

    speculative_method: SpeculativeMethod | None = Field(
        default=None, description="The speculative decoding method to use."
    )
    """The speculative decoding method to use.

    One of ``"eagle"``, ``"mtp"``, ``"dflash"``, ``"dflash2"``, or
    ``"dspark"``. When ``None``, speculative decoding is disabled.
    """

    num_speculative_tokens: int | None = Field(
        default=None,
        # So the default below runs when the field is unset.
        validate_default=True,
        description=(
            "The number of speculative tokens. Unset selects a per-method "
            "default: 2 for ``eagle``/``mtp``, and the draft checkpoint's "
            "trained width for ``dflash`` and ``dspark``."
        ),
    )
    """The number of tokens the draft proposes per verification pass.

    ``None`` means unset: ``eagle`` and ``mtp`` resolve it to ``2`` at
    construction, while ``dflash``-style block drafts leave it for the
    architecture to resolve from the draft checkpoint's trained width.
    Larger values can raise the average draft acceptance length and peak
    speedup, but they may hurt acceptance rates at later positions and
    increase kernel latencies from the additional tokens.
    """

    @field_validator("num_speculative_tokens", mode="after")
    @classmethod
    def _resolve_autoregressive_draft_width(
        cls, value: int | None, info: ValidationInfo
    ) -> int | None:
        # DFlash leaves it unset for the architecture to fill.
        method = info.data.get("speculative_method")
        if value is None and method in _ONE_TOKEN_PER_STEP:
            return 2
        return value

    num_speculative_tokens_per_batch_size: list[VerifyWidthRange] | None = (
        Field(
            default=None,
            description=(
                "Batch-size schedule for how many drafted tokens to verify, as "
                "inclusive ranges. For example "
                '\'[{"batch_start": 1, "batch_end": 16, "num_tokens": 3}, '
                '{"batch_start": 17, "batch_end": 64, "num_tokens": 1}]\'. '
                "Unset verifies every drafted token."
            ),
        )
    )
    """How many of the drafted tokens the target verifies, by decode batch size.

    A step always drafts :attr:`num_speculative_tokens` proposals; this narrows
    how many of them the target checks.

    Ranges are inclusive on both ends. The first must start at batch size 1 so
    every runtime batch size resolves to a count; gaps and the tail past the
    final range carry the previous count forward, and every count is capped at
    :attr:`num_speculative_tokens` since a step cannot verify more drafts than
    it carries. ``None`` verifies every drafted token, which is the behavior
    when the field is unset.

    Applies to every speculative method. A block drafter (``dflash``, ``dspark``) still
    drafts its whole checkpoint-fixed block, and narrowing changes only how
    much of that block the target verifies. The exception is a count of
    ``0``, where a draft that declares ``supports_zero_draft_rows`` skips the
    draft forward too.
    """

    @property
    def verify_width_schedule(self) -> list[DepthScheduleEntry] | None:
        """The schedule as validated, sorted ``(start, end, count)`` triples.

        ``None`` when no schedule was configured.
        """
        ranges = self.num_speculative_tokens_per_batch_size
        if ranges is None:
            return None
        return [(r.batch_start, r.batch_end, r.num_tokens) for r in ranges]

    @field_validator("num_speculative_tokens_per_batch_size", mode="after")
    @classmethod
    def _validate_verify_width_schedule(
        cls, ranges: list[VerifyWidthRange] | None
    ) -> list[VerifyWidthRange] | None:
        if ranges is None:
            return None
        # Validating here rather than at pipeline build means a malformed
        # schedule fails while the config is being read, next to the value that
        # caused it, instead of minutes into a model load.
        normalized = normalize_depth_schedule(
            [(r.batch_start, r.batch_end, r.num_tokens) for r in ranges]
        )
        return [
            VerifyWidthRange(batch_start=start, batch_end=end, num_tokens=count)
            for start, end, count in normalized
        ]

    adaptive_speculative_widths: str | None = Field(
        default=None,
        description=(
            "Verify widths to choose among at runtime by measured decode "
            "tokens per second, as a comma-separated list such as "
            '"1,3,5", or "all" for every width from 1 to '
            "num_speculative_tokens. Unset keeps the width static. Cannot be "
            "combined with num_speculative_tokens_per_batch_size."
        ),
    )
    """The verify widths a throughput-driven controller chooses among.

    ``None`` keeps the width static. Cannot be combined with
    :attr:`num_speculative_tokens_per_batch_size`. Startup cost scales with
    how many widths are named.
    """

    adaptive_speculative_min_batch_size: int = Field(
        default=1,
        ge=1,
        description=(
            "Smallest decode batch size that captures every adaptive verify "
            "width. Smaller batches capture only the widest and round a "
            "narrower request up to it. Requires adaptive_speculative_widths."
        ),
    )
    """Decode batch size below which only the widest adaptive width is captured.

    Small batches are memory-bound, so narrowing barely changes their step cost
    while each extra width multiplies the captured graphs. Verifying more
    drafts than asked is always correct, since every step drafts them all.
    """

    @model_validator(mode="after")
    def _validate_adaptive_speculative_widths(self) -> Self:
        if self.adaptive_speculative_widths is None:
            if self.adaptive_speculative_min_batch_size > 1:
                raise ValueError(
                    "adaptive_speculative_min_batch_size requires "
                    "adaptive_speculative_widths."
                )
            return self
        if self.num_speculative_tokens_per_batch_size is not None:
            raise ValueError(
                "adaptive_speculative_widths and "
                "num_speculative_tokens_per_batch_size both choose the verify "
                "width; set only one."
            )
        # A block draft reads its ceiling from the checkpoint later, so only the
        # syntax is checked for it here; the pipeline checks the ceiling.
        parse_adaptive_widths(
            self.adaptive_speculative_widths, self.num_speculative_tokens
        )
        return self

    num_speculative_tokens_mixed_batch: int | None = Field(
        default=None,
        ge=0,
        description=(
            "How many drafted tokens the target verifies on a mixed "
            "prefill+decode batch. Unset uses the count a pure decode batch "
            "of the same size would."
        ),
    )
    """How many of the drafted tokens the target verifies on a mixed batch.

    Narrows for a different reason than
    :attr:`num_speculative_tokens_per_batch_size`: that schedule trades
    acceptance for decode throughput at large batch sizes, while this trades it
    for the latency of the prefill rows sharing the step, which sit on the
    critical path of a prompt's time to first token.

    Capped at :attr:`num_speculative_tokens` where it is read, not here:
    a block draft leaves that width for the architecture to resolve from the
    draft checkpoint, so the ceiling is not yet known at config-validation
    time.

    ``None`` leaves mixed batches on the batch-size schedule, which is the
    behavior when the field is unset. Reachable only alongside
    ``--enable-spec-decode-mixed-batches``; without it a mixed batch verifies
    nothing at all.
    """

    synthetic_acceptance_rate: float | None = Field(
        default=None,
        description=(
            "Synthetic acceptance rate for benchmarking (``0.0`` to ``1.0``). "
            "When set, the rejection sampler bypasses the real "
            "draft/target comparison and accepts each draft position "
            "with a calibrated probability so the mean joint acceptance "
            "across ``num_speculative_tokens`` positions matches this value."
        ),
    )
    """A benchmarking-only override that accepts drafts with a calibrated
    probability, ignoring real logits.

    Must be between ``0.0`` and ``1.0``. When set, each draft position is
    accepted with a probability calibrated so that the mean joint
    acceptance across :attr:`num_speculative_tokens` positions matches this
    value. Use it to model hypothetical speedups without changing the draft
    model; leave unset for real serving.
    """

    @field_validator("synthetic_acceptance_rate")
    @classmethod
    def _validate_synthetic_acceptance_rate(
        cls, v: float | None
    ) -> float | None:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(
                "synthetic_acceptance_rate must be between 0.0 and 1.0,"
                f" got {v}"
            )
        return v

    use_relaxed_acceptance_for_thinking: bool = Field(
        default=False,
        description=(
            "Enables relaxed acceptance for speculative decoding "
            "draft positions inside a ``<think>...</think>`` block. The "
            "target's top-N candidates (filtered by a probability "
            "threshold ``top1_prob - relaxed_delta``) are compared "
            "against the draft token; matching any candidate accepts "
            "the draft. Outside the thinking span, the existing strict "
            "acceptance rule still applies. Requires "
            "``draft_proposal='argmax'``."
        ),
    )

    relaxed_topk: int = Field(
        default=10,
        description=(
            "Top-N candidates from the target distribution to consider "
            "when relaxed acceptance is active. Ignored when "
            "``use_relaxed_acceptance_for_thinking`` is ``False``."
        ),
    )

    relaxed_delta: float = Field(
        default=0.6,
        description=(
            "Probability gap below the top-1 candidate inside which "
            "candidates remain eligible for relaxed acceptance. A draft "
            "token is accepted if it matches any top-N candidate whose "
            "probability is at least ``top1_prob - relaxed_delta``. "
            "Ignored when ``use_relaxed_acceptance_for_thinking`` is "
            "``False``."
        ),
    )

    use_greedy_acceptance: bool = Field(
        default=False,
        description=(
            "Use greedy (argmax) draft acceptance instead of the stochastic "
            "sampler. The greedy path has no mid-graph allocation, so the "
            "fused speculative graph can be CUDA-graph captured. Valid only "
            "for greedy serving (temperature 0, top_k 1); incompatible with "
            "relaxed and synthetic acceptance."
        ),
    )

    draft_proposal: Literal["argmax", "sampled"] = Field(
        default="argmax",
        description=(
            "How the draft model proposes tokens. 'argmax' (default) "
            "proposes deterministically. 'sampled' makes the draft sample "
            "its own proposal and keep the distribution it drew from, so "
            "verification runs true speculative sampling instead of "
            "typical acceptance. Incompatible with "
            "``use_relaxed_acceptance_for_thinking``. An architecture that "
            "cannot sample its draft refuses 'sampled' at startup."
        ),
    )

    draft_head_parallelism: Literal["device0", "replica"] = Field(
        default="device0",
        description=(
            "Where a data-parallel block draft runs its sequential head. "
            "'device0' (default) runs it on device 0 over the whole batch. "
            "'replica' runs it on each replica's own device over that "
            "replica's rows, trading extra device memory for a shorter "
            "draft step. Has no effect without data parallelism or on an "
            "architecture whose draft head is not split."
        ),
    )

    draft_probs_layout: Literal["device0", "replica"] = Field(
        default="device0",
        description=(
            "Where the sampled draft's proposal distributions live between "
            "steps. 'device0' (default) keeps one ``[batch, K, vocab]`` "
            "tensor on device 0. 'replica' keeps each data-parallel "
            "replica's rows on its own device and verifies them there, "
            "which drops the allgather onto device 0 and the verify's "
            "single-device cost. Requires ``draft_proposal='sampled'`` and "
            "``draft_head_parallelism='replica'``; the engine feeding the "
            "graph has to bind one distribution per replica. MAX Serve's "
            "speculative pipeline does not."
        ),
    )

    logit_penalties: bool = Field(
        default=False,
        description=(
            "Declares presence/frequency penalty inputs on the fused "
            "speculative graph and applies them to the verified logits "
            "before acceptance. For engines that feed those inputs (Mach); "
            "MAX Serve's speculative pipeline does not, and a block draft "
            "refuses them."
        ),
    )

    @model_validator(mode="after")
    def _validate_draft_proposal(self) -> Self:
        if (
            self.draft_proposal == "sampled"
            and self.use_relaxed_acceptance_for_thinking
        ):
            raise ValueError(
                "draft_proposal='sampled' cannot be combined with"
                " use_relaxed_acceptance_for_thinking: relaxed acceptance"
                " takes the drafted token to be the draft's argmax, which a"
                " sampled proposal does not guarantee"
            )
        if self.draft_probs_layout == "replica" and (
            self.draft_proposal != "sampled"
            or self.draft_head_parallelism != "replica"
        ):
            raise ValueError(
                "draft_probs_layout='replica' needs draft_proposal='sampled'"
                " and draft_head_parallelism='replica': the distributions"
                " are drawn on each replica's device and stay there"
            )
        return self

    @field_validator("relaxed_topk")
    @classmethod
    def _validate_relaxed_topk(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"relaxed_topk must be >= 1, got {v}")
        return v

    @field_validator("relaxed_delta")
    @classmethod
    def _validate_relaxed_delta(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"relaxed_delta must be between 0.0 and 1.0, got {v}"
            )
        return v

    _config_file_section_name: str = "speculative_config"

    @property
    def draft_width(self) -> int:
        """The number of tokens drafted per step.

        Set for every config the pipeline builds: the architecture supplies
        it for checkpoints that fix it, and the rest take the default.
        """
        assert self.num_speculative_tokens is not None, (
            "num_speculative_tokens is unset; the config was not built by"
            " PipelineConfig.from_args()."
        )
        return self.num_speculative_tokens

    def is_eagle(self) -> bool:
        """Returns whether the configured method is EAGLE.

        EAGLE drafts share the target's embedding and ``lm_head`` layers
        and read the target's hidden states.
        """
        return self.speculative_method == "eagle"

    def is_mtp(self) -> bool:
        """Returns whether the configured method is multi-token prediction (MTP)."""
        return self.speculative_method == "mtp"

    def is_dflash(self) -> bool:
        """Returns whether the configured method is a DFlash block draft.

        True for both ``"dflash"`` and ``"dflash2"``: v2 keeps v1's fused
        graph shape and block-drafting contract. Each fused graph is built
        for one drafter, though, so the v1 and Eagle arms in ``lib.config``
        exclude v2 and a v2 target gets its own arm. Use :meth:`is_dflash2`
        where they differ, and :meth:`is_block_draft` where only "the draft
        arrives a block at a time" matters.
        """
        return self.speculative_method in ("dflash", "dflash2")

    def is_dflash2(self) -> bool:
        """Returns whether the configured method is DFlash2 specifically."""
        return self.speculative_method == "dflash2"

    def is_dspark(self) -> bool:
        """Returns whether the configured method is a DSpark block draft.

        DSpark drafts a block at a time like DFlash, but its drafts run a
        different step, so it is a method of its own rather than a DFlash
        variant.
        """
        return self.speculative_method == "dspark"

    def is_block_draft(self) -> bool:
        """Returns whether the draft proposes a block of tokens per step.

        True for every method outside the one-token-per-step methods
        (``"eagle"``, ``"mtp"``): ``"dflash"``, ``"dflash2"`` and
        ``"dspark"``. Pipeline-class selection and KV cache sizing branch on
        this; selecting a fused graph branches on the method itself.
        """
        return (
            self.speculative_method is not None
            and self.speculative_method not in _ONE_TOKEN_PER_STEP
        )
