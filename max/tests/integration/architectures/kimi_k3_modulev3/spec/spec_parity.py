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
"""Greedy parity and determinism harness for Kimi K3 (ModuleV3) + DSpark.

Serves the same checkpoint twice through the offline pipeline API -- first
with DSpark speculative decoding, then plainly -- and reports, per prompt,
whether the two greedy token streams are identical, the first divergence,
the token ids around it, and the plain run's top-2 logit gap there. A
divergence whose gap is a few bfloat16 ulps of the top logit is a rounding
near-tie, not a wiring bug; anything larger is the state the verify and
rollback landed on, or the drafter itself.

The report also carries what proves speculation actually engaged: the
pipeline's own draft-token counts, which must show drafts proposed and
(unless the drafter never agrees with the target) accepted -- greedy parity
holds for a pipeline whose draft path silently never ran, so parity alone
cannot prove the pairing works. ``teacher_forced`` adds the output-level
oracle: every committed token must be the plain pipeline's argmax at its
own position, true under greedy acceptance whatever the acceptance rate
was.

``repeat N`` reruns the speculative side and requires every run to emit
identical tokens: the KDA rollback must be a pure function of the accepted
count, and the same step must not depend on what a previous batch left in a
pool.

The tests import this module for :func:`run` and the graders it uses.
"""

from __future__ import annotations

import gc
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from max.driver import DeviceSpec
from max.pipelines.architectures import register_all_models
from max.pipelines.lib import (
    PIPELINE_REGISTRY,
    KVCacheConfig,
    MAXModelConfig,
    OverlapTextGenerationPipeline,
    PipelineArgs,
    PipelineConfig,
    PipelineRuntimeConfig,
    _hf_config,
)
from max.pipelines.lib.generate import BatchPipeline, GenerateIterator
from max.pipelines.speculative import SpeculativeConfig

__all__ = [
    "NEAR_TIE_ULPS",
    "Divergence",
    "ParityReport",
    "PromptReport",
    "TeacherForcedMiss",
    "classify",
    "first_divergence",
    "prompt_report",
    "read_prompts",
    "run",
]

_BF16_MANTISSA_BITS = 7
NEAR_TIE_ULPS = 4.0
"""The largest top-2 gap, in bfloat16 ulps of the top logit, that rounding
between two runs of the same model explains. GLM-5.3-Flash's MTP parity
measured at most two."""


@dataclass(frozen=True)
class Divergence:
    """Where the speculative run first left the plain run's greedy choice.

    The logits are the plain run's at the divergence, conditioned on the
    shared prefix, so they are the logits that produced the plain token.
    """

    index: int
    """Position in the generated tokens the runs disagree at."""

    spec_token: int
    plain_token: int
    top_token: int
    plain_top_logit: float
    second_token: int
    second_logit: float
    gap: float
    """``plain_top_logit - second_logit``."""

    gap_ulps: float
    """``gap`` in bfloat16 ulps of the top logit, the near-tie scale."""

    spec_logit: float
    spec_rank: int
    """The spec token's rank in the plain run's logits."""


@dataclass(frozen=True)
class TeacherForcedMiss:
    """One committed token that is not the plain run's argmax there."""

    prompt_index: int
    position: int
    spec_token: int
    argmax_token: int
    gap: float
    """``argmax logit - spec token's logit``."""

    gap_ulps: float


@dataclass(frozen=True)
class PromptReport:
    """One prompt's two greedy streams and where they came apart."""

    index: int
    prompt_length: int
    identical: bool
    """No token divergence anywhere, and the streams are the same length."""

    spec_tokens: tuple[int, ...]
    plain_tokens: tuple[int, ...]
    divergence: Divergence | None = None
    same_length: bool = True


@dataclass(frozen=True)
class ParityReport:
    """The whole harness run's result."""

    repeats: int
    deterministic: bool
    """Whether every speculative rerun emitted identical tokens."""

    drafts_proposed: int
    drafts_accepted: int
    """The pipeline's own draft-token counts, summed over the spec runs."""

    teacher_forced_misses: tuple[TeacherForcedMiss, ...]
    plain_teacher_forced_misses: tuple[TeacherForcedMiss, ...]
    """The same oracle over the plain stream: its own floor, since the
    re-prefill and the decode the stream came from round differently."""
    graded_positions: int
    """How many committed positions the oracle graded."""
    fold_sensitive_graded: int
    """How many graded positions a short fold would corrupt: step bonuses
    past the first decode step, which read a state a fold already landed.
    Zero means the run cannot grade a fold at all."""
    prompts: tuple[PromptReport, ...]

    @property
    def identical(self) -> bool:
        return all(report.identical for report in self.prompts)

    @property
    def near_ties_only(self) -> bool:
        """Whether the teacher-forced misses are all rounding near-ties."""
        return all(
            miss.gap_ulps <= NEAR_TIE_ULPS
            for miss in self.teacher_forced_misses
        )


def read_prompts(path: Path) -> list[list[int] | str]:
    """Returns one prompt per nonempty line: token ids or a text string.

    A line that parses as a JSON array of integers is a token-id prompt;
    every other line is text.
    """
    prompts: list[list[int] | str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ids = json.loads(line)
        except json.JSONDecodeError:
            prompts.append(line)
            continue
        if not isinstance(ids, list) or not all(
            isinstance(token, int) for token in ids
        ):
            raise ValueError(f"{path}: not a JSON array of token ids: {line}")
        prompts.append(ids)
    if not prompts:
        raise ValueError(f"{path}: no prompts")
    return prompts


def first_divergence(
    spec_tokens: Sequence[int], plain_tokens: Sequence[int]
) -> int | None:
    """Returns the first index the two token streams disagree at.

    ``None`` when they agree on the whole shared prefix -- including when
    only their lengths differ, which the caller reports separately so it
    never indexes past the shorter stream's end.
    """
    for index, (spec, plain) in enumerate(
        zip(spec_tokens, plain_tokens, strict=False)
    ):
        if spec != plain:
            return index
    return None


def _bf16_ulps(gap: float, top_logit: float) -> float:
    """Returns ``gap`` in bfloat16 ulps of the top logit."""
    if top_logit == 0.0:
        return gap * (2.0**_BF16_MANTISSA_BITS)
    ulp = 2.0 ** (np.floor(np.log2(abs(top_logit))) - _BF16_MANTISSA_BITS)
    return gap / ulp


def classify(
    logits_row: np.ndarray, spec_token: int, plain_token: int, index: int
) -> Divergence:
    """Grades one divergence against the plain run's logits there."""
    top_token = int(np.argmax(logits_row))
    order = np.argsort(-logits_row)
    second_token = int(order[1] if order[0] == top_token else order[0])
    top_logit = float(logits_row[top_token])
    second_logit = float(logits_row[second_token])
    gap = top_logit - second_logit
    return Divergence(
        index=index,
        spec_token=int(spec_token),
        plain_token=int(plain_token),
        top_token=top_token,
        plain_top_logit=top_logit,
        second_token=second_token,
        second_logit=second_logit,
        gap=gap,
        gap_ulps=_bf16_ulps(gap, top_logit),
        spec_logit=float(logits_row[int(spec_token)]),
        spec_rank=int(np.sum(logits_row > logits_row[int(spec_token)])) + 1,
    )


def prompt_report(
    index: int,
    prompt: list[int] | str,
    prompt_length: int,
    spec: list[int],
    plain: list[int],
    logits_stack: np.ndarray,
) -> PromptReport:
    """Builds one prompt's report, the same path :func:`run` takes.

    ``logits_stack`` is the plain run's ``[num_new_tokens, vocab]`` stack for
    this row, so the divergence's logits row is ``logits_stack[where]``.
    """
    where = first_divergence(spec, plain)
    divergence = None
    if where is not None:
        divergence = classify(
            logits_stack[where], spec[where], plain[where], where
        )
    return PromptReport(
        index=index,
        prompt_length=prompt_length,
        identical=where is None and len(spec) == len(plain),
        spec_tokens=tuple(spec),
        plain_tokens=tuple(plain),
        divergence=divergence,
        same_length=len(spec) == len(plain),
    )


@dataclass
class _DraftCounts:
    """Draft-token counts accumulated over one speculative run."""

    proposed: int = 0
    accepted: int = 0


def _poll_draft_counts(
    pipeline: BatchPipeline,
    counts: _DraftCounts | None,
    last_metrics: object,
) -> object:
    """Accumulates the newest synced batch's draft counts into ``counts``.

    One synced batch's counts per poll; the same object means no new batch
    completed between two polls. Returns the metrics object this poll saw,
    to pass back as ``last_metrics`` on the next one.
    """
    if counts is None or not isinstance(
        pipeline, OverlapTextGenerationPipeline
    ):
        return last_metrics
    metrics = pipeline.batch_spec_decode_metrics()
    if metrics is not None and metrics is not last_metrics:
        counts.proposed += metrics.draft_tokens_generated
        counts.accepted += metrics.draft_tokens_accepted
        return metrics
    return last_metrics


def _generate(
    pipeline: BatchPipeline,
    prompts: Sequence[list[int] | str],
    max_new_tokens: int,
    *,
    logits: bool,
    counts: _DraftCounts | None = None,
    ignore_eos: bool = False,
) -> tuple[
    list[list[int]], list[np.ndarray] | None, list[int], list[list[int]]
]:
    """Runs one greedy batch; returns tokens, logits, prompt lengths, chunks.

    EOS ends rows as the model emits it: keeping the batch past it is not an
    option under speculative decoding, where a step adds several tokens to a
    row and ``ignore_eos`` raises. When the two runs agree they stop at the
    same EOS, so parity still compares whole streams. The single-token
    teacher-forced probes run on the plain pipeline, which may keep the
    batch fixed.

    ``chunks`` is each row's per-execute token counts, in execute order:
    one :meth:`GenerateIterator.step` is one pipeline step, so a chunk ends
    on the token that step's verify chose itself (the bonus) -- the step
    boundaries the target-chosen oracle grades against, whatever the
    acceptance rate or draft-width policy.
    """
    ids = [p for p in prompts if isinstance(p, list)]
    messages = [
        [{"role": "user", "content": p}] for p in prompts if isinstance(p, str)
    ]
    if ids and messages:
        raise ValueError("every prompt line must be ids or every one text")
    last_metrics = None
    chunks: list[list[int]] = [[] for _ in prompts]
    with GenerateIterator(
        pipeline,
        input_ids=ids or None,
        messages=messages or None,
        max_new_tokens=max_new_tokens,
        top_k=1,
        ignore_eos=ignore_eos,
        output_logits=logits,
    ) as batch:
        while not batch.done:
            for row, step_tokens in batch.step().items():
                chunks[row].append(len(step_tokens))
            last_metrics = _poll_draft_counts(pipeline, counts, last_metrics)
        last_metrics = _poll_draft_counts(pipeline, counts, last_metrics)
        lengths = list(batch.prompt_lengths)
        tokens = [
            [int(t) for t in sequence[length:]]
            for sequence, length in zip(batch.sequences, lengths, strict=True)
        ]
        rows = (
            [np.asarray(row, dtype=np.float32) for row in batch.logits]
            if logits
            else None
        )
    return tokens, rows, lengths, chunks


def _step_bonus_positions(
    chunks: list[list[int]],
) -> tuple[list[set[int]], int]:
    """Per row, the positions of the tokens the verify itself chose.

    One execute is one pipeline step: the last token it commits for a row
    is that step's bonus, the target's own choice; the tokens before it in
    the chunk are the drafter's accepted proposals. The final chunk is
    skipped -- EOS or the ``max_new_tokens`` cut can end it on a drafter
    token -- so only the non-final chunks' last positions are graded. This
    reads the run's real step boundaries, so it holds at any synthetic
    rate and any draft-width policy.

    Returns the graded position set per row, and how many of those
    positions sit past the first decode step's bonus: the first verify
    reads the prefill state, which no decode-step fold has touched, so a
    fold short of the accepted block only corrupts bonuses from the third
    chunk on.
    """
    graded: list[set[int]] = []
    fold_sensitive = 0
    for row_chunks in chunks:
        positions = set()
        position = 0
        for index, length in enumerate(row_chunks):
            position += length
            if index < len(row_chunks) - 1:
                positions.add(position - 1)
                if index >= 2:
                    fold_sensitive += 1
        graded.append(positions)
    return graded, fold_sensitive


def _teacher_forced(
    pipeline: BatchPipeline,
    prompts: list[list[int] | str],
    spec_tokens: list[list[int]],
    *,
    target_chosen: list[set[int]] | None = None,
) -> tuple[TeacherForcedMiss, ...]:
    """Returns every committed token that is not the plain argmax there.

    Under greedy acceptance every committed token is the target's own
    choice given its prefix, whatever the acceptance rate was, so this
    oracle grades the output without needing the drafter to ever agree.
    Each position is one single-token batched generate over the committed
    prefix. Text prompts carry no token ids, so they are skipped.

    ``target_chosen`` grades a run whose committed stream mixes the
    drafter's tokens with the target's (a synthetic-acceptance run): only
    the positions the verify chose itself -- the step bonuses
    :func:`_step_bonus_positions` reads off the run -- are the target's own
    argmax given their prefix. The other positions are drafter-chosen and
    miss by construction, so they are not probed at all.
    """
    misses: list[TeacherForcedMiss] = []
    rows = [
        index
        for index, prompt in enumerate(prompts)
        if isinstance(prompt, list) and spec_tokens[index]
    ]
    if not rows:
        return ()
    # Probe to the longest row's depth, not the shortest: a row that
    # ended early (an EOS) must not clip the positions the longer rows
    # still committed -- ``active`` already skips rows a position is past.
    depth = max(len(spec_tokens[index]) for index in rows)
    for position in range(depth):
        active = [
            index
            for index in rows
            if len(spec_tokens[index]) > position
            and (target_chosen is None or position in target_chosen[index])
        ]
        if not active:
            continue
        prefixes: list[list[int]] = []
        for index in active:
            prompt = prompts[index]
            assert isinstance(prompt, list)
            prefixes.append([*prompt, *spec_tokens[index][:position]])
        want = [spec_tokens[index][position] for index in active]
        _, logits, _, _ = _generate(
            pipeline, prefixes, 1, logits=True, ignore_eos=True
        )
        assert logits is not None
        for row, (index, token) in enumerate(zip(active, want, strict=True)):
            logits_row = logits[row][0]
            argmax = int(np.argmax(logits_row))
            if argmax == token:
                continue
            gap = float(logits_row[argmax] - logits_row[token])
            misses.append(
                TeacherForcedMiss(
                    prompt_index=index,
                    position=position,
                    spec_token=token,
                    argmax_token=argmax,
                    gap=gap,
                    gap_ulps=_bf16_ulps(gap, float(logits_row[argmax])),
                )
            )
    return tuple(misses)


def _pipeline(
    target: Path,
    drafter: Path | None,
    devices: int,
    max_length: int | None,
    num_speculative_tokens: int | None,
    synthetic_acceptance: float | None,
    device_graph_capture: bool,
    prefer_module_v3: bool,
    trust_remote_code: bool,
    prompts: list[list[int] | str],
    *,
    speculative: bool,
    kv_cache_format: str | None = None,
) -> BatchPipeline:
    # run() registers the architectures before it gets here; a caller
    # prebuilding a pipeline outside it (the shared plain pipeline) must
    # land after the same registration, and repeated calls are safe.
    register_all_models()
    # A speculative config rewrites the cached HF config in place, so the
    # plain run must not see the spec run's.
    _hf_config._config_cache = {}
    device_specs = [DeviceSpec.accelerator(i) for i in range(devices)]
    speculative_config = (
        SpeculativeConfig(
            speculative_method="dspark",
            num_speculative_tokens=num_speculative_tokens,
            use_greedy_acceptance=synthetic_acceptance is None,
            synthetic_acceptance_rate=synthetic_acceptance,
        )
        if speculative
        else None
    )
    config = PipelineConfig.from_args(
        PipelineArgs(
            model_path=str(target),
            trust_remote_code=trust_remote_code,
            draft_model=(
                MAXModelConfig(
                    model_path=str(drafter), device_specs=device_specs
                )
                if speculative
                else None
            ),
            device_specs=device_specs,
            max_length=max_length,
            kv_cache=KVCacheConfig(
                enable_prefix_caching=False, kv_cache_format=kv_cache_format
            ),
            runtime=PipelineRuntimeConfig(
                max_batch_size=len(prompts),
                device_graph_capture=speculative and device_graph_capture,
                # The plain run reads its logits back, which only a
                # synchronous pipeline copies to host.
                disable_overlap=not speculative,
                prefer_module_v3=prefer_module_v3,
            ),
            speculative=speculative_config,
        )
    )
    _, pipeline = PIPELINE_REGISTRY.retrieve(config)
    assert isinstance(pipeline, OverlapTextGenerationPipeline)
    return pipeline


def run(
    target: Path,
    drafter: Path,
    prompts_path: Path,
    *,
    devices: int = 1,
    max_new_tokens: int = 64,
    repeat: int = 1,
    num_speculative_tokens: int | None = 7,
    synthetic_acceptance: float | None = None,
    max_length: int | None = None,
    device_graph_capture: bool = False,
    prefer_module_v3: bool = True,
    trust_remote_code: bool = False,
    teacher_forced: bool = False,
    plain_pipeline: BatchPipeline | None = None,
    kv_cache_format: str | None = None,
) -> ParityReport:
    """Serves the pair speculatively and plainly, and compares the two.

    Args:
        target: The K3 checkpoint directory.
        drafter: The DSpark drafter checkpoint directory.
        prompts_path: One prompt per line, token ids or text.
        devices: GPUs to shard the pair over.
        max_new_tokens: Tokens each run generates per prompt.
        repeat: Speculative runs to make and require identical.
        num_speculative_tokens: Draft width. Seven, the DSpark checkpoint's
            trained width; ``None`` leaves it to the architecture.
        synthetic_acceptance: MAX's synthetic rate, or ``None`` for real
            acceptance.
        max_length: The context cap, or ``None`` for the checkpoint's.
        device_graph_capture: Whether the speculative side captures device
            graphs.
        prefer_module_v3: Whether the registry picks the ModuleV3 target.
        trust_remote_code: Whether the checkpoints' remote-code configs may
            load, which the real K3 needs.
        teacher_forced: Whether to add the output-level oracle, which
            re-prefills every committed token through the plain pipeline:
            under greedy acceptance each must be the argmax at its
            position. Costs one generate per token. Under a synthetic rate
            only the step bonuses -- the tokens the verify chose itself,
            read off the run's real step boundaries -- are graded, since
            the accepted drafts are the drafter's choices and miss by
            construction.
        plain_pipeline: A prebuilt plain pipeline to grade against, as
            :func:`_pipeline` builds for ``speculative=False`` with these
            prompts. The caller owns it and may share one build across
            several runs; ``None`` builds a fresh one per call.
        kv_cache_format: The shared pool's storage dtype override, or
            ``None`` for the pipeline's own default. Both runs (spec and
            plain) must sit on the same dtype or the comparison would
            price the dtype difference, not the speculative path -- the
            latent drafter's production arm is ``"float8_e4m3fn"`` and
            the selectable one ``"bfloat16"``, and which one a run was
            on was previously unstamped.
    """
    register_all_models()
    prompts = read_prompts(prompts_path)
    if teacher_forced and not all(isinstance(p, list) for p in prompts):
        raise ValueError(
            "--teacher-forced needs token-id prompts: text prompts carry no"
            " ids to re-prefill, so the oracle would grade nothing"
        )

    spec_pipeline = _pipeline(
        target,
        drafter,
        devices,
        max_length,
        num_speculative_tokens,
        synthetic_acceptance,
        device_graph_capture,
        prefer_module_v3,
        trust_remote_code,
        prompts,
        speculative=True,
        kv_cache_format=kv_cache_format,
    )
    if device_graph_capture:
        assert isinstance(spec_pipeline, OverlapTextGenerationPipeline)
        spec_pipeline.warmup_graph_capture()
    spec_runs: list[list[list[int]]] = []
    spec_chunks: list[list[int]] | None = None
    counts = _DraftCounts()
    for _ in range(repeat):
        spec_tokens, _, _, chunks = _generate(
            spec_pipeline,
            prompts,
            max_new_tokens,
            logits=False,
            counts=counts,
        )
        spec_runs.append(spec_tokens)
        spec_chunks = chunks
    del spec_pipeline
    gc.collect()

    owns_plain = plain_pipeline is None
    plain = (
        _pipeline(
            target,
            None,
            devices,
            max_length,
            num_speculative_tokens,
            synthetic_acceptance,
            device_graph_capture,
            prefer_module_v3,
            trust_remote_code,
            prompts,
            speculative=False,
            kv_cache_format=kv_cache_format,
        )
        if plain_pipeline is None
        else plain_pipeline
    )
    plain_tokens, plain_logits, prompt_lengths, _ = _generate(
        plain, prompts, max_new_tokens, logits=True
    )
    assert plain_logits is not None
    misses: tuple[TeacherForcedMiss, ...] = ()
    plain_misses: tuple[TeacherForcedMiss, ...] = ()
    graded_positions = 0
    fold_sensitive_graded = 0
    if teacher_forced:
        assert spec_chunks is not None
        target_chosen, fold_sensitive_graded = _step_bonus_positions(
            spec_chunks
        )
        if synthetic_acceptance is None:
            misses = _teacher_forced(plain, prompts, spec_runs[0])
            plain_misses = _teacher_forced(plain, prompts, plain_tokens)
            graded_positions = sum(
                len(spec_runs[0][index])
                for index, prompt in enumerate(prompts)
                if isinstance(prompt, list)
            )
        else:
            # A synthetic stream mixes drafter-chosen tokens with the
            # target's own; only the target-chosen bonuses are graded.
            misses = _teacher_forced(
                plain,
                prompts,
                spec_runs[0],
                target_chosen=target_chosen,
            )
            graded_positions = sum(
                len(positions) for positions in target_chosen
            )
    if owns_plain:
        del plain
        gc.collect()

    deterministic = all(run_ == spec_runs[0] for run_ in spec_runs[1:])
    reports = [
        prompt_report(
            index,
            prompt,
            prompt_lengths[index],
            spec,
            plain,
            logits_stack,
        )
        for index, (prompt, spec, plain, logits_stack) in enumerate(
            zip(prompts, spec_runs[0], plain_tokens, plain_logits, strict=True)
        )
    ]
    return ParityReport(
        repeats=repeat,
        deterministic=deterministic,
        drafts_proposed=counts.proposed,
        drafts_accepted=counts.accepted,
        teacher_forced_misses=misses,
        plain_teacher_forced_misses=plain_misses,
        graded_positions=graded_positions,
        fold_sensitive_graded=fold_sensitive_graded,
        prompts=tuple(reports),
    )
