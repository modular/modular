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
"""Driving the DSpark drafter's two block entries over one cache, for tests.

The paged entry is only correct if it drafts what the dense one drafts, so both
tests of it walk the same scenarios: a batch of requests whose contexts grow by
a known number of rows per step, written into a paged leaf, then a block drafted
over the window that step's rows extended -- against the dense entry run on the
same rows, one request at a time (the dense entry holds one context pool shared
by the whole batch, which is one prompt with as many anchors as it has rows).

``test_dspark_paged.py`` runs this on a miniature drafter in float32 on the CPU;
``test_dspark_paged_parity.py`` runs it on the mini checkpoint on a GPU. What
differs between them is the geometry, the weights and the dtype, so everything
here takes those from the caller.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from max import functional as F
from max import tree
from max.driver import Buffer
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn import Module
from max.nn.common_layers.embedding import VocabParallelEmbedding
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.linear import ColumnParallelLinear
from max.nn.kv_cache import KVCacheInputsPerDevice, KVCacheParams
from max.pipelines.architectures.mistral_large4_modulev3.dspark import (
    DSparkDrafter,
    DSparkDrafterConfig,
    head_from_weight,
    window_start,
)
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, DeviceMesh
from test_common.simple_kv_cache import paged_kv_cache_inputs

TOLERANCE = 1e-4
"""The logit bar where the arithmetic is really float32 (the CPU gate)."""


@dataclass(frozen=True)
class Scenario:
    """A batch's worth of context, written over one or more steps."""

    name: str
    lengths: tuple[int, ...]
    """Context rows each request ends up holding."""
    steps: tuple[tuple[int, ...], ...]
    """Rows each request writes per step; they sum to ``lengths``."""


SHORT_SCENARIOS = [
    Scenario("anchor_zero", (1,), ((1,),)),
    Scenario("anchor_inside_window", (3,), ((3,),)),
    Scenario("two_requests", (3, 7), ((3, 7),)),
    Scenario("second_step", (6,), ((3,), (3,))),
]
"""Every window shape a short context reaches, written in one or two steps."""


def window_scenario(
    window: int, *, name: str = "anchor_past_window"
) -> Scenario:
    """One request whose context overruns the window, written in one step.

    The window is what makes this scenario worth its rows: at an anchor past
    ``window - 1`` the block reads only the last ``window`` slots, so the paged
    entry's gather and the dense entry's mask have to agree on which slots those
    are rather than on the whole context.
    """
    return Scenario(name, (window + 5,), ((window + 5,),))


@dataclass
class Parity:
    """One step's two engines, over one batch."""

    scenario: str
    step: int
    anchor_positions: np.ndarray
    paged_logits: np.ndarray
    paged_tokens: np.ndarray
    dense_logits: list[np.ndarray] = field(default_factory=list)
    dense_tokens: list[np.ndarray] = field(default_factory=list)
    request_errors: list[float] = field(default_factory=list)
    """Each request's max ``|dlogit|``, in request order."""

    @property
    def abs_error(self) -> float:
        """The batch's max ``|dlogit|``, over every request and block row."""
        return max(self.request_errors, default=0.0)


@dataclass
class Compiled:
    """Both entries of one drafter, sharing one weight set.

    ``place`` moves a numpy array onto the device the graphs run on, as the
    float32 CPU harness and the bf16 GPU one need different answers.
    """

    dense: Any
    paged: Any
    config: DSparkDrafterConfig
    kv_params: KVCacheParams
    embed_weight: Tensor
    head_weight: Tensor
    place: Callable[[np.ndarray], Tensor]


class PagedStep(Module[..., tuple[Tensor, Tensor]]):
    """One served step's phases, over the graph's own tensors.

    The fused block graph hands the drafter a ``PagedCacheValues`` the pairing
    built, so this is what turns the cache's flat graph inputs back into that
    record -- the same two lines the ML4 text model uses -- and then runs the
    two phases a step has: write the rows this iteration commits, run the block
    over the window they extended.

    The embedding and the head arrive as whole ``[vocab, hidden]`` weights, the
    one-device form of the target's borrowed tables; a graph whose vocabulary
    is sharded over the ranks enters at :class:`ShardedPagedStep` instead.

    Holding the drafter as a child renames its parameters under ``drafter.``,
    which is why a compile hands the weights over prefixed.
    """

    def __init__(
        self,
        drafter: DSparkDrafter,
        kv_params: KVCacheParams,
        mesh: DeviceMesh,
    ) -> None:
        super().__init__()
        self.drafter = drafter
        self.kv_params = kv_params
        self.mesh = mesh

    def forward(
        self,
        write_taps: Tensor,
        write_positions: Tensor,
        write_offsets: Tensor,
        anchor_ids: Tensor,
        anchor_positions: Tensor,
        embed_weight: Tensor,
        head_weight: Tensor,
        *kv: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Writes the step's rows, then drafts over the extended window."""
        values = _cache_values(self.kv_params, self.mesh, kv)
        self.drafter.write_context(
            write_taps, write_positions, write_offsets, values
        )
        return self.drafter.forward_block_paged(
            self.drafter.embed_block(anchor_ids, embed_weight),
            anchor_ids,
            anchor_positions,
            _block_position(values, write_offsets),
            head_from_weight(head_weight),
        )


class ShardedPagedStep(Module[..., tuple[Tensor, Tensor]]):
    """One served step's phases behind the target's vocabulary shards.

    The fused graph borrows the target's embedding and LM head, and the target
    shards both over its tensor-parallel ranks: a rank holds a slice of the
    vocabulary, so the drafter cannot look a token up in the weight and the
    head returns a rank's columns of the logits. This holds the target's own two
    modules at the mesh the block runs on, and enters the drafter where the
    served pairing does -- :meth:`~DSparkDrafter.block_from_anchors` for the
    rows and :meth:`~DSparkDrafter.base_logits`' all-gather for the logits.
    """

    def __init__(
        self,
        drafter: DSparkDrafter,
        kv_params: KVCacheParams,
        mesh: DeviceMesh,
    ) -> None:
        super().__init__()
        self.drafter = drafter
        self.kv_params = kv_params
        self.mesh = mesh
        config = drafter.config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, dim=config.hidden_size
        )
        self.lm_head = ColumnParallelLinear(
            in_dim=config.hidden_size, out_dim=config.vocab_size, bias=False
        )

    def forward(
        self,
        write_taps: Tensor,
        write_positions: Tensor,
        write_offsets: Tensor,
        anchor_ids: Tensor,
        anchor_positions: Tensor,
        *kv: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Writes the step's rows, then drafts over the extended window."""
        values = _cache_values(self.kv_params, self.mesh, kv)
        self.drafter.write_context(
            write_taps, write_positions, write_offsets, values
        )
        block = self.drafter.block_from_anchors(
            self.embed_tokens(anchor_ids.cast(DType.int32))
        )
        return self.drafter.forward_block_paged(
            block,
            anchor_ids,
            anchor_positions,
            _block_position(values, write_offsets),
            self.lm_head,
        )


def _cache_values(
    kv_params: KVCacheParams, mesh: DeviceMesh, kv: Sequence[Tensor]
) -> PagedCacheValues:
    """The step's cache, from the graph's flat cache inputs."""
    return PagedCacheValues.from_upstream(
        kv_params.unflatten_kv_inputs(iter(value._graph_value for value in kv)),
        DeviceMapping.replicated(mesh),
    )


def _block_position(
    values: PagedCacheValues, row_offsets: Tensor
) -> PagedCacheValues:
    """The leaf as the block driver stages it for the read: past this step's rows.

    A step has two positions. :meth:`~DSparkDrafter.write_context` places its
    rows at the *ctx* position -- the lengths before them -- while the block
    driver hands the read the *block* position, the same leaf with this step's
    rows counted (``cache_lengths[b] + rows written by b``). Serving stages both
    because the two phases take different graph inputs; this harness writes
    into one leaf, so it derives the read's position from the write's own
    offsets. That matters to the read's anchor guard, which bounds the anchor
    by the leaf it is handed.
    """
    counts = row_offsets[1:] - row_offsets[:-1]
    # The offsets' length is its own symbolic dim; the leaf's is the batch's.
    counts = F.rebind(counts, values.cache_lengths.shape)
    lengths = values.cache_lengths + counts.cast(values.cache_lengths.dtype)
    return dataclasses.replace(values, cache_lengths=lengths)


def dense_input_types(
    config: DSparkDrafterConfig, dtype: DType, device: DeviceRef
) -> list[TensorType]:
    """The dense step's graph inputs, in ``DSparkDrafter.forward``'s order."""
    borrowed = TensorType(
        dtype, [config.vocab_size, config.hidden_size], device=device
    )
    return [
        TensorType(dtype, ["rows", config.tap_width], device=device),
        TensorType(DType.int32, ["rows"], device=device),
        TensorType(DType.int32, ["batch"], device=device),
        TensorType(DType.int32, ["batch"], device=device),
        borrowed,
        borrowed,
    ]


def paged_input_types(
    config: DSparkDrafterConfig,
    kv_params: KVCacheParams,
    dtype: DType,
    device: DeviceRef,
) -> list[TensorType | BufferType]:
    """The paged step's graph inputs, in :meth:`PagedStep.forward`'s order.

    Past the dense entry's rows and the two borrowed matrices, the cache's own
    flat leaves follow: the same list the pairing's ``kv_params`` declares, so a
    caller cannot get their order wrong. They are buffer types as the leaf
    declares them (the store op writes in place), which is why this list is not
    ``TensorType``-only.
    """
    borrowed = TensorType(
        dtype, [config.vocab_size, config.hidden_size], device=device
    )
    return [
        TensorType(dtype, ["rows", config.tap_width], device=device),
        TensorType(DType.int32, ["rows"], device=device),
        TensorType(DType.uint32, ["offsets_len"], device=device),
        TensorType(DType.int32, ["batch"], device=device),
        TensorType(DType.int32, ["batch"], device=device),
        borrowed,
        borrowed,
        *kv_params.flattened_kv_inputs(),
    ]


def _leaves(
    cache: KVCacheInputsPerDevice[Buffer, Buffer], cache_lengths: Sequence[int]
) -> list[Tensor]:
    """The cache's flat graph arguments, at the given pre-step lengths.

    The block driver stages the leaf the same way: it replaces the lengths and
    reads everything else -- pages, lookup table, bounds -- off the cache the
    earlier steps wrote, which is what a step's "ctx position" means.

    Those lengths place the write. The read addresses its rows through
    ``anchor_positions`` but bounds the anchor by ``cache_lengths``, so a step
    that reads past what it is about to write stages the block position
    instead (:func:`_block_position`) -- what the block driver hands the block
    phase, and why the step classes below write through one position and read
    through the other.
    """
    lengths = Buffer.from_numpy(np.array(cache_lengths, dtype=np.uint32))
    lengths = lengths.to(cache.cache_lengths.device)
    staged = dataclasses.replace(cache, cache_lengths=lengths)
    return [
        leaf if isinstance(leaf, Tensor) else _to_tensor(leaf)
        for leaf in tree.leaves(staged)
    ]


def _to_tensor(value: Any) -> Tensor:
    return value if isinstance(value, Tensor) else Tensor(storage=value)


def parity(
    compiled: Compiled,
    scenario: Scenario,
    taps: Sequence[np.ndarray],
    anchor_ids: Sequence[int],
) -> list[Parity]:
    """Runs one scenario's every step through both entries.

    Args:
        compiled: The two compiled entries, sharing their weights.
        scenario: The contexts to write and how to split the writes.
        taps: One ``[length, tap_width]`` array per request, in scenario order.
        anchor_ids: One anchor token per request.

    Returns:
        One :class:`Parity` per step, holding each request's logits and tokens
        from both entries.
    """
    config = compiled.config
    cache = paged_kv_cache_inputs(
        compiled.kv_params,
        prompt_lens=list(scenario.lengths),
        cache_lengths=[0] * len(scenario.lengths),
    )
    written = [0] * len(scenario.lengths)
    results: list[Parity] = []
    for step, counts in enumerate(scenario.steps):
        before = list(written)
        rows = []
        positions = []
        for request, count in enumerate(counts):
            start = written[request]
            rows.append(taps[request][start : start + count])
            positions.append(np.arange(start, start + count, dtype=np.int32))
            written[request] += count
        anchors = [total - 1 for total in written]
        logits, tokens = compiled.paged(
            compiled.place(np.concatenate(rows, axis=0)),
            compiled.place(np.concatenate(positions)),
            compiled.place(
                np.concatenate([[0], np.cumsum(counts)]).astype(np.uint32)
            ),
            compiled.place(np.array(anchor_ids, dtype=np.int32)),
            compiled.place(np.array(anchors, dtype=np.int32)),
            compiled.embed_weight,
            compiled.head_weight,
            *_leaves(cache, before),
        )
        result = Parity(
            scenario=scenario.name,
            step=step,
            anchor_positions=np.array(anchors, dtype=np.int32),
            paged_logits=logits.cast(DType.float32).to_numpy(),
            paged_tokens=tokens.to_numpy(),
        )
        for request, anchor in enumerate(anchors):
            start = window_start(anchor, config.window)
            dense_logits, dense_tokens = compiled.dense(
                compiled.place(taps[request][start : anchor + 1]),
                compiled.place(np.arange(start, anchor + 1, dtype=np.int32)),
                compiled.place(np.array([anchor], dtype=np.int32)),
                compiled.place(np.array([anchor_ids[request]], dtype=np.int32)),
                compiled.embed_weight,
                compiled.head_weight,
            )
            reference = dense_logits.cast(DType.float32).to_numpy()
            result.dense_logits.append(reference)
            result.dense_tokens.append(dense_tokens.to_numpy())
            result.request_errors.append(
                float(
                    np.max(
                        np.abs(
                            result.paged_logits[request : request + 1].reshape(
                                -1
                            )
                            - reference.reshape(-1)
                        )
                    )
                )
            )
        results.append(result)
    return results


def check(results: Sequence[Parity], *, logits: float | None) -> list[str]:
    """Compares each request against the dense entry, reporting per step.

    Args:
        results: One :class:`Parity` per step, from :func:`parity`.
        logits: The bar on the logits, or ``None`` to report them instead of
            holding them to one. A bf16 engine reports them: at logit
            magnitudes of tens, one bf16 ulp is 0.06, so two orderings of the
            same sum differ by more than any useful bar on rounding alone, while
            a flipped token is what a reader sees.

    Returns:
        One message per failure, so a caller can assert them all at once.
    """
    failures: list[str] = []
    for step in results:
        for request, anchor in enumerate(step.anchor_positions):
            where = (
                f"{step.scenario} step {step.step} request {request}"
                f" (anchor {anchor})"
            )
            if not np.array_equal(
                step.paged_tokens[request : request + 1],
                step.dense_tokens[request],
            ):
                failures.append(
                    f"{where}: the paged block drafted"
                    f" {step.paged_tokens[request].tolist()} but the dense one"
                    f" {step.dense_tokens[request].reshape(-1).tolist()}"
                )
            # The error is per request: a batch-wide maximum reported under
            # every request's name points at the wrong one.
            if logits is not None and step.request_errors[request] > logits:
                failures.append(
                    f"{where}: max |dlogit|"
                    f" {step.request_errors[request]:.3e} > {logits:g}"
                )
        print(
            f"{step.scenario:20s} step {step.step}"
            f" anchors={step.anchor_positions.tolist()}"
            f" max|dlogit|={step.abs_error:.3e}",
            flush=True,
        )
    return failures
