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

from __future__ import annotations

import logging
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import Enum

from max.interfaces import (
    Pipeline,
    RequestID,
    TextGenerationInputs,
    TextGenerationOutput,
)
from max.kv_cache import InsufficientBlocksError, PagedKVCacheManager
from max.pipelines.core.context import TextContext
from max.pipelines.lib import LoRAManager
from max.profiler import traced
from max.serve.telemetry.metrics import METRICS

from ..config import TokenGenerationSchedulerConfig
from ..lora_scheduler_utils import (
    can_allocate_lora_request,
    is_active_lora,
    is_lora,
)
from .token_budget import (
    ActiveTokenBudget,
    BudgetStatus,
    RequestType,
    TokenBudget,
    TokenBudgetCollection,
    TotalContextTokenBudget,
)

logger = logging.getLogger("max.serve")


@dataclass
class ReplicaRequests:
    """This class tracks the requests assigned to each replica.

    This class is an implementation detail of TextBatchConstructor and should not be
    used outside of this file.
    """

    ce_reqs: OrderedDict[RequestID, TextContext] = field(
        default_factory=OrderedDict
    )
    tg_reqs: OrderedDict[RequestID, TextContext] = field(
        default_factory=OrderedDict
    )

    # LoRA-related bookkeeping for this replica.
    active_loras: set[str] = field(default_factory=set)
    deferred_lora_requests: dict[RequestID, TextContext] = field(
        default_factory=dict
    )

    def add_active_lora(
        self, context: TextContext, lora_manager: LoRAManager | None
    ) -> None:
        """Mark a LoRA as active for this replica and refresh its LRU in the manager."""
        if lora_manager is None:
            return

        if is_lora(context, lora_manager):
            lora_manager.activate_adapter(context.model_name)
            self.active_loras.add(context.model_name)

    def update_deferred_lora_requests(self) -> None:
        """Return deferred LoRA requests back to the CE queue in FIFO order."""
        for req_id, ctx in self.deferred_lora_requests.items():
            self.ce_reqs[req_id] = ctx
            self.ce_reqs.move_to_end(req_id, last=False)
        self.deferred_lora_requests.clear()

    def can_allocate_lora_request(
        self, context: TextContext, lora_manager: LoRAManager | None
    ) -> bool:
        """Check LoRA budget and defer the request if it cannot be safely activated."""
        if lora_manager is None:
            return True

        if not can_allocate_lora_request(
            context, self.active_loras, lora_manager
        ):
            self.deferred_lora_requests[context.request_id] = context
            return False

        return True


@dataclass
class ReplicaBatch:
    """This class represents a batch of requests for a single replica.

    This class mainly serves as a container for batch and num_steps. This is nicer
    than passing around Tuple[dict[RequestID, TextContext], int] everywhere.

    This class is an implementation detail of TextBatchConstructor and should not be
    used outside of this file.
    """

    batch: dict[RequestID, TextContext]
    num_steps: int
    token_budget: TokenBudgetCollection

    def __len__(self) -> int:
        return len(self.batch)

    def is_empty(self) -> bool:
        return len(self.batch) == 0


class PreemptionReason(str, Enum):
    KV_CACHE_MEMORY = "kv_cache_memory"
    MAX_NUM_LORAS = "max_num_loras"

    @property
    def error_message(self) -> str:
        match self:
            case PreemptionReason.MAX_NUM_LORAS:
                return "Preempted a request due to max-num-loras limit exceeded. This can affect the end-to-end performance. Consider increasing max-num-loras."

            case PreemptionReason.KV_CACHE_MEMORY:
                return "Preempted a request due to lack of KV pages. This can affect the end-to-end performance. Consider increasing device-memory-utilization via `--device-memory-utilization` to provide more KV cache memory."


class BatchSchedulingStrategy(str, Enum):
    """Strategy for prioritizing CE (prefill) vs TG (decode) batch construction.

    This enum controls how replicas prioritize between context encoding (CE/prefill)
    and token generation (TG/decode) requests when constructing batches. The strategy
    can either enforce a global priority across all replicas or allow each replica to
    independently determine priority based on its local queue state.

    The default behavior (None/unset) corresponds to PER_REPLICA mode, where each
    replica independently decides priority based on its queue state and the
    enable_in_flight_batching configuration setting.
    """

    PREFILL_FIRST = "prefill_first"
    """Always prioritize CE (context encoding/prefill) requests.

    When this strategy is set, all replicas will prioritize building CE batches
    before processing any TG requests. This maximizes prompt throughput and
    minimizes time-to-first-token at the cost of potentially higher inter-token
    latency for ongoing generations.
    """

    DECODE_FIRST = "decode_first"
    """Always prioritize TG (token generation/decode) requests.

    When this strategy is set, all replicas will prioritize building TG batches
    and only process CE requests when no TG work is available. This minimizes
    inter-token latency for active generations at the cost of potentially higher
    time-to-first-token for new requests.
    """

    BALANCED = "balanced"
    """Adaptively prioritize based on relative queue sizes across replicas.

    When this strategy is set, the scheduler considers the global state across
    all replicas to determine priority. If the majority of pending work is CE,
    prioritize CE; if the majority is TG, prioritize TG. This provides a middle
    ground between PREFILL_FIRST and DECODE_FIRST strategies.
    """

    PER_REPLICA = "per_replica"
    """Each replica independently manages its own batching priority (default).

    This is the default behavior when no strategy is explicitly set. Each replica
    determines its own priority based on its local queue state and the
    enable_in_flight_batching configuration:

    - If enable_in_flight_batching=False: prioritize CE when CE queue is non-empty
    - If enable_in_flight_batching=True: prioritize TG when TG queue is non-empty

    This mode provides maximum flexibility and allows replicas to adapt to their
    individual workload characteristics, which is particularly useful in
    load-balanced deployments where request patterns may vary across replicas.
    """


class TextBatchConstructor:
    """Construct per-replica text batches from CE (prefill) and TG queues.

    This class encapsulates the high-level policy for forming execution
    batches for text pipelines. It operates on two logical phases of each
    request:

    - **Context encoding (CE / prefill)**: processing prompt tokens.
    - **Token generation (TG / decode)**: generating continuation tokens for
      already-prefilled requests.

    The batching policy is expressed entirely in terms of scheduler
    configuration (:class:`TokenGenerationSchedulerConfig`), model context
    metadata (e.g., lengths and LoRA adapter names), and KV cache / resource
    limits. The sections below describe the *intended behaviour* rather than
    the exact implementation.

    .. note::
       When chunked prefill is enabled and a request is split across multiple
       CE iterations, no output is emitted for that request until its full
       prefill completes. The request remains in the CE queue across
       iterations until all prompt tokens have been processed.

    **Replica assignment**

    The constructor supports data-parallel execution across
    ``data_parallel_degree`` replicas:

    - Each replica maintains its own CE and TG queues and forms batches
      independently using the same policy.
    - New requests are assigned to replicas using load-based assignment,
      which selects the replica with the fewest active requests. This
      provides better load balancing than round-robin, particularly when
      request sizes vary significantly or when requests complete at
      different rates.
    - The replica with the minimum request count is selected based on
      information from the paged KV cache manager. This accounts for
      both active processing and queued requests.
    - All replicas share the same logical KV memory budget through the
      paged KV cache manager.

    **Non-LoRA batch construction**

    The following describes batch construction when no LoRA manager is
    attached.

    **1. CE vs TG prioritization**

    For each replica the constructor maintains separate CE and TG queues.
    Priority depends on the ``enable_in_flight_batching`` setting:

    - When **False** (default):

      - If there are any CE requests waiting, the constructor prioritises
        building a CE batch until the CE queue is drained or budgets / limits
        are reached.
      - TG batches are formed only when there is no CE work pending on that
        replica. This favours fast admission of new prompts at the cost of
        slightly higher latency for ongoing generations.

    - When **True**:

      - If there are TG requests, the constructor prioritises TG, building a
        TG batch first and then optionally filling any remaining capacity in
        that iteration with CE work.
      - CE-only batches are formed when there is no TG work on the replica.
      - This favours throughput and inter-token latency for active
        generations at the cost of time-to-first-token for new requests, while
        still opportunistically progressing prefill when there is headroom.

    **2. Token budgets**

    Two logical token budgets can limit batch size. A context is admitted to
    the batch only if *all* active budgets agree that it fits. Budgets may
    report that:

    - Additional contexts can be admitted.
    - The current context should be admitted but further admissions in this
      batch should stop.
    - The context cannot fit and should be deferred to a later batch.

    *Active-token budget*

    - Capacity: ``target_tokens_per_batch_ce``.
    - Interprets the current active window of each context and enforces a soft
      limit on how many prompt tokens are processed in one CE batch.
    - When ``enable_chunked_prefill`` is **True**, large prompts may be
      *chunked* so that only a prefix of the prompt is processed in the
      current CE batch. The remainder is scheduled for follow-up CE
      iterations. This allows the constructor to:

      - Fill CE batches to approximately ``target_tokens_per_batch_ce``
        tokens.
      - Admit very large prompts without blocking the batch behind a single
        oversized request.

    - When chunked prefill is **disabled**, prompts are either admitted in
      full or deferred to a later batch once the active-token budget has been
      reached.

    *Total-context budget (optional)*

    - Enabled when ``max_batch_total_tokens`` is not ``None``.
    - Tracks the total resident context across the batch, accounting for
      current context length and planned forward steps, and ensures the sum
      does not exceed ``max_batch_total_tokens``.
    - This budget is only applied when a CE request is present, or to be
      added to the batch.

    **3. Max batch size limits**

    Independently of token budgets, explicit batch-size caps ensure that no
    batch grows without bound:

    - ``max_batch_size`` limits the number of requests in a batch per replica.

    Once a batch reaches its respective maximum size, no further requests are
    considered for that iteration, even if token budgets or cache capacity
    would allow more.

    **4. Decode step limits**

    The number of decode steps per TG iteration is bounded by
    ``max_forward_steps_tg``. When ``max_seq_len`` is provided, the final
    number of decode steps for an iteration is the *minimum* of
    ``max_forward_steps_tg`` and the number of steps each request can run
    before reaching ``max_seq_len``.

    **5. KV cache watermark and memory pressure**

    When a paged KV cache is used, KV memory becomes an additional limiting
    factor:

    - During CE, the constructor estimates how many KV blocks would be used if
      a candidate request were admitted.
    - If admitting the request would push usage above ``kvcache_ce_watermark``
      *and* there is active TG work on the replica, the request is deferred to
      protect room for ongoing generations. This prevents CE from starving
      existing TG batches of KV memory.
    - During TG, if allocating KV blocks for a candidate request fails due to
      insufficient capacity, the constructor may preempt other TG candidates
      to free KV space, ensuring that at least a subset of requests can
      continue generating. When preemption is required, the constructor evicts
      the most recently enqueued TG request first, minimising wasted work. A
      warning is logged when this occurs.

    Overall, KV cache limits, token budgets, and max batch sizes jointly
    determine the final CE and TG batch composition for each iteration.

    **LoRA batch construction**

    When a :class:`LoRAManager` is attached, the constructor applies
    additional constraints to respect the maximum number of concurrently
    loaded LoRA adapters and to avoid evicting adapters needed for active
    generations.

    - LoRA and data parallelism are mutually exclusive: if a LoRA manager is
      present, ``data_parallel_degree`` must be 1 so that all LoRA state is
      local to a single replica.
    - Each replica tracks a set of *protected* active LoRAs that are currently
      participating in the TG batch and must not be evicted.

    **1. CE batching with LoRA**

    For CE requests that target LoRA-adapted models:

    - A LoRA request is admitted into the CE batch only if activating its
      adapter would not exceed the LoRA manager's capacity, taking into
      account:

      - The protected LoRAs already in use by TG.
      - Any additional LoRAs the constructor intends to activate in this CE
        iteration.

    - If admitting a new LoRA request would require evicting a protected LoRA,
      that request is temporarily deferred and re-queued for a later CE batch,
      rather than forcing an eviction that would disrupt ongoing TG work.
    - When a LoRA request is finally admitted to CE, its adapter is activated
      and tracked as active for the replica, so that subsequent TG iterations
      can safely use it.

    This ensures that CE never "steals" LoRA capacity away from requests that
    are currently generating tokens.

    **2. TG batching with LoRA**

    For TG requests that depend on LoRA adapters:

    - The constructor requires that the corresponding LoRA adapter is
      currently active. If it is not, the request is *preempted* from the TG
      batch, reset, and returned to CE so that its LoRA weights can be safely
      reloaded in a future batch. A warning is logged when this occurs.
    - TG batches therefore only ever contain LoRA requests for which the
      correct adapters are already resident, guaranteeing that generations are
      computed with the intended LoRA parameters.

    """

    def __init__(
        self,
        scheduler_config: TokenGenerationSchedulerConfig,
        pipeline: Pipeline[
            TextGenerationInputs[TextContext], TextGenerationOutput
        ],
        kv_cache: PagedKVCacheManager,
        batch_scheduling_strategy: BatchSchedulingStrategy = BatchSchedulingStrategy.PER_REPLICA,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.pipeline = pipeline
        self.kv_cache = kv_cache
        self.batch_scheduling_strategy = batch_scheduling_strategy

        self._lora_manager: LoRAManager | None = LoRAManager.get_lora_manager(
            pipeline
        )

        self.num_replicas = self.scheduler_config.data_parallel_degree
        if self._lora_manager and self.num_replicas > 1:
            raise ValueError("LoRA does not support data parallelism.")

        self.replicas: list[ReplicaRequests] = [
            ReplicaRequests() for _ in range(self.num_replicas)
        ]
        self._request_id_to_replica_idx: dict[RequestID, int] = {}
        self._request_id_to_lora_name: dict[RequestID, str | None] = {}

        self.total_preemption_count: int = 0
        self.last_preemption_logging_time: float = time.monotonic()

    def _create_new_token_budget(self) -> TokenBudgetCollection:
        token_budgets: list[TokenBudget] = [
            ActiveTokenBudget(
                capacity=self.scheduler_config.target_tokens_per_batch_ce,
                allow_chunking=self.scheduler_config.enable_chunked_prefill,
                applicable_types=RequestType.all(),
            )
        ]

        if self.scheduler_config.max_batch_total_tokens is not None:
            token_budgets.append(
                TotalContextTokenBudget(
                    capacity=self.scheduler_config.max_batch_total_tokens,
                    allow_chunking=self.scheduler_config.enable_chunked_prefill,
                    applicable_types=[
                        RequestType.CE,
                        RequestType.MIXED,
                    ],
                    cost_alignment=self.kv_cache.page_size,
                )
            )

        return TokenBudgetCollection(
            token_budgets=token_budgets,
        )

    def get_next_replica_idx(
        self, external_requests_per_replica: list[int] | None = None
    ) -> int:
        """Returns the next replica index to assign the request to.

        Uses load-based assignment by selecting the replica with the fewest
        active requests. This provides better load balancing than round-robin,
        especially when request sizes vary or when requests complete at
        different rates.

        Args:
            external_requests_per_replica: The number of requests per replica
                that are not managed by the batch constructor.

        Returns:
            The replica index that should receive the next request.
        """
        if external_requests_per_replica is None:
            external_requests_per_replica = [0] * self.num_replicas

        replica_idx = min(
            range(self.num_replicas),
            key=lambda idx: (
                len(self.replicas[idx].ce_reqs)
                + len(self.replicas[idx].tg_reqs)
                + external_requests_per_replica[idx]
            ),
        )
        return replica_idx

    def enqueue_new_request(
        self, ctx: TextContext, replica_idx: int | None = None
    ) -> None:
        """Add a new CE request to a replica.

        Args:
            ctx: The request to enqueue.
            replica_idx: The replica index to assign the request to.
                If None, the next replica index will be automatically chosen.
        """

        # Pick the replica to enqueue the request to.
        if replica_idx is None:
            replica_idx = self.get_next_replica_idx()
        replica = self.replicas[replica_idx]
        self._request_id_to_replica_idx[ctx.request_id] = replica_idx
        self._request_id_to_lora_name[ctx.request_id] = (
            ctx.model_name
            if self._lora_manager and is_lora(ctx, self._lora_manager)
            else None
        )

        # Add the request to the appropriate dict based on whether it needs CE.
        if ctx.tokens.generated_length == 0:
            replica.ce_reqs[ctx.request_id] = ctx
        else:
            replica.tg_reqs[ctx.request_id] = ctx

    def advance_requests(
        self, inputs: TextGenerationInputs[TextContext]
    ) -> None:
        """Advances request state based on executed CE batches.

        This method updates per-replica queues by moving executed context encoding (CE)
        requests into the text generation (TG) queues. If the last request in a batch
        is chunked and still requires additional CE work, it is moved back to the CE
        queue for that replica.

        Args:
            inputs: the inputs for the batch.
        """
        for per_replica_batch, replica in zip(
            inputs.batches, self.replicas, strict=True
        ):
            # It is possible that the batch is empty for a replica.
            if len(per_replica_batch) == 0:
                continue

            # Move the requests from CE to TG
            for context in per_replica_batch:
                replica.tg_reqs[context.request_id] = context

            # Move Chunked requests back to the CE request queue
            last_request = per_replica_batch[-1]
            if last_request.tokens.generated_length == 0:
                del replica.tg_reqs[last_request.request_id]
                replica.ce_reqs[last_request.request_id] = last_request
                replica.ce_reqs.move_to_end(last_request.request_id, last=False)

    def contains(self, request_id: RequestID) -> bool:
        """Checks if a request is in the batch constructor for any replica."""
        return request_id in self._request_id_to_replica_idx

    def release_request(self, request_id: RequestID) -> None:
        """
        Releases a request from the batch constructor for all replicas.

        This method searches for the given request_id in both context encoding (CE)
        and text generation (TG) request queues for each replica. If found, it removes
        the request entry and calls self.pipeline.release(request_id) to free resources.

        Args:
            request_id: The RequestID of the request to be released.
        """
        if not self.contains(request_id):
            raise ValueError(f"Request {request_id} not found in any replica.")

        # Retrieve the replica index for the request
        replica_idx = self._request_id_to_replica_idx[request_id]
        replica = self.replicas[replica_idx]

        if request_id in replica.ce_reqs:
            del replica.ce_reqs[request_id]
        elif request_id in replica.tg_reqs:
            del replica.tg_reqs[request_id]
        elif request_id in replica.deferred_lora_requests:
            del replica.deferred_lora_requests[request_id]
        # Request may already be in an active batch and therefore not appear in
        # any pending queue; continue cleanup in that case.

        # Clean up LoRA state if no other request uses this adapter.
        # Note: We only check the current replica because LoRA currently requires
        # data_parallel_degree == 1. If DP > 1 LoRA becomes supported, this check
        # would need to search across all replicas.
        lora_name = self._request_id_to_lora_name.pop(request_id, None)
        if lora_name is not None:
            # Check _request_id_to_lora_name rather than the queues because
            # requests may be in the active batch (not in any queue) but still
            # using this LoRA adapter.
            lora_still_needed = (
                lora_name in self._request_id_to_lora_name.values()
            )
            if not lora_still_needed:
                replica.active_loras.discard(lora_name)

        # Release from paged cache (scheduler manages primary KV cache lifecycle)
        if self.kv_cache is not None:
            self.kv_cache.release(request_id, replica_idx=replica_idx)

        # Pipeline release handles special cases (spec decoding draft model KV cache)
        # For regular pipelines, release() is a no-op
        self.pipeline.release(request_id)

        # _request_id_to_replica_idx is the source of truth for whether a request
        # is managed by the scheduler (checked by contains()).
        # Remove from here, marking the request as fully released.
        del self._request_id_to_replica_idx[request_id]

    def clear_tg_reqs(self) -> None:
        """Clears all TG requests from all replicas."""
        for replica in self.replicas:
            for request_id in replica.tg_reqs:
                del self._request_id_to_replica_idx[request_id]

            replica.tg_reqs.clear()

    @property
    def all_ce_reqs(self) -> dict[RequestID, TextContext]:
        """Returns a dictionary of all CE requests from all replicas."""
        return {
            req_id: ctx
            for replica in self.replicas
            for req_id, ctx in replica.ce_reqs.items()
        }

    @property
    def all_tg_reqs(self) -> dict[RequestID, TextContext]:
        """Returns a dictionary of all TG requests from all replicas."""
        return {
            req_id: ctx
            for replica in self.replicas
            for req_id, ctx in replica.tg_reqs.items()
        }

    @traced
    def _return_to_request_queue(
        self, context: TextContext, replica_idx: int
    ) -> None:
        """Resets a request and returns it to the request queue"""

        # Release from paged cache if it was claimed (scheduler manages primary KV cache lifecycle)
        if self.kv_cache is not None:
            for replica_idx in range(self.num_replicas):
                if self.kv_cache.contains(context.request_id, replica_idx):
                    self.kv_cache.release(context.request_id, replica_idx)
                    break

        # Pipeline release handles special cases (spec decoding draft model KV cache)
        # For regular pipelines, release() is a no-op
        self.pipeline.release(context.request_id)

        context.reset()

        # Move to CE Queue
        replica_requests = self.replicas[replica_idx]
        if context.request_id in replica_requests.tg_reqs:
            del replica_requests.tg_reqs[context.request_id]

        replica_requests.ce_reqs[context.request_id] = context
        replica_requests.ce_reqs.move_to_end(context.request_id, last=False)

    @traced
    def _preempt_request(
        self, context: TextContext, replica_idx: int, reason: PreemptionReason
    ) -> None:
        """Preempts the most recently received request from active batch"""

        # Return to the Request Queue
        self._return_to_request_queue(context, replica_idx)

        # Log Preemption
        current_time = time.monotonic()
        self.total_preemption_count += 1
        METRICS.preemption()
        if current_time - self.last_preemption_logging_time > 1:
            self.last_preemption_logging_time = current_time
            logger.info(
                reason.error_message
                + f" Total Preemption Count: {self.total_preemption_count}"
            )

    def _identify_priority(self, replica_idx: int) -> RequestType:
        # If there are no CE requests, prioritize TG
        if len(self.replicas[replica_idx].ce_reqs) == 0:
            return RequestType.TG

        # If there are no TG requests, prioritize Ce
        if len(self.replicas[replica_idx].tg_reqs) == 0:
            return RequestType.CE

        # If we've enabled in flight batching, prioritize TG
        if self.scheduler_config.enable_in_flight_batching:
            return RequestType.TG

        # Otherwise, prioritize CE
        return RequestType.CE

    def _add_ce_requests(self, batch: ReplicaBatch, replica_idx: int) -> None:
        replica_requests = self.replicas[replica_idx]
        max_batch_size = self.scheduler_config.max_batch_size

        # If there is anything in the batch, we can assume its TG, and the requests
        # are also counted in the tg_reqs.
        starting_tg_reqs_count = len(batch)
        max_batch_size = self.scheduler_config.max_batch_size
        while (
            len(batch) < max_batch_size
            # At a high level, active ce requests + tg_requests should not exceed the total max batch size.
            and len(batch)
            + len(replica_requests.tg_reqs)
            - starting_tg_reqs_count
            < max_batch_size
            and len(replica_requests.ce_reqs) > 0
        ):
            # Pop new request off the queue.
            req_id, ctx = replica_requests.ce_reqs.popitem(last=False)

            # Check LoRA budget before resource allocation
            if not replica_requests.can_allocate_lora_request(
                ctx, self._lora_manager
            ):
                continue

            # Exit early if the budget is already exhausted
            if batch.token_budget.remaining <= 0:
                self._return_to_request_queue(ctx, replica_idx)
                return

            # Check if the request fits in memory
            if self.kv_cache is not None:
                # Claim the request if needed.
                if not self.kv_cache.contains(req_id, replica_idx=replica_idx):
                    self.kv_cache.claim(req_id, replica_idx=replica_idx)

                # Check that the CE request does not go above the watermark
                pct_blocks_used_after_ce_request = max(
                    0.0,
                    min(
                        self.kv_cache.get_pct_used_blocks_after_allocation(
                            ctx, replica_idx=replica_idx
                        ),
                        1.0,
                    ),
                )
                if (
                    pct_blocks_used_after_ce_request
                    > self.scheduler_config.kvcache_ce_watermark
                ) and (len(batch) != 0 or len(replica_requests.tg_reqs) != 0):
                    self._return_to_request_queue(ctx, replica_idx)
                    break

                # Try to allocate kv cache blocks
                try:
                    self.kv_cache.alloc(
                        ctx, replica_idx=replica_idx, num_steps=1
                    )
                except InsufficientBlocksError:
                    if len(replica_requests.tg_reqs) == 0 and len(batch) == 0:
                        raise

                    # Return the Object to the request queue
                    self._return_to_request_queue(ctx, replica_idx)
                    break

            # Check if it fits within the token budget
            match batch.token_budget.status_after_context(
                ctx, num_steps=1, request_type=RequestType.CE
            ):
                case BudgetStatus.BUDGET_EXHAUSTED:
                    self._return_to_request_queue(ctx, replica_idx)
                    return
                case BudgetStatus.BUDGET_REACHED:
                    batch.batch[req_id] = ctx
                    batch.num_steps = 1
                    replica_requests.add_active_lora(ctx, self._lora_manager)
                    batch.token_budget.add_to_budget(
                        ctx, request_type=RequestType.CE
                    )
                    break
                case BudgetStatus.BUDGET_AVAILABLE:
                    batch.batch[req_id] = ctx
                    batch.num_steps = 1
                    batch.token_budget.add_to_budget(
                        ctx, request_type=RequestType.CE
                    )
                    replica_requests.add_active_lora(ctx, self._lora_manager)
                # case _:
                #     raise ValueError(f"Unexpected budget status: {status}")

        # Update deferred LoRA requests
        replica_requests.update_deferred_lora_requests()

    def _add_tg_requests(self, batch: ReplicaBatch, replica_idx: int) -> None:
        replica_requests = self.replicas[replica_idx]

        # Add based on the oldest request, respecting KV cache limits and token budgets.
        candidate_ids = deque(replica_requests.tg_reqs.keys())
        max_batch_size = self.scheduler_config.max_batch_size
        max_seq_len = self.scheduler_config.max_seq_len
        while len(batch) < max_batch_size and len(candidate_ids) > 0:
            # Pop the oldest request
            candidate_id = candidate_ids.popleft()
            candidate_context = replica_requests.tg_reqs[candidate_id]

            # Determine the number of steps to schedule based on the max_seq_len
            # of the pipeline model.
            if max_seq_len is not None:
                batch.num_steps = min(
                    batch.num_steps,
                    candidate_context.compute_num_available_steps(
                        max_seq_len,
                    ),
                )

            # Verify LoRA is active for TG requests
            # LoRA requests should have been activated during CE
            if is_lora(
                candidate_context, self._lora_manager
            ) and not is_active_lora(candidate_context, self._lora_manager):
                self._preempt_request(
                    candidate_context,
                    replica_idx,
                    reason=PreemptionReason.MAX_NUM_LORAS,
                )
                continue

            # Check if it fits within the token budget
            # This is quite cheap, compared to the paged cache allocations
            # So we can do it first.
            status = batch.token_budget.status_after_context(
                candidate_context,
                num_steps=batch.num_steps,
                request_type=RequestType.TG,
            )
            if status == BudgetStatus.BUDGET_EXHAUSTED:
                break

            # At this point, we can assume that the paged cache is active.
            while True:
                try:
                    self.kv_cache.alloc(
                        candidate_context,
                        replica_idx=replica_idx,
                        num_steps=batch.num_steps,
                    )
                    break
                except InsufficientBlocksError:
                    if len(candidate_ids) == 0:
                        if len(batch) == 0:
                            raise
                        else:
                            return

                    # Pop the oldest candidate id
                    oldest_id = candidate_ids.pop()
                    oldest_context = replica_requests.tg_reqs.pop(oldest_id)
                    self._preempt_request(
                        oldest_context,
                        replica_idx,
                        reason=PreemptionReason.KV_CACHE_MEMORY,
                    )

            match status:
                case BudgetStatus.BUDGET_REACHED:
                    batch.batch[candidate_context.request_id] = (
                        candidate_context
                    )
                    batch.token_budget.add_to_budget(
                        candidate_context, request_type=RequestType.TG
                    )
                    break
                case BudgetStatus.BUDGET_AVAILABLE:
                    batch.batch[candidate_context.request_id] = (
                        candidate_context
                    )
                    batch.token_budget.add_to_budget(
                        candidate_context, request_type=RequestType.TG
                    )
                case _:
                    raise ValueError(f"Unexpected budget status: {status}")

    def _shrink_num_steps(self, batch: ReplicaBatch) -> None:
        """Shrinks the number of steps for a batch to fit within the token budget."""
        # If we are already running for 1 step, we cannot shrink further.
        if batch.num_steps == 1:
            return

        # Otherwise, we need to find the maximum number of steps that can be run for any request in the batch
        max_num_steps = 1
        for ctx in batch.batch.values():
            if ctx.max_length is None:
                return

            candidate_num_steps = ctx.compute_num_available_steps(
                ctx.max_length
            )
            if candidate_num_steps > max_num_steps:
                max_num_steps = candidate_num_steps

        if max_num_steps < batch.num_steps:
            batch.num_steps = max_num_steps

    @traced
    def _construct_replica_batch(
        self, replica_idx: int, priority_override: RequestType | None = None
    ) -> ReplicaBatch:
        """Constructs a batch for a single replica.

        Args:
            replica_idx: The index of the replica to construct a batch for.
            priority_override: Optional RequestType to override the priority
                identified by _identify_priority. If None, priority is determined
                automatically based on queue state and scheduler configuration.
        """

        # Initialize batch
        batch = ReplicaBatch(
            batch={},
            num_steps=self.scheduler_config.max_forward_steps_tg,
            token_budget=self._create_new_token_budget(),
        )

        # Use override if provided, otherwise identify priority automatically
        priority = (
            priority_override
            if priority_override is not None
            else self._identify_priority(replica_idx)
        )

        match priority:
            case RequestType.CE:
                self._add_ce_requests(batch, replica_idx)

                if len(batch) == 0 and priority_override is None:
                    self._add_tg_requests(batch, replica_idx)

            case RequestType.TG:
                self._add_tg_requests(batch, replica_idx)

                if (
                    self.scheduler_config.enable_in_flight_batching
                    and len(batch) > 0
                    and priority_override is None
                ):
                    self._add_ce_requests(batch, replica_idx)

        # Shrink the number of steps
        self._shrink_num_steps(batch)

        return batch

    def construct_batch(self) -> TextGenerationInputs[TextContext]:
        """Constructs Pipeline Inputs which includes a batch for each replica."""

        priority_override = None
        replica_priorities: set[RequestType] | list[RequestType]
        match self.batch_scheduling_strategy:
            case BatchSchedulingStrategy.DECODE_FIRST:
                replica_priorities = set(
                    self._identify_priority(idx)
                    for idx in range(self.num_replicas)
                )
                if RequestType.TG in replica_priorities:
                    priority_override = RequestType.TG
            case BatchSchedulingStrategy.PREFILL_FIRST:
                replica_priorities = set(
                    self._identify_priority(idx)
                    for idx in range(self.num_replicas)
                )
                if RequestType.CE in replica_priorities:
                    priority_override = RequestType.CE
            case BatchSchedulingStrategy.BALANCED:
                replica_priorities = list(
                    self._identify_priority(idx)
                    for idx in range(self.num_replicas)
                )

                # Count occurrences of each priority type
                ce_count = replica_priorities.count(RequestType.CE)
                tg_count = replica_priorities.count(RequestType.TG)

                # Set priority to the majority case, defaulting to TG if tied
                if ce_count > tg_count:
                    priority_override = RequestType.CE
                else:
                    priority_override = RequestType.TG

        batches_per_replica = [
            self._construct_replica_batch(
                replica_idx, priority_override=priority_override
            )
            for replica_idx in range(self.num_replicas)
        ]

        return TextGenerationInputs[TextContext](
            batches=[
                list(batch.batch.values()) for batch in batches_per_replica
            ],
            # Take the min num_steps across all replicas that have a non-empty batch.
            # This ensures that when there is a single request and DP>1, we run with
            # the full num_steps and not num_steps=1.
            num_steps=min(
                (
                    batch.num_steps
                    for batch in batches_per_replica
                    if len(batch.batch) > 0
                ),
                default=0,
            ),
        )
