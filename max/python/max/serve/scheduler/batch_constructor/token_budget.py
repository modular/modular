# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from abc import ABC, abstractmethod
from enum import Enum

from max.interfaces.pipeline_variants import TextGenerationContext


class RequestType(str, Enum):
    TG = "text_generation"
    CE = "context_encoding"
    MIXED = "mixed"
    UNKNOWN = "unknown"

    @classmethod
    def all(cls) -> list[RequestType]:
        return [RequestType.TG, RequestType.CE, RequestType.MIXED]


class BudgetStatus(str, Enum):
    """Enumeration describing the result of applying a token budget to a context.

    Attributes:
        BUDGET_AVAILABLE: The context fits within the budget and there is still
            remaining capacity for additional contexts.
        BUDGET_EXHAUSTED: The context cannot be added to the budget, even with
            chunking. This occurs when the budget is already full or when a hard
            or soft limit prevents accepting the context.
        BUDGET_REACHED: The context fits within the budget (possibly after
            chunking) and the budget is now at or near capacity.
    """

    BUDGET_AVAILABLE = "budget_available"
    BUDGET_EXHAUSTED = "budget_exhausted"
    BUDGET_REACHED = "budget_reached"


class TokenBudget(ABC):
    """Abstract base class for token budgets used during batch construction.

    A ``TokenBudget`` tracks how many tokens have been consumed from a particular
    dimension (for example, active prompt tokens or total context length) and
    exposes a common protocol for:

    * Checking whether a :class:`TextGenerationContext` can be admitted to the
      batch via :meth:`status_after_context`.
    * Recording the token cost once a context has been accepted via
      :meth:`add_to_budget`.

    Implementations are free to interpret the token dimension they track, but
    they must respect the following contract:

    * If ``allow_chunking`` is True, implementations **may** call
      ``context.chunk`` to reduce the effective token cost against
      the budget.
    * :meth:`status_after_context` takes an optional ``num_steps`` argument
      describing how many generation steps the scheduler intends to run for the
      context. Budgets that care about future growth (for example, total-context
      limits) should incorporate this into their effective cost; others may
      ignore it and remain per-step.
    * :meth:`add_to_budget` is only called after a non-``BUDGET_EXHAUSTED``
      status and is responsible for incrementing :attr:`used` by the same
      effective token cost that was evaluated in :meth:`status_after_context`.

    Attributes:
        capacity: Maximum number of tokens allowed for this budget.
        allow_chunking: Whether this budget may shrink the context via
            ``context.chunk`` in order to fit within the remaining
            capacity.
        used: Number of tokens currently consumed from this budget.
    """

    def __init__(
        self,
        capacity: int,
        allow_chunking: bool,
        applicable_types: list[RequestType],
    ) -> None:
        self.capacity = capacity
        self.allow_chunking = allow_chunking
        self.applicable_types = applicable_types

        self.used = 0
        self.active_request_type = RequestType.UNKNOWN

    @property
    def remaining(self) -> int:
        return self.capacity - self.used

    @abstractmethod
    def status_after_context(
        self,
        context: TextGenerationContext,
        request_type: RequestType,
        num_steps: int = 1,
    ) -> BudgetStatus:
        pass

    @abstractmethod
    def add_to_budget(
        self,
        context: TextGenerationContext,
        request_type: RequestType,
        num_steps: int = 1,
    ) -> None:
        pass


class TokenBudgetCollection:
    """Composite applying multiple :class:`TokenBudget` instances to a context.

    This helper allows the scheduler to treat several independent budgets
    (for example, active-token and total-context budgets) as a single logical
    budget. All budgets in the collection are evaluated for each context.
    """

    def __init__(self, token_budgets: list[TokenBudget]) -> None:
        """Create a collection of token budgets applied to the same context.

        The collection evaluates budgets in order and short-circuits on the
        first non-:data:`BudgetStatus.BUDGET_AVAILABLE` result. This allows the
        scheduler to enforce several independent limits (for example, active
        token and total-context budgets) with a single interface.

        Args:
            token_budgets: The list of budgets to apply to each context.
        """
        self.token_budgets = token_budgets

    def status_after_context(
        self,
        context: TextGenerationContext,
        request_type: RequestType,
        num_steps: int = 1,
    ) -> BudgetStatus:
        """Evaluate all budgets against a context and return the first violation.

        Budgets are evaluated in the order they were provided at construction
        time. The first budget that returns a status other than
        :data:`BudgetStatus.BUDGET_AVAILABLE` determines the overall result.
        If all budgets report :data:`BudgetStatus.BUDGET_AVAILABLE`, that
        status is returned.

        Args:
            context: The context being considered for inclusion in the batch.
            request_type: The type of request being evaluated.
            num_steps: Planned number of generation steps for this context. This
                is forwarded to each underlying :class:`TokenBudget`.

        Returns:
            The first non-available :class:`BudgetStatus` reported by any
            underlying budget, or :data:`BudgetStatus.BUDGET_AVAILABLE` if
            all budgets accept the context.
        """
        for token_budget in self.token_budgets:
            status = token_budget.status_after_context(
                context, request_type, num_steps
            )
            if status != BudgetStatus.BUDGET_AVAILABLE:
                return status
        return BudgetStatus.BUDGET_AVAILABLE

    def add_to_budget(
        self,
        context: TextGenerationContext,
        request_type: RequestType,
        num_steps: int = 1,
    ) -> None:
        """Apply the token cost to all underlying budgets for an accepted context.

        Args:
            context: The context that was just admitted into the batch.
            request_type: The type of request being added to the budget.
            num_steps: Planned number of generation steps that were considered
                when :meth:`status_after_context` was called.
        """
        for token_budget in self.token_budgets:
            token_budget.add_to_budget(context, request_type, num_steps)


class ActiveTokenBudget(TokenBudget):
    """Token budget that tracks the active window of each context.

    This budget is intended for limiting the number of tokens processed during
    a single context-encoding (CE) step. For each accepted context, the token
    cost is :attr:`TextGenerationContext.active_length`, and the budget
    may optionally shrink the active window via ``context.chunk`` when
    ``allow_chunking`` is enabled.

    The capacity and current usage are tracked via :attr:`capacity`,
    :attr:`used`, and :meth:`remaining`.
    """

    def status_after_context(
        self,
        context: TextGenerationContext,
        request_type: RequestType,
        num_steps: int = 1,
    ) -> BudgetStatus:
        """Evaluate whether the context's active tokens fit within the budget.

        This method examines ``context.active_length`` relative to the number of
        tokens remaining in the budget. It is intentionally **per-step**: the
        ``num_steps`` parameter is accepted for interface compatibility but is
        not used when computing the effective cost of a context. If the active window would
        exceed the remaining capacity and ``allow_chunking`` is enabled, it may
        call ``context.chunk(tokens_remaining)`` to shrink the active
        window so that it fits.

        **Important side effects**:

        * May mutate ``context`` by reducing its active window when chunking
          is enabled.
        * Does **not** update :attr:`used`. The caller must invoke
          :meth:`add_to_budget` after a non-``BUDGET_EXHAUSTED`` status in
          order to record the cost.

        Args:
            context: The :class:`TextGenerationContext` being considered.
            request_type: The type of request being evaluated.
            num_steps: Planned number of generation steps. Currently ignored for
                active-token budgets, which operate strictly on a per-step basis.

        Returns:
            A :class:`BudgetStatus` indicating if and how the context fits:

            * :data:`BudgetStatus.BUDGET_AVAILABLE` - context fits with room
              remaining.
            * :data:`BudgetStatus.BUDGET_REACHED` - context fits exactly or
              brings the budget to its limit.
            * :data:`BudgetStatus.BUDGET_EXHAUSTED` - context cannot be
              accommodated, even after any attempted chunking.

        Raises:
            ValueError: If chunking is enabled but ``context.chunk`` is
                unable to reduce the active window to within the remaining
                capacity.
        """
        if (
            self.active_request_type not in self.applicable_types
            and request_type not in self.applicable_types
        ):
            return BudgetStatus.BUDGET_AVAILABLE

        tokens_remaining = self.remaining

        # Already at or beyond capacity - no more contexts can be accepted.
        if tokens_remaining <= 0:
            return BudgetStatus.BUDGET_EXHAUSTED

        # Fits without any modification.
        if context.active_length <= tokens_remaining:
            if context.active_length == tokens_remaining:
                return BudgetStatus.BUDGET_REACHED
            return BudgetStatus.BUDGET_AVAILABLE

        # Would exceed the remaining capacity.
        if not self.allow_chunking:
            return BudgetStatus.BUDGET_REACHED

        # Try to shrink the active window so that it fits.
        try:
            context.chunk(tokens_remaining)
            return BudgetStatus.BUDGET_REACHED
        except ValueError:
            return BudgetStatus.BUDGET_EXHAUSTED

    def add_to_budget(
        self,
        context: TextGenerationContext,
        request_type: RequestType,
        num_steps: int = 1,
    ) -> None:
        """Record the token cost for an accepted context's active tokens.

        This should be called only after :meth:`status_after_context` has
        returned a non-:data:`BudgetStatus.BUDGET_EXHAUSTED` result for the
        same ``context``.

        Args:
            context: The context that was just admitted into the batch (possibly
                after being chunked).
            request_type: The type of request being added to the budget.
            num_steps: Planned number of generation steps. Currently ignored for
                active-token budgets, which consume only the per-step active window.
        """
        if (
            self.active_request_type != RequestType.MIXED
            and request_type != self.active_request_type
        ):
            self.active_request_type = request_type

        self.used += context.active_length


class TotalContextTokenBudget(TokenBudget):
    """Token budget that tracks the full context length for each request.

    Unlike :class:`ActiveTokenBudget`, which only costs the active window per
    step, this budget has an effective cost derived from
    :attr:`TextGenerationContext.current_length` and the planned number of
    generation steps. It is intended for enforcing limits such as
    ``max_batch_context_length`` that bound the total number of tokens resident
    in a batch across multiple steps.
    """

    def status_after_context(
        self,
        context: TextGenerationContext,
        request_type: RequestType,
        num_steps: int = 1,
    ) -> BudgetStatus:
        """Evaluate whether the context's total length fits within the budget.

        This method considers an effective cost based on
        :attr:`TextGenerationContext.current_length` and ``num_steps`` against
        the remaining capacity. Concretely, it assumes that over ``num_steps``
        generation steps the context will grow by ``num_steps - 1`` tokens,
        yielding an effective cost of

        ``context.current_length + (num_steps - 1)``.

        If the context would exceed the budget and ``allow_chunking`` is
        enabled, it may call
        ``context.chunk(tokens_remaining)`` to reduce the effective
        cost, though in practice chunking is typically more relevant for
        active-token budgets.

        Args:
            context: The :class:`TextGenerationContext` being considered.
            request_type: The type of request being evaluated.
            num_steps: Planned number of generation steps for this context.

        Returns:
            A :class:`BudgetStatus` indicating if and how the context fits:

            * :data:`BudgetStatus.BUDGET_AVAILABLE` - context fits with room
              remaining.
            * :data:`BudgetStatus.BUDGET_REACHED` - context exactly consumes
              the remaining capacity.
            * :data:`BudgetStatus.BUDGET_EXHAUSTED` - context cannot be
              accommodated within the remaining capacity.

        Raises:
            ValueError: If chunking is enabled but ``context.chunk`` does
                not succeed in reducing the effective cost to the remaining
                capacity.
        """
        if request_type not in self.applicable_types:
            return BudgetStatus.BUDGET_AVAILABLE

        tokens_remaining = self.remaining

        # Already at or beyond capacity - no more contexts can be accepted.
        if tokens_remaining <= 0:
            return BudgetStatus.BUDGET_EXHAUSTED

        total_length = context.current_length + (num_steps - 1)

        if total_length < tokens_remaining:
            return BudgetStatus.BUDGET_AVAILABLE
        elif total_length == tokens_remaining:
            return BudgetStatus.BUDGET_REACHED

        if not self.allow_chunking or context.active_length == 1:
            return BudgetStatus.BUDGET_EXHAUSTED

        # Practically, there is no runtime reason to fail chunking.
        # If it does, it is a bug in this token budget class, and should be fixed.
        context.chunk(tokens_remaining)
        return BudgetStatus.BUDGET_REACHED

    def add_to_budget(
        self,
        context: TextGenerationContext,
        request_type: RequestType,
        num_steps: int = 1,
    ) -> None:
        """Record the token cost for an accepted context's total length.

        This should be called only after :meth:`status_after_context` has
        returned a non-:data:`BudgetStatus.BUDGET_EXHAUSTED` result for the
        same ``context``.

        **Side effect**:
            Increments :attr:`used` by the same effective cost that was
            evaluated in :meth:`status_after_context`, namely
            ``context.current_length + (num_steps - 1)``.

        Args:
            context: The context that was just admitted into the batch.
            request_type: The type of request being added to the budget.
            num_steps: Planned number of generation steps for this context.
        """
        self.used += context.current_length + (num_steps - 1)
        print(f"Used: {self.used}")
