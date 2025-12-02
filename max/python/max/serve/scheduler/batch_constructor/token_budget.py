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

    A ``TokenBudget`` tracks how many tokens have been "spent" in a particular
    dimension (for example, active prompt tokens or total context length) and
    exposes a common protocol for:

    * Checking whether a :class:`TextGenerationContext` can be admitted to the
      batch via :meth:`status_after_context`.
    * Updating the internal accounting once a context has been accepted via
      :meth:`add_to_budget`.

    Implementations are free to interpret the token dimension they track, but
    they must respect the following contract:

    * If ``allow_chunking`` is True, implementations **may** call
      ``context.maybe_chunk`` to reduce the effective token count they charge
      to the budget.
    * ``add_to_budget`` is only called after a non-``BUDGET_EXHAUSTED`` status
      and is responsible for incrementing :attr:`used`.

    Attributes:
        capacity: Maximum number of tokens allowed for this budget.
        allow_chunking: Whether this budget may shrink the context via
            ``context.maybe_chunk`` in order to fit within the remaining
            capacity.
        used: Number of tokens currently charged against this budget.
    """

    def __init__(self, capacity: int, allow_chunking: bool) -> None:
        self.capacity = capacity
        self.allow_chunking = allow_chunking

        self.used = 0

    @property
    def remaining(self) -> int:
        return self.capacity - self.used

    def reset(self) -> None:
        self.used = 0

    @abstractmethod
    def status_after_context(
        self, context: TextGenerationContext
    ) -> BudgetStatus:
        pass

    @abstractmethod
    def add_to_budget(self, context: TextGenerationContext) -> None:
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

    def reset(self) -> None:
        """Reset all underlying budgets to their initial state."""
        for token_budget in self.token_budgets:
            token_budget.reset()

    def status_after_context(
        self, context: TextGenerationContext
    ) -> BudgetStatus:
        """Evaluate all budgets against a context and return the first violation.

        Budgets are evaluated in the order they were provided at construction
        time. The first budget that returns a status other than
        :data:`BudgetStatus.BUDGET_AVAILABLE` determines the overall result.
        If all budgets report :data:`BudgetStatus.BUDGET_AVAILABLE`, that
        status is returned.

        Args:
            context: The context being considered for inclusion in the batch.

        Returns:
            The first non-available :class:`BudgetStatus` reported by any
            underlying budget, or :data:`BudgetStatus.BUDGET_AVAILABLE` if
            all budgets accept the context.
        """
        for token_budget in self.token_budgets:
            status = token_budget.status_after_context(context)
            if status != BudgetStatus.BUDGET_AVAILABLE:
                return status
        return BudgetStatus.BUDGET_AVAILABLE

    def add_to_budget(self, context: TextGenerationContext) -> None:
        """Charge all underlying budgets for an accepted context."""
        for token_budget in self.token_budgets:
            token_budget.add_to_budget(context)


class ActiveTokenBudget(TokenBudget):
    """Token budget that accounts for the active window of each context.

    This budget is intended for limiting the number of tokens processed during
    a single context-encoding (CE) step. For each accepted context it charges
    :attr:`TextGenerationContext.active_length` tokens to the budget, and it
    may optionally shrink the active window via ``context.maybe_chunk`` when
    ``allow_chunking`` is enabled.

    The capacity and current usage are tracked via :attr:`capacity`,
    :attr:`used`, and :meth:`remaining`.
    """

    def __init__(self, capacity: int, allow_chunking: bool) -> None:
        super().__init__(capacity=capacity, allow_chunking=allow_chunking)

    def status_after_context(
        self, context: TextGenerationContext
    ) -> BudgetStatus:
        """Evaluate whether the context's active tokens fit within the budget.

        This method examines ``context.active_length`` relative to the number of
        tokens remaining in the budget. If the active window would exceed the
        remaining capacity and ``allow_chunking`` is enabled, it may call
        ``context.maybe_chunk(tokens_remaining)`` to shrink the active window
        so that it fits.

        **Important side effects**:

        * May mutate ``context`` by reducing its active window when chunking
          is enabled.
        * Does **not** update :attr:`used`. The caller must invoke
          :meth:`add_to_budget` after a non-``BUDGET_EXHAUSTED`` status in
          order to commit the charge.

        Args:
            context: The :class:`TextGenerationContext` being considered.

        Returns:
            A :class:`BudgetStatus` indicating if and how the context fits:

            * :data:`BudgetStatus.BUDGET_AVAILABLE` - context fits with room
              remaining.
            * :data:`BudgetStatus.BUDGET_REACHED` - context fits exactly or
              brings the budget to its limit.
            * :data:`BudgetStatus.BUDGET_EXHAUSTED` - context cannot be
              accommodated, even after any attempted chunking.

        Raises:
            ValueError: If chunking is enabled but ``context.maybe_chunk`` is
                unable to reduce the active window to within the remaining
                capacity.
        """
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
        new_length = context.maybe_chunk(tokens_remaining)
        if new_length > tokens_remaining:
            raise ValueError(
                "Chunked active length exceeds remaining budget: "
                f"{new_length} > {tokens_remaining}"
            )

        if new_length == tokens_remaining:
            return BudgetStatus.BUDGET_REACHED

        if new_length == 0:
            # Nothing left to charge, but budget is effectively at capacity.
            return BudgetStatus.BUDGET_REACHED

        return BudgetStatus.BUDGET_AVAILABLE

    def add_to_budget(self, context: TextGenerationContext) -> None:
        """Update the budget for an accepted context's active tokens.

        This should be called only after :meth:`status_after_context` has
        returned a non-:data:`BudgetStatus.BUDGET_EXHAUSTED` result for the
        same ``context``.

        Args:
            context: The context that was just admitted into the batch (possibly
                after being chunked).
        """
        self.used += context.active_length


class TotalContextTokenBudget(TokenBudget):
    """Token budget that tracks the full context length for each request.

    Unlike :class:`ActiveTokenBudget`, which charges only the active window per
    step, this budget charges :attr:`TextGenerationContext.current_length` for
    each accepted context. It is intended for enforcing limits such as
    ``max_batch_context_length`` that bound the total number of tokens resident
    in a batch.
    """

    def __init__(self, capacity: int, allow_chunking: bool) -> None:
        super().__init__(capacity=capacity, allow_chunking=allow_chunking)

    def status_after_context(
        self, context: TextGenerationContext
    ) -> BudgetStatus:
        """Evaluate whether the context's total length fits within the budget.

        This method considers :attr:`TextGenerationContext.current_length`
        against the remaining capacity. If the context would exceed the budget
        and ``allow_chunking`` is enabled, it may call
        ``context.maybe_chunk(tokens_remaining)`` to reduce the effective
        charge, though in practice chunking is typically more relevant for
        active-token budgets.

        Args:
            context: The :class:`TextGenerationContext` being considered.

        Returns:
            A :class:`BudgetStatus` indicating if and how the context fits:

            * :data:`BudgetStatus.BUDGET_AVAILABLE` - context fits with room
              remaining.
            * :data:`BudgetStatus.BUDGET_REACHED` - context exactly consumes
              the remaining capacity.
            * :data:`BudgetStatus.BUDGET_EXHAUSTED` - context cannot be
              accommodated within the remaining capacity.

        Raises:
            ValueError: If chunking is enabled but ``context.maybe_chunk`` does
                not succeed in reducing the effective charge to the remaining
                capacity.
        """
        tokens_remaining = self.remaining

        # Already at or beyond capacity - no more contexts can be accepted.
        if tokens_remaining <= 0:
            return BudgetStatus.BUDGET_EXHAUSTED

        total_length = context.current_length

        if total_length <= tokens_remaining:
            if total_length == tokens_remaining:
                return BudgetStatus.BUDGET_REACHED
            return BudgetStatus.BUDGET_AVAILABLE

        if not self.allow_chunking:
            return BudgetStatus.BUDGET_EXHAUSTED

        new_total = context.maybe_chunk(tokens_remaining)
        if new_total > tokens_remaining:
            raise ValueError(
                "Chunked total context length exceeds remaining budget: "
                f"{new_total} > {tokens_remaining}"
            )

        if new_total == tokens_remaining:
            return BudgetStatus.BUDGET_REACHED

        if new_total == 0:
            return BudgetStatus.BUDGET_REACHED

        return BudgetStatus.BUDGET_AVAILABLE

    def add_to_budget(self, context: TextGenerationContext) -> None:
        """Charge the budget for an accepted context's total length.

        This should be called only after :meth:`status_after_context` has
        returned a non-:data:`BudgetStatus.BUDGET_EXHAUSTED` result for the
        same ``context``.

        **Side effect**:
            Increments :attr:`used` by ``context.current_length``.

        Args:
            context: The context that was just admitted into the batch.
        """
        self.used += context.current_length
