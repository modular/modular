# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import numpy as np
from max.pipelines.core import TextContext
from max.serve.scheduler.batch_constructor.token_budget import (
    ActiveTokenBudget,
    BudgetStatus,
    TotalContextTokenBudget,
)


def test_token_budget__active_token_budget_with_chunking() -> None:
    # We have a budget of 100 tokens, and we are allowing chunking
    active_token_budget = ActiveTokenBudget(capacity=100, allow_chunking=True)

    for i in range(11):
        context = TextContext(
            tokens=np.ones(10, dtype=np.int32), max_length=100
        )

        status = active_token_budget.status_after_context(context)

        if i < 9:
            assert status == BudgetStatus.BUDGET_AVAILABLE
            active_token_budget.add_to_budget(context)
        elif i == 9:
            # The tenth context should hit the budget limit exactly
            assert status == BudgetStatus.BUDGET_REACHED
            active_token_budget.add_to_budget(context)
        else:
            # There is no room left in the budget
            assert status == BudgetStatus.BUDGET_EXHAUSTED

    assert active_token_budget.remaining == 0


def test_token_budget__active_token_budget_without_chunking() -> None:
    active_token_budget = ActiveTokenBudget(capacity=100, allow_chunking=False)

    for i in range(11):
        context = TextContext(
            tokens=np.ones(11, dtype=np.int32), max_length=100
        )

        status = active_token_budget.status_after_context(context)

        if i < 9:
            assert status == BudgetStatus.BUDGET_AVAILABLE
            active_token_budget.add_to_budget(context)
        elif i == 9:
            assert status == BudgetStatus.BUDGET_REACHED
            active_token_budget.add_to_budget(context)
        else:
            assert status == BudgetStatus.BUDGET_EXHAUSTED

    # This is a soft limit, so we should be able to go over the budget by a few tokens.
    assert active_token_budget.remaining < 0


def test_token_budget__total_context_budget_num_steps_available_and_reached() -> (
    None
):
    """TotalContextTokenBudget should account for num_steps when checking capacity."""
    # Capacity large enough to admit two 10-token contexts, accounting for num_steps.
    total_budget = TotalContextTokenBudget(capacity=25, allow_chunking=False)

    # current_length = 10, num_steps = 3 => total_length = 10 + (3 - 1) = 12 < 25
    context = TextContext(tokens=np.ones(10, dtype=np.int32), max_length=100)
    status = total_budget.status_after_context(context, num_steps=3)
    assert status == BudgetStatus.BUDGET_AVAILABLE

    # Commit the current length plus (num_steps - 1) to the budget:
    # used = 10 + (3 - 1) = 12, remaining = 25 - 12 = 13.
    total_budget.add_to_budget(context, num_steps=3)
    assert total_budget.remaining == 13

    # Now with remaining=13, a new 10-token context and num_steps=4 gives
    # total_length = 10 + (4 - 1) = 13, which should exactly reach the budget.
    context2 = TextContext(tokens=np.ones(10, dtype=np.int32), max_length=100)
    status2 = total_budget.status_after_context(context2, num_steps=4)
    assert status2 == BudgetStatus.BUDGET_REACHED


def test_token_budget__total_context_budget_num_steps_exhausted() -> None:
    """TotalContextTokenBudget should reject contexts that would overflow with num_steps > 1."""
    total_budget = TotalContextTokenBudget(capacity=14, allow_chunking=False)

    # current_length = 10, num_steps = 6 => total_length = 10 + (6 - 1) = 15 > 14
    context = TextContext(tokens=np.ones(10, dtype=np.int32), max_length=100)
    status = total_budget.status_after_context(context, num_steps=6)
    assert status == BudgetStatus.BUDGET_EXHAUSTED


def test_token_budget__total_context_budget_with_chunking_shrinks_context() -> (
    None
):
    """TotalContextTokenBudget with chunking should trim contexts to fit capacity."""
    # Budget capacity is smaller than the incoming context length; chunking should
    # reduce the effective total length to exactly the remaining capacity.
    total_budget = TotalContextTokenBudget(capacity=20, allow_chunking=True)

    # Initial context is longer than the budget.
    context = TextContext(tokens=np.ones(30, dtype=np.int32), max_length=100)
    assert context.current_length == 30
    assert context.active_length == 30

    status = total_budget.status_after_context(context, num_steps=1)
    # Chunking should allow the context, exactly reaching the budget.
    assert status == BudgetStatus.BUDGET_REACHED

    # Context should have been chunked down to the remaining capacity.
    # Current length is not affected by chunking.
    assert context.current_length == 30
    assert context.active_length == 20

    # After committing, the budget should be fully used.
    total_budget.add_to_budget(context, num_steps=1)
    assert total_budget.used == 20
    assert total_budget.remaining == 0


def test_token_budget__total_context_budget_chunking_disabled_for_unit_active_length() -> (
    None
):
    """Chunking is not applied when active_length == 1, even if allow_chunking is True."""
    total_budget = TotalContextTokenBudget(capacity=10, allow_chunking=True)

    # Create a context where only a single token is active, but the total
    # sequence length is much larger (simulating TG-style usage).
    context = TextContext(tokens=np.ones(50, dtype=np.int32), max_length=100)
    context.skip_processing(49)

    assert context.active_length == 1
    assert context.current_length == 50

    status = total_budget.status_after_context(context, num_steps=1)
    # Since active_length == 1, TotalContextTokenBudget should not attempt
    # chunking and must report the budget as exhausted.
    assert status == BudgetStatus.BUDGET_EXHAUSTED
    assert total_budget.used == 0
    assert total_budget.remaining == 10
