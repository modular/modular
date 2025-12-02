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
