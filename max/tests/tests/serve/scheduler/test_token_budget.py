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

import numpy as np
from max.interfaces import TokenBuffer
from max.pipelines.core import TextContext
from max.serve.scheduler.batch_constructor.token_budget import (
    ActiveTokenBudget,
    BudgetStatus,
    RequestType,
    TotalContextTokenBudget,
)


def test_token_budget__active_token_budget_with_chunking() -> None:
    # We have a budget of 100 tokens, and we are allowing chunking
    active_token_budget = ActiveTokenBudget(
        capacity=100,
        allow_chunking=True,
        applicable_types=[RequestType.CE, RequestType.TG, RequestType.MIXED],
    )

    for i in range(11):
        context = TextContext(
            tokens=TokenBuffer(np.ones(10, dtype=np.int64)), max_length=100
        )

        status = active_token_budget.status_after_context(
            context, request_type=RequestType.CE
        )

        if i < 9:
            assert status == BudgetStatus.BUDGET_AVAILABLE
            active_token_budget.add_to_budget(
                context, request_type=RequestType.CE
            )
        elif i == 9:
            # The tenth context should hit the budget limit exactly
            assert status == BudgetStatus.BUDGET_REACHED
            active_token_budget.add_to_budget(
                context, request_type=RequestType.CE
            )
        else:
            # There is no room left in the budget
            assert status == BudgetStatus.BUDGET_EXHAUSTED

    assert active_token_budget.remaining == 0


def test_token_budget__active_token_budget_without_chunking() -> None:
    active_token_budget = ActiveTokenBudget(
        capacity=100, allow_chunking=False, applicable_types=[RequestType.CE]
    )

    for i in range(11):
        context = TextContext(
            tokens=TokenBuffer(np.ones(11, dtype=np.int64)), max_length=100
        )

        status = active_token_budget.status_after_context(
            context, request_type=RequestType.CE
        )

        if i < 9:
            assert status == BudgetStatus.BUDGET_AVAILABLE
            active_token_budget.add_to_budget(
                context, request_type=RequestType.CE
            )
        elif i == 9:
            assert status == BudgetStatus.BUDGET_REACHED
            active_token_budget.add_to_budget(
                context, request_type=RequestType.CE
            )
        else:
            assert status == BudgetStatus.BUDGET_EXHAUSTED

    # This is a soft limit, so we should be able to go over the budget by a few tokens.
    assert active_token_budget.remaining < 0


def test_token_budget__total_context_budget_with_cost_alignment_alignment() -> (
    None
):
    """TotalContextTokenBudget should align the effective cost to the page size."""

    total_context_budget = TotalContextTokenBudget(
        capacity=30,
        allow_chunking=False,
        applicable_types=[RequestType.CE],
        cost_alignment=8,  # token cost is aligned to 8 for this test
    )

    # current_length = 10, num_steps = 3 => total_length = 10 + (3 - 1) = 12
    # aligned cost = align_up(12, 8) = 16
    context = TextContext(
        tokens=TokenBuffer(np.ones(10, dtype=np.int64)), max_length=100
    )
    status = total_context_budget.status_after_context(
        context, num_steps=3, request_type=RequestType.CE
    )
    assert status == BudgetStatus.BUDGET_AVAILABLE

    # Commit the aligned length to the budget:
    # used = 16, remaining = 30 - 16 = 14.
    total_context_budget.add_to_budget(
        context, num_steps=3, request_type=RequestType.CE
    )
    assert total_context_budget.remaining == 14

    # Now with remaining=14, a new 10-token context and num_steps=4 gives
    # total_length = 10 + (4 - 1) = 13, aligned cost = align_up(13, 8) = 16
    # which is greater than remaining=14, so it should be rejected.
    context2 = TextContext(
        tokens=TokenBuffer(np.ones(10, dtype=np.int64)), max_length=100
    )
    status2 = total_context_budget.status_after_context(
        context2, num_steps=4, request_type=RequestType.CE
    )
    assert status2 == BudgetStatus.BUDGET_EXHAUSTED


def test_token_budget__total_context_budget_num_steps_available_and_reached() -> (
    None
):
    """TotalContextTokenBudget should account for num_steps when checking capacity."""
    # Capacity large enough to admit two 10-token contexts, accounting for
    # num_steps and page-size alignment.
    total_budget = TotalContextTokenBudget(
        capacity=25,
        allow_chunking=False,
        applicable_types=[RequestType.CE, RequestType.TG, RequestType.MIXED],
        cost_alignment=1,  # no alignment needed for this test
    )

    # current_length = 10, num_steps = 3 => total_length = 10 + (3 - 1) = 12 < 25
    context = TextContext(
        tokens=TokenBuffer(np.ones(10, dtype=np.int64)), max_length=100
    )
    status = total_budget.status_after_context(
        context, num_steps=3, request_type=RequestType.CE
    )
    assert status == BudgetStatus.BUDGET_AVAILABLE

    # Commit the current length plus (num_steps - 1) to the budget:
    # used = 10 + (3 - 1) = 12, remaining = 25 - 12 = 13.
    total_budget.add_to_budget(
        context, num_steps=3, request_type=RequestType.CE
    )
    assert total_budget.remaining == 13

    # Now with remaining=13, a new 10-token context and num_steps=4 gives
    # total_length = 10 + (4 - 1) = 13, which should exactly reach the budget.
    context2 = TextContext(
        tokens=TokenBuffer(np.ones(10, dtype=np.int64)), max_length=100
    )
    status2 = total_budget.status_after_context(
        context2, num_steps=4, request_type=RequestType.CE
    )
    assert status2 == BudgetStatus.BUDGET_REACHED


def test_token_budget__total_context_budget_num_steps_exhausted() -> None:
    """TotalContextTokenBudget should reject contexts that would overflow with num_steps > 1."""
    total_budget = TotalContextTokenBudget(
        capacity=14,
        allow_chunking=False,
        applicable_types=[RequestType.CE],
        cost_alignment=1,  # no alignment needed for this test
    )

    # current_length = 10, num_steps = 6 => total_length = 10 + (6 - 1) = 15 > 14
    context = TextContext(
        tokens=TokenBuffer(np.ones(10, dtype=np.int64)), max_length=100
    )
    status = total_budget.status_after_context(
        context, num_steps=6, request_type=RequestType.CE
    )
    assert status == BudgetStatus.BUDGET_EXHAUSTED


def test_token_budget__total_context_budget_with_chunking_exhausts_overage() -> (
    None
):
    """TotalContextTokenBudget should not chunk; total-context overage is exhausted."""
    total_budget = TotalContextTokenBudget(
        capacity=20,
        allow_chunking=True,
        applicable_types=[RequestType.CE],
        cost_alignment=1,  # no alignment needed for this test
    )

    context = TextContext(
        tokens=TokenBuffer(np.ones(30, dtype=np.int64)), max_length=100
    )
    assert len(context.tokens) == 30
    assert context.tokens.active_length == 30

    status = total_budget.status_after_context(
        context, num_steps=1, request_type=RequestType.CE
    )
    assert status == BudgetStatus.BUDGET_EXHAUSTED
    assert len(context.tokens) == 30
    assert context.tokens.active_length == 30
    assert total_budget.used == 0
    assert total_budget.remaining == 20


def test_token_budget__total_context_budget_overage_does_not_chunk() -> None:
    """TotalContextTokenBudget should not chunk even when remaining exceeds active_length."""
    total_budget = TotalContextTokenBudget(
        capacity=100,
        allow_chunking=True,
        applicable_types=[RequestType.CE],
        cost_alignment=1,  # no alignment needed for this test
    )

    context = TextContext(
        tokens=TokenBuffer(np.ones(200, dtype=np.int64)), max_length=300
    )
    context.tokens.skip_processing(190)

    assert len(context.tokens) == 200
    assert context.tokens.active_length == 10

    status = total_budget.status_after_context(
        context, num_steps=1, request_type=RequestType.CE
    )
    assert status == BudgetStatus.BUDGET_EXHAUSTED


def test_token_budget__total_context_budget_chunking_disabled_for_unit_active_length() -> (
    None
):
    """Chunking is not applied when active_length == 1, even if allow_chunking is True."""
    total_budget = TotalContextTokenBudget(
        capacity=10,
        allow_chunking=True,
        applicable_types=[RequestType.CE],
        cost_alignment=1,  # no alignment needed for this test
    )

    # Create a context where only a single token is active, but the total
    # sequence length is much larger (simulating TG-style usage).
    context = TextContext(
        tokens=TokenBuffer(np.ones(50, dtype=np.int64)), max_length=100
    )
    context.tokens.skip_processing(49)

    assert context.tokens.active_length == 1
    assert len(context.tokens) == 50

    status = total_budget.status_after_context(
        context, num_steps=1, request_type=RequestType.CE
    )
    # Since active_length == 1, TotalContextTokenBudget should not attempt
    # chunking and must report the budget as exhausted.
    assert status == BudgetStatus.BUDGET_EXHAUSTED
    assert total_budget.used == 0
    assert total_budget.remaining == 10


def test_token_budget__total_context_budget__ce_after_tg() -> None:
    # This scenario, would only happen with mixed batching
    total_budget = TotalContextTokenBudget(
        capacity=10,
        allow_chunking=True,
        applicable_types=[RequestType.MIXED, RequestType.CE],
        cost_alignment=1,  # no alignment needed for this test
    )

    # Create
    context = TextContext(
        tokens=TokenBuffer(np.ones(20, dtype=np.int64)), max_length=100
    )
    # Move it to the end, assuming it is a TG request
    context.tokens.skip_processing(19)

    # This should still show as available, as its a TG request, and this budget does not apply
    status = total_budget.status_after_context(
        context, num_steps=1, request_type=RequestType.TG
    )
    assert status == BudgetStatus.BUDGET_AVAILABLE

    # Add to budget
    total_budget.add_to_budget(
        context, num_steps=1, request_type=RequestType.TG
    )

    # Create a new CE request with a small number of tokens
    context = TextContext(
        tokens=TokenBuffer(np.ones(3, dtype=np.int64)), max_length=100
    )

    # This should be rejected, as we already have a TG object in batch that is beyond the threshold.
    status = total_budget.status_after_context(
        context, num_steps=1, request_type=RequestType.CE
    )
    assert status == BudgetStatus.BUDGET_EXHAUSTED
