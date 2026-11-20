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
"""The plan the API server serves with, from what the model worker reported."""

from __future__ import annotations

import multiprocessing

from max.pipelines.lib.memory_plan import MemoryPlan
from max.serve.pipelines.model_worker import _served_plan
from max.serve.process_control import Liveness


def _plan(max_length: int, batch: int, total_tokens: int | None) -> MemoryPlan:
    return MemoryPlan(
        planned_max_batch_size=batch,
        footprint=0,
        planned_max_length=max_length,
        planned_max_batch_total_tokens=total_tokens,
    )


def test_served_plan__takes_the_bounds_the_worker_reported() -> None:
    liveness = Liveness(multiprocessing.get_context("spawn"))
    liveness.report_plan(65536, 128)
    assert liveness.reported_plan() == (65536, 128)
    served = _served_plan(_plan(131072, 512, 131072), liveness)
    assert served is not None
    assert served.planned_max_length == 65536
    assert served.planned_max_batch_size == 128
    # A total-token cap that tracked max_length follows it down.
    assert served.planned_max_batch_total_tokens == 65536


def test_served_plan__keeps_a_user_total_token_cap() -> None:
    liveness = Liveness(multiprocessing.get_context("spawn"))
    liveness.report_plan(65536, 128)
    served = _served_plan(_plan(131072, 512, 200000), liveness)
    assert served is not None
    assert served.planned_max_batch_total_tokens == 200000


def test_served_plan__without_a_report_keeps_the_parent_plan() -> None:
    liveness = Liveness(multiprocessing.get_context("spawn"))
    assert liveness.reported_plan() == (None, None)
    plan = _plan(131072, 512, None)
    assert _served_plan(plan, liveness) is plan
    assert _served_plan(None, liveness) is None
