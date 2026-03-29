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
"""Unit tests for the pipeline scheduling framework.

Exercises the core framework abstractions without AMD kernel-specific imports:
OpNode, LoopBody, ASAP/ALAP scheduling, dependency graph construction,
phase recipes, edge rules, the schedule compiler, and LDS contention modeling.

Tests that exercise AMD-specific schedules (MI355X profiles, PingPongSchedule,
SingleBufferSchedule, etc.) live in test_pipeline_ldg.mojo alongside the
kernel code.
"""

from std.collections import List
from std.testing import assert_equal, assert_true

from pipeline.types import (
    DepEdge,
    DepKind,
    OpDesc,
    OpRole,
    Phase,
    ResourceKind,
)
from pipeline.dependency_graph import (
    LoopBody,
    OpNode,
)
from pipeline.config import (
    BlockSizing,
    FragOrder,
    LoopCarriedSpec,
    PipelineConfig,
    WarpStaggerRule,
)
from pipeline.phase_derivation import (
    single_buffer_prologue_recipe,
    single_buffer_epilogue_recipe,
)
from pipeline.program_builder import derive_safe_max_globals
from pipeline.schedulers import (
    _asap_lower_bound,
    _evaluate_makespan,
)


# =============================================================================
# Helpers: inline op constructors (framework-only, no AMD imports)
# =============================================================================


def _global_load(tag: Int, latency: Int = 200) -> OpNode:
    return OpNode(
        op=OpDesc.op(tag, ResourceKind.GLOBAL_MEM, latency, OpRole.GLOBAL_LOAD)
    )


def _frag_load(tag: Int, latency: Int = 20) -> OpNode:
    return OpNode(
        op=OpDesc.op(tag, ResourceKind.LDS, latency, OpRole.FRAGMENT_LOAD)
    )


def _mma(tag: Int, latency: Int = 16) -> OpNode:
    return OpNode(
        op=OpDesc.op(tag, ResourceKind.MMA_UNIT, latency, OpRole.COMPUTE)
    )


def _simple_config() -> PipelineConfig:
    """Minimal PipelineConfig for single-buffer tests."""
    return PipelineConfig(
        depth=1,
        prefetch=1,
        drain_passes=2,
        prologue_fill=1,
        loop_carried=LoopCarriedSpec.none(),
        block_sizing=BlockSizing.uniform(1),
        frag_order=FragOrder(b_before_a=False),
        m_mmas=1,
        n_mmas=1,
        num_halves=1,
        mma_serial=False,
        mma_latency=16,
        vm_per_load_a=1,
        vm_per_load_b=1,
        ch0_match_field=0,
        ch1_match_field=0,
        warp_stagger=WarpStaggerRule.none(),
    )


# =============================================================================
# OpNode tests
# =============================================================================


def test_opnode_resource_from_opdesc() raises:
    """OpNode.resource should match the OpDesc's resource kind."""
    var gl = _global_load(0)
    var fl = _frag_load(1)
    var m = _mma(2)

    assert_equal(
        gl.resource._value,
        ResourceKind.GLOBAL_MEM._value,
        "global_load resource",
    )
    assert_equal(fl.resource._value, ResourceKind.LDS._value, "frag resource")
    assert_equal(
        m.resource._value, ResourceKind.MMA_UNIT._value, "mma resource"
    )

    print("  test_opnode_resource_from_opdesc PASSED")


def test_opnode_latency_from_opdesc() raises:
    """OpNode.latency should default to the OpDesc's latency."""
    var gl = _global_load(0, latency=123)
    assert_equal(gl.latency, 123, "latency should come from OpDesc")

    # Explicit override.
    var custom = OpNode(
        op=OpDesc.op(0, ResourceKind.LDS, 20, OpRole.FRAGMENT_LOAD), latency=99
    )
    assert_equal(custom.latency, 99, "explicit latency should override")

    print("  test_opnode_latency_from_opdesc PASSED")


# =============================================================================
# ASAP / ALAP tests
# =============================================================================


def test_linear_chain_asap() raises:
    """ASAP on a linear chain: load(200) → frag(20) → mma(16)."""
    var body = LoopBody()
    body.ops.append(_global_load(0, latency=200))
    body.ops.append(_frag_load(1, latency=20))
    body.ops.append(_mma(2, latency=16))
    body.edges.append(DepEdge(0, 1, DepKind.FLOW, 0))
    body.edges.append(DepEdge(1, 2, DepKind.FLOW, 0))

    var asap = body.compute_asap()
    assert_equal(asap[0], 0, "load ASAP")
    assert_equal(asap[1], 200, "frag ASAP")
    assert_equal(asap[2], 220, "mma ASAP")

    print("  test_linear_chain_asap PASSED")


def test_diamond_asap() raises:
    """ASAP on a diamond: two loads fan-in to one mma."""
    var body = LoopBody()
    body.ops.append(_global_load(0, latency=200))  # load_a
    body.ops.append(_global_load(1, latency=100))  # load_b (shorter)
    body.ops.append(_mma(2, latency=16))
    body.edges.append(DepEdge(0, 2, DepKind.FLOW, 0))
    body.edges.append(DepEdge(1, 2, DepKind.FLOW, 0))

    var asap = body.compute_asap()
    assert_equal(asap[2], 200, "mma should wait for slower load")

    print("  test_diamond_asap PASSED")


def test_loop_carried_edges_skip_asap() raises:
    """Loop-carried edges (d>=1) should not affect ASAP within one iteration."""
    var body = LoopBody()
    body.ops.append(_global_load(0, latency=200))
    body.ops.append(_mma(1, latency=16))
    # d=1 edge: should NOT make mma depend on load.
    body.edges.append(DepEdge(0, 1, DepKind.FLOW, 1))

    var asap = body.compute_asap()
    assert_equal(asap[1], 0, "loop-carried edge should not constrain ASAP")

    print("  test_loop_carried_edges_skip_asap PASSED")


def test_compute_alap() raises:
    """ALAP on a linear chain: load(200) → frag(20) → mma(16)."""
    var body = LoopBody()
    body.ops.append(_global_load(0, latency=200))
    body.ops.append(_frag_load(1, latency=20))
    body.ops.append(_mma(2, latency=16))
    body.edges.append(DepEdge(0, 1, DepKind.FLOW, 0))
    body.edges.append(DepEdge(1, 2, DepKind.FLOW, 0))

    var asap = body.compute_asap()
    var makespan = 0
    for i in range(len(asap)):
        var end = asap[i] + body.ops[i].latency
        if end > makespan:
            makespan = end

    var alap = body.compute_alap(makespan)
    # On a linear chain with no slack, ALAP == ASAP.
    assert_equal(alap[0], 0, "load ALAP")
    assert_equal(alap[1], 200, "frag ALAP")
    assert_equal(alap[2], 220, "mma ALAP")

    print("  test_compute_alap PASSED")


def test_alap_slack() raises:
    """ALAP - ASAP gives scheduling slack (mobility)."""
    var body = LoopBody()
    body.ops.append(_global_load(0, latency=200))
    body.ops.append(_global_load(1, latency=50))  # independent, shorter
    body.ops.append(_mma(2, latency=16))
    # Only load_a feeds mma.
    body.edges.append(DepEdge(0, 2, DepKind.FLOW, 0))

    var asap = body.compute_asap()
    var makespan = 0
    for i in range(len(asap)):
        var end = asap[i] + body.ops[i].latency
        if end > makespan:
            makespan = end

    var alap = body.compute_alap(makespan)
    # load_b is independent: ASAP=0, ALAP = makespan - 50 = 166.
    var slack = alap[1] - asap[1]
    assert_true(slack > 0, "independent op should have positive slack")

    print("  test_alap_slack PASSED")


# =============================================================================
# Validation tests
# =============================================================================


def test_validate_catches_self_loop() raises:
    """Validation should catch a d=0 self-loop."""
    var body = LoopBody()
    body.ops.append(_global_load(0))
    body.edges.append(DepEdge(0, 0, DepKind.FLOW, 0))

    var caught = False
    try:
        body.validate()
    except:
        caught = True

    assert_true(caught, "validate() should raise on self-loop with d=0")
    print("  test_validate_catches_self_loop PASSED")


# =============================================================================
# LDS contention model tests
# =============================================================================


def test_lds_contention_zero_is_noop() raises:
    """LDS contention penalty=0 gives same makespan as default model."""
    var body = LoopBody()
    body.ops.append(_global_load(0, latency=200))
    body.ops.append(_frag_load(1, latency=20))
    body.ops.append(_mma(2, latency=16))
    body.edges.append(DepEdge(0, 1, DepKind.FLOW, 0))
    body.edges.append(DepEdge(1, 2, DepKind.FLOW, 0))

    var order = List[Int]()
    order.append(0)
    order.append(1)
    order.append(2)
    var ms_default = _evaluate_makespan(body, order)
    var ms_zero = _evaluate_makespan(body, order, 0)
    assert_equal(ms_default, ms_zero, "Penalty=0 should match default makespan")

    print("  test_lds_contention_zero_is_noop PASSED")


def test_lds_contention_penalty_increases_makespan() raises:
    """LDS contention penalty > 0 increases makespan when LDS ops overlap."""
    var body = LoopBody()
    body.ops.append(
        OpNode(op=OpDesc.op(0, ResourceKind.GLOBAL_MEM, 10, OpRole.GLOBAL_LOAD))
    )
    body.ops.append(
        OpNode(op=OpDesc.op(1, ResourceKind.GLOBAL_MEM, 10, OpRole.GLOBAL_LOAD))
    )
    body.ops.append(
        OpNode(op=OpDesc.op(3, ResourceKind.LDS, 20, OpRole.FRAGMENT_LOAD))
    )
    # No deps — all independent.
    var order = List[Int]()
    order.append(0)
    order.append(1)
    order.append(2)
    var ms_none = _evaluate_makespan(body, order, 0)
    var ms_pen = _evaluate_makespan(body, order, 15)

    assert_equal(ms_none, 20, "Without penalty: all overlap, makespan=20")
    assert_equal(ms_pen, 50, "With penalty=15: LDS ops serialize, makespan=50")

    print("  test_lds_contention_penalty_increases_makespan PASSED")


def test_lds_contention_lower_bound() raises:
    """LDS contention penalty updates the ASAP lower bound."""
    var body = LoopBody()
    body.ops.append(
        OpNode(op=OpDesc.op(0, ResourceKind.GLOBAL_MEM, 10, OpRole.GLOBAL_LOAD))
    )
    body.ops.append(
        OpNode(op=OpDesc.op(1, ResourceKind.GLOBAL_MEM, 10, OpRole.GLOBAL_LOAD))
    )
    body.ops.append(
        OpNode(op=OpDesc.op(3, ResourceKind.LDS, 10, OpRole.FRAGMENT_LOAD))
    )
    body.ops.append(
        OpNode(op=OpDesc.op(4, ResourceKind.LDS, 10, OpRole.FRAGMENT_LOAD))
    )
    body.ops.append(
        OpNode(op=OpDesc.op(5, ResourceKind.MMA_UNIT, 16, OpRole.COMPUTE))
    )

    var lb_none = _asap_lower_bound(body, 0)
    var lb_pen = _asap_lower_bound(body, 20)

    assert_equal(lb_none, 16, "Without penalty: lb = mma_total = 16")
    assert_equal(lb_pen, 80, "With penalty=20: lb = lds_total = 80")

    print("  test_lds_contention_lower_bound PASSED")


# =============================================================================
# derive_safe_max_globals tests
# =============================================================================


def test_derive_safe_max_globals_bf16() raises:
    """BF16 with num_k_mmas=2 should return max_globals=1 (safe)."""
    var result = derive_safe_max_globals(num_k_mmas=2)
    assert_equal(result, 1)
    print("  test_derive_safe_max_globals_bf16 PASSED")


def test_derive_safe_max_globals_fp8() raises:
    """FP8 with num_k_mmas=1 should return max_globals=0 (unsafe)."""
    var result = derive_safe_max_globals(num_k_mmas=1)
    assert_equal(result, 0)
    print("  test_derive_safe_max_globals_fp8 PASSED")


def test_derive_safe_max_globals_large_k() raises:
    """Large num_k_mmas (4) should return max_globals=1 (safe)."""
    var result = derive_safe_max_globals(num_k_mmas=4)
    assert_equal(result, 1)
    print("  test_derive_safe_max_globals_large_k PASSED")


# =============================================================================
# Phase recipe tests
# =============================================================================


def test_phase_recipe_prologue_structure() raises:
    """Single-buffer prologue recipe should produce expected step count."""
    var recipe = single_buffer_prologue_recipe()
    # Recipe: load, store, barrier, load(prefetch), lc_frag, fence = 6 steps.
    assert_equal(len(recipe), 6, "prologue recipe should have 6 steps")

    print("  test_phase_recipe_prologue_structure PASSED")


def test_phase_recipe_epilogue_structure() raises:
    """Single-buffer epilogue recipe should produce expected step count."""
    var recipe = single_buffer_epilogue_recipe()
    # Drain 1: fence, non-lc frags, barrier, store, all computes, fence = 6
    # Drain 2: barrier, all frags, all computes, fence = 4
    # Total = 10 steps.
    assert_equal(len(recipe), 10, "epilogue recipe should have 10 steps")

    print("  test_phase_recipe_epilogue_structure PASSED")


# =============================================================================
# Main
# =============================================================================


def main() raises:
    # OpNode tests
    print("\n  --- OpNode Tests ---")
    test_opnode_resource_from_opdesc()
    test_opnode_latency_from_opdesc()

    # ASAP / ALAP tests
    print("\n  --- ASAP / ALAP Tests ---")
    test_linear_chain_asap()
    test_diamond_asap()
    test_loop_carried_edges_skip_asap()
    test_compute_alap()
    test_alap_slack()

    # Validation tests
    print("\n  --- Validation Tests ---")
    test_validate_catches_self_loop()

    # LDS contention model tests
    print("\n  --- LDS Contention Model Tests ---")
    test_lds_contention_zero_is_noop()
    test_lds_contention_penalty_increases_makespan()
    test_lds_contention_lower_bound()

    # derive_safe_max_globals tests
    print("\n  --- derive_safe_max_globals Tests ---")
    test_derive_safe_max_globals_bf16()
    test_derive_safe_max_globals_fp8()
    test_derive_safe_max_globals_large_k()

    # Phase recipe tests
    print("\n  --- Phase Recipe Tests ---")
    test_phase_recipe_prologue_structure()
    test_phase_recipe_epilogue_structure()

    print("\nAll pipeline framework tests passed!")
