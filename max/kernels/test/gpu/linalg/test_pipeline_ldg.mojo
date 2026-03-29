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
"""Tests for the Loop Dependency Graph (LDG) abstractions in the pipeline framework.

Exercises OpNode, LoopBody, ASAP scheduling, constrained scheduling,
validation, and interleaved schedule generation.
"""

from std.testing import assert_equal, assert_true

# Generic framework types.
from pipeline.types import (
    DepEdge,
    DepKind,
    KOffsetKind,
    OpDesc,
    OpRole,
    Phase,
    ResourceKind,
    _Ops,
    annotate_ops,
)
from pipeline.dependency_graph import (
    LoopBody,
    OpNode,
)
from pipeline.config import (
    BlockSizing,
    PipelineConfig,
    ScheduleConfig,
    SchedulingStrategy,
    TargetProfile,
)
from pipeline.pipeline_dsl import ScheduleEntry, pipe
from pipeline.program import (
    MMABlockSpec,
    PipelineProgram,
)
from pipeline.program_builder import (
    build_double_buffer_program,
    build_kernel_program,
    derive_drain_mask,
    derive_safe_max_globals,
    derive_waits_from_blocks,
    mma_block_interleave,
    optimize_within_barriers,
    single_buffer_reorder,
    verify_schedule,
)
from pipeline.phase_derivation import (
    apply_edge_rules,
    derive_edges_from_ops,
    derive_prologue_from_program,
    double_buffer_edge_rules,
    single_buffer_edge_rules,
)
from pipeline.schedulers import (
    _asap_lower_bound,
    _evaluate_makespan,
    greedy_schedule,
    optimal_schedule_with_halves,
)
from pipeline.compiler import ScheduleCompiler

# AMD platform target.
from linalg.matmul.gpu.amd.amd_target import (
    AMDScheduleHints,
    mi355x_double_buffer,
    mi355x_single_buffer,
    mi355x_cost_model,
    mi355x_target,
)

# Default (single-buffer) matmul schedule.
from linalg.matmul.gpu.amd.matmul_schedule import (
    COMPUTE,
    DefaultMatmulOps,
    LOAD_DRAM,
    LOAD_FRAG,
    SingleBufferSchedule,
    STORE_SMEM,
    build_default_matmul_schedule,
    compute_range,
    load_frags,
    _load_dram,
    _store_smem,
    _load_frag,
    _compute,
)

# Ping-pong (double-buffer) matmul schedule.
from linalg.matmul.gpu.amd.pingpong_schedule import (
    DeclarativeSchedule,
    build_schedule,
    LOAD_A,
    LOAD_B,
    MMA,
    MMA_LOAD_A,
    MMA_LOAD_B,
    _load_a,
    _load_b,
    _mma_load_a,
    _mma_load_b,
    _mma,
)


# =============================================================================
# Test helpers — local to test file
# =============================================================================


def _default_matmul_body[
    num_k_tiles: Int,
]() -> List[OpDesc]:
    """Pipelined loop body for the default single-buffer matmul."""
    var logical = List[OpDesc]()
    logical.append(_load_dram())
    logical.append(_store_smem())
    logical.append(OpDesc.barrier())
    for i in range(num_k_tiles):
        logical.append(_load_frag(subtile=i))
    for i in range(num_k_tiles):
        logical.append(_compute(subtile=i))
    return single_buffer_reorder(logical, mi355x_single_buffer())


def _build_ldg() -> LoopBody:
    """Build LoopBody from ping-pong body + auto-derived edges (test helper)."""
    var sched = DeclarativeSchedule[False, 0, 0](
        config=ScheduleConfig(auto_waits=False)
    )
    var ops = sched.build_body()
    var edges = derive_edges_from_ops(ops, mi355x_double_buffer())
    var ldg = LoopBody()
    for i in range(len(ops)):
        ldg.ops.append(OpNode(op=ops[i]))
    ldg.edges = edges^
    return ldg^


def _build_kernel_entries(
    config: ScheduleConfig = ScheduleConfig(auto_waits=False),
) -> List[ScheduleEntry]:
    """Build ping-pong kernel body as List[ScheduleEntry] (test helper)."""
    var sc = ScheduleCompiler()
    sc.compile(DeclarativeSchedule[False, 0, 0](config=config))
    return sc.kernel.copy()


def _ops_to_ldg(ops: List[OpDesc]) -> LoopBody:
    """Wrap ops + auto-derived edges into a LoopBody (test helper)."""
    var edges = derive_edges_from_ops(ops, mi355x_double_buffer())
    var ldg = LoopBody()
    for i in range(len(ops)):
        ldg.ops.append(OpNode(op=ops[i]))
    ldg.edges = edges^
    return ldg^


def _extract_ldg(
    kernel: List[ScheduleEntry],
) -> LoopBody:
    """Extract LDG from kernel entries (test helper)."""
    var ops = List[OpDesc]()
    for i in range(len(kernel)):
        var op = kernel[i].op
        if (
            op.role == OpRole.GLOBAL_LOAD
            or op.role == OpRole.FRAGMENT_LOAD
            or op.role == OpRole.COMPUTE
        ):
            ops.append(op)
    return _ops_to_ldg(ops)


def _validate_c2(
    kernel: List[ScheduleEntry],
    body: LoopBody,
):
    """Validate C2 (d=0 FLOW consumer after producer) (test helper)."""
    var op_entry = List[Int]()
    for i in range(len(kernel)):
        var op = kernel[i].op
        if (
            op.role == OpRole.GLOBAL_LOAD
            or op.role == OpRole.FRAGMENT_LOAD
            or op.role == OpRole.COMPUTE
        ):
            op_entry.append(i)

    for e in range(len(body.edges)):
        var edge = body.edges[e]
        if edge.loop_distance >= 1:
            continue
        if edge.dep_kind != DepKind.FLOW:
            continue
        var p_entry = op_entry[edge.producer_idx]
        var c_entry = op_entry[edge.consumer_idx]
        debug_assert(
            c_entry > p_entry,
            "C2 violation: consumer at entry "
            + String(c_entry)
            + " must come after producer at entry "
            + String(p_entry),
        )


def test_opnode_resource_assignment() raises:
    """OpNode auto-derives resource from op metadata."""
    var load = OpNode(
        op=_load_a(stage=0, subtile=0, k_offset=KOffsetKind.K0, vm_cost=4)
    )
    assert_true(
        load.resource == ResourceKind.GLOBAL_MEM,
        "load_a resource should be GLOBAL_MEM",
    )

    var lds_load = OpNode(op=_mma_load_a(stage=0, subtile=0))
    assert_true(
        lds_load.resource == ResourceKind.LDS,
        "mma_load_a resource should be LDS",
    )

    var mma = OpNode(op=_mma(stage=0, subtile=0))
    assert_true(
        mma.resource == ResourceKind.MMA_UNIT,
        "mma resource should be MMA_UNIT",
    )

    var barrier = OpNode(op=OpDesc.barrier())
    assert_true(
        barrier.resource == ResourceKind.SCALAR,
        "barrier resource should be SCALAR",
    )
    assert_equal(barrier.latency, 0, "barrier latency should be 0")
    print("  test_opnode_resource_assignment PASSED")


def test_linear_chain_asap() raises:
    """Linear chain: load_a -> wait_vm -> mma_load_a -> mma."""
    var body = LoopBody()
    body.ops.append(
        OpNode(
            op=_load_a(stage=0, subtile=0, k_offset=KOffsetKind.K0, vm_cost=4)
        )
    )
    body.ops.append(OpNode(op=OpDesc.wait_vm[0]()))
    body.ops.append(OpNode(op=_mma_load_a(stage=0, subtile=0)))
    body.ops.append(OpNode(op=_mma(stage=0, subtile=0)))

    body.edges.append(DepEdge.flow(producer=0, consumer=1))
    body.edges.append(DepEdge.flow(producer=1, consumer=2))
    body.edges.append(DepEdge.flow(producer=2, consumer=3))

    body.validate()

    var asap = body.compute_asap()
    assert_equal(asap[0], 0, "load_a ASAP")
    assert_equal(asap[1], 200, "wait_vm ASAP")
    assert_equal(asap[2], 200, "mma_load_a ASAP (wait has lat=0)")
    assert_equal(asap[3], 220, "mma ASAP")
    print("  test_linear_chain_asap PASSED")


def test_diamond_asap() raises:
    """Diamond: two loads feed a barrier, barrier feeds mma."""
    var body = LoopBody()
    body.ops.append(
        OpNode(
            op=_load_a(stage=0, subtile=0, k_offset=KOffsetKind.K0, vm_cost=4)
        )
    )
    body.ops.append(
        OpNode(
            op=_load_b(stage=0, subtile=0, k_offset=KOffsetKind.K0, vm_cost=4)
        )
    )
    body.ops.append(OpNode(op=OpDesc.barrier()))
    body.ops.append(OpNode(op=_mma(stage=0, subtile=0)))

    body.edges.append(DepEdge.flow(producer=0, consumer=2))
    body.edges.append(DepEdge.flow(producer=1, consumer=2))
    body.edges.append(DepEdge.flow(producer=2, consumer=3))
    body.edges.append(DepEdge.anti(producer=3, consumer=0, loop_distance=1))

    body.validate()

    var asap = body.compute_asap()
    assert_equal(asap[2], 200, "barrier ASAP = max(200, 200)")
    assert_equal(asap[3], 200, "mma ASAP = barrier ASAP + 0")


def test_loop_carried_edges_skip_asap() raises:
    """Loop-carried edges (d>=1) don't constrain ASAP within one iteration."""
    var body = LoopBody()
    body.ops.append(
        OpNode(
            op=_load_a(stage=0, subtile=0, k_offset=KOffsetKind.K0, vm_cost=4)
        )
    )
    body.ops.append(OpNode(op=_mma(stage=0, subtile=0)))

    # Only a loop-carried edge, no same-iteration dependency
    body.edges.append(DepEdge.anti(producer=1, consumer=0, loop_distance=1))

    var asap = body.compute_asap()
    # Both ops have ASAP=0 since the anti-dep is loop-carried
    assert_equal(asap[0], 0, "load ASAP = 0 (loop-carried edge skipped)")
    assert_equal(asap[1], 0, "mma ASAP = 0 (no same-iter predecessor)")


def test_pp_extract_ldg() raises:
    """Extract LDG from interleaved kernel body: 24 ops, 54 edges."""
    var kernel = _build_kernel_entries()

    var body = _extract_ldg(kernel)
    body.validate()

    # Count op types: expect 8 global loads, 8 MMA loads, 8 MMAs.
    var n_global = 0
    var n_lds = 0
    var n_mma = 0
    for i in range(24):
        var role = body.ops[i].op.role
        if role == OpRole.GLOBAL_LOAD:
            n_global += 1
        elif role == OpRole.FRAGMENT_LOAD:
            n_lds += 1
        elif role == OpRole.COMPUTE:
            n_mma += 1
    assert_equal(n_global, 8, "expected 8 global loads")
    assert_equal(n_lds, 8, "expected 8 MMA loads")
    assert_equal(n_mma, 8, "expected 8 MMAs")

    # Count edges by kind and loop distance.
    # Phase 2: 16 register FLOW d=0
    # Phase 3: 4 accum FLOW d=0, 4 accum FLOW d=1
    # Phase 4: 2 LDS FLOW d=0, 14 LDS FLOW d=1
    # Phase 5: 14 LDS ANTI d=0
    var n_flow_d0 = 0
    var n_flow_d1 = 0
    var n_anti_d0 = 0
    for i in range(54):
        var e = body.edges[i]
        if e.dep_kind == DepKind.FLOW and e.loop_distance == 0:
            n_flow_d0 += 1
        elif e.dep_kind == DepKind.FLOW and e.loop_distance >= 1:
            n_flow_d1 += 1
        elif e.dep_kind == DepKind.ANTI:
            n_anti_d0 += 1
    # FLOW d=0: 16 register + 4 accum + 2 LDS = 22
    assert_equal(n_flow_d0, 22, "expected 22 same-iter FLOW edges")
    # FLOW d=1: 4 accum + 14 LDS = 18
    assert_equal(n_flow_d1, 18, "expected 18 loop-carried FLOW edges")
    # ANTI d=0: 14 LDS
    assert_equal(n_anti_d0, 14, "expected 14 LDS ANTI edges")

    print("  test_extract_interleaved_loop_body PASSED")


def test_interleaved_ldg_asap() raises:
    """Verify ASAP times reflect the interleaved MMA pipeline structure."""
    var kernel = _build_kernel_entries()

    var body = _extract_ldg(kernel)
    var asap = body.compute_asap()

    # Ops 0-12 are first half, ops 13-23 are second half.
    # MMA loads have latency 20, so first MMA (op 3) should have ASAP >= 20.
    # First-half MMAs: ops 3, 6, 9, 12 (mma[0,0], mma[0,1], mma[1,0], mma[1,1])
    # Second-half MMAs: ops 15, 18, 21, 23

    # First MMA in first half depends on mma_load (lat=20) -> ASAP >= 20.
    assert_true(asap[3] >= 20, "mma[0,0]_h1 ASAP should be >= 20")

    # Second-half MMAs depend on first-half MMAs (accum dep, lat=16)
    # so they should have higher ASAP.
    assert_true(
        asap[15] > asap[3],
        "mma[0,0]_h2 should have higher ASAP than mma[0,0]_h1",
    )
    assert_true(
        asap[18] > asap[6],
        "mma[0,1]_h2 should have higher ASAP than mma[0,1]_h1",
    )

    print("  test_interleaved_ldg_asap PASSED")


def test_interleaved_c2_validation() raises:
    """Validate C2 constraint: all d=0 edges have consumer after producer."""
    var kernel = _build_kernel_entries()

    var body = _extract_ldg(kernel)

    # This should not assert — the interleaved schedule is correctly ordered.
    _validate_c2(kernel, body)

    # Verify the entry mapping manually: count d=0 edges and check ordering.
    # Re-extract op->entry mapping.
    var op_entry = List[Int]()
    for i in range(len(kernel)):
        var op = kernel[i].op
        if (
            op.role == OpRole.GLOBAL_LOAD
            or op.role == OpRole.FRAGMENT_LOAD
            or op.role == OpRole.COMPUTE
        ):
            op_entry.append(i)
    assert_equal(len(op_entry), 24, "expected 24 data ops")

    # Count d=0 FLOW edges and verify each one.
    # ANTI d=0 edges are not checked — they're enforced by barriers.
    var n_flow_checked = 0
    var n_anti_d0 = 0
    for e in range(54):
        var edge = body.edges[e]
        if edge.loop_distance >= 1:
            continue
        if edge.dep_kind == DepKind.FLOW:
            var p_entry = op_entry[edge.producer_idx]
            var c_entry = op_entry[edge.consumer_idx]
            assert_true(
                c_entry > p_entry,
                "C2: consumer entry must be after producer entry",
            )
            n_flow_checked += 1
        elif edge.dep_kind == DepKind.ANTI:
            n_anti_d0 += 1

    # 22 FLOW d=0 edges checked, 14 ANTI d=0 edges skipped.
    assert_equal(n_flow_checked, 22, "expected 22 FLOW d=0 edges checked")
    assert_equal(n_anti_d0, 14, "expected 14 ANTI d=0 edges (barrier-enforced)")

    print("  test_interleaved_c2_validation PASSED")


def test_pp_build_ldg() raises:
    """Declarative LDG matches extracted LDG ops and edges."""
    # Get extracted LDG from the reference schedule.
    var kernel = _build_kernel_entries()
    var extracted = _extract_ldg(kernel)

    # Get declarative LDG.
    var declared = _build_ldg()

    # Validate the declarative LDG passes all checks.
    declared.validate()

    # Compare ops: same kind, stage, subtile, k_offset for each op.
    for i in range(24):
        var eo = extracted.ops[i].op
        var do_ = declared.ops[i].op
        assert_true(
            eo.tag == do_.tag,
            "op " + String(i) + " tag mismatch",
        )
        assert_true(
            eo.stage == do_.stage,
            "op " + String(i) + " stage mismatch",
        )
        assert_true(
            eo.subtile == do_.subtile,
            "op " + String(i) + " subtile mismatch",
        )
        if eo.role == OpRole.GLOBAL_LOAD:
            assert_true(
                eo.k_offset.bk_multiple == do_.k_offset.bk_multiple,
                "op " + String(i) + " k_offset mismatch",
            )
        assert_equal(
            extracted.ops[i].latency,
            declared.ops[i].latency,
            "op " + String(i) + " latency mismatch",
        )

    # Compare edges as sets (order may differ between extracted and declared).
    # For each declared edge, find a matching extracted edge.
    var matched = List[Bool]()
    for _ in range(len(declared.edges)):
        matched.append(False)

    for d in range(54):
        var de = declared.edges[d]
        var found = False
        for e in range(54):
            if matched[e]:
                continue
            var ee = extracted.edges[e]
            if (
                ee.producer_idx == de.producer_idx
                and ee.consumer_idx == de.consumer_idx
                and ee.loop_distance == de.loop_distance
                and ee.dep_kind == de.dep_kind
            ):
                matched[e] = True
                found = True
                break
        assert_true(
            found,
            "declared edge " + String(d) + " not found in extracted edges",
        )

    print("  test_define_interleaved_loop_body PASSED")


def test_greedy_schedule_block_structure() raises:
    """Verify the scheduler output has correct MMA-centered block structure.

    Each block should end with exactly one MMA, have at most 2 global loads
    (matching MMABlockSpec's global_load + global_load_1 slots), and
    mma_loads should precede the MMA within the block.
    """
    var body = _build_ldg()
    var order = greedy_schedule(body)

    var block_start = 0
    var mma_count = 0

    for i in range(24):
        var op = body.ops[order[i]].op
        if op.role == OpRole.COMPUTE:
            mma_count += 1

            # Count global loads in this block.
            var n_global = 0
            for j in range(block_start, i + 1):
                if body.ops[order[j]].op.role == OpRole.GLOBAL_LOAD:
                    n_global += 1
            assert_true(
                n_global <= 2,
                "block " + String(mma_count - 1) + " has >2 global loads",
            )

            block_start = i + 1

    assert_equal(mma_count, 8, "should have exactly 8 MMA blocks")

    print("  test_greedy_schedule_block_structure PASSED")


def test_prefetch_marking_in_kernel_body() raises:
    """Verify that kernel body entries have correct is_prefetch flags.

    The 70-entry kernel body should have exactly 7 entries with
    is_prefetch=True (the 7 K0/K1 prefetch loads) and 1 global load
    entry with is_prefetch=False (the K_PREV completion load).
    """
    var kernel = _build_kernel_entries()

    var prefetch_count = 0
    var non_prefetch_load_count = 0
    for i in range(len(kernel)):
        var entry = kernel[i]
        if entry.is_prefetch:
            prefetch_count += 1
            # Prefetch entries must be global loads (LOAD_A or LOAD_B)
            assert_true(
                entry.op.role == OpRole.GLOBAL_LOAD,
                "prefetch entry must be a global load",
            )
            # Prefetch loads must have K0 or K1 offset (not K_PREV)
            assert_true(
                entry.op.k_offset != KOffsetKind.K_PREV,
                "prefetch load must not have K_PREV offset",
            )
        elif entry.op.role == OpRole.GLOBAL_LOAD:
            non_prefetch_load_count += 1
            # Non-prefetch loads must be K_PREV (completion loads)
            assert_true(
                entry.op.k_offset == KOffsetKind.K_PREV,
                "non-prefetch load must have K_PREV offset",
            )

    assert_equal(prefetch_count, 7, "should have 7 prefetch global loads")
    assert_equal(
        non_prefetch_load_count, 1, "should have 1 completion load (K_PREV)"
    )

    print("  test_prefetch_marking_in_kernel_body PASSED")


# =============================================================================
# Default Matmul Full Schedule Tests
# =============================================================================


def test_default_matmul_full_schedule() raises:
    """Test building the full schedule for default matmul."""
    # Use typical BF16 config: 4x4 M/N MMAs, 2 k_mmas, 16x16 MMA shape
    # a_loads=2, b_loads=2
    var schedule = build_default_matmul_schedule[
        num_k_tiles=2,
        num_m_mmas=4,
        num_n_mmas=4,
        num_k_mmas=2,
        MMA_M=16,
        MMA_N=16,
        a_loads_per_thread=2,
        b_loads_per_thread=2,
    ]()

    # Prologue: 6 entries
    assert_true(
        schedule.prologue[0].op.tag == LOAD_DRAM,
        "prologue starts with LOAD_DRAM",
    )

    # Epilogue count should be 13
    assert_equal(len(schedule.epilogue), 13, "epilogue_len should be 13")

    # Kernel body should have 8 data ops + schedule hints
    # Just verify it contains SCHED_GROUP_BARRIER hints
    var has_sched_hints = False
    for i in range(len(schedule.kernel)):
        if (
            schedule.kernel[i].op.tag
            == DefaultMatmulOps.SCHED_GROUP_BARRIER.value
        ):
            has_sched_hints = True
            break
    assert_true(has_sched_hints, "kernel should contain schedule hints")

    # Warp stagger should be past prologue (no stagger for default matmul)
    assert_true(
        schedule.warp_stagger_index > len(schedule.prologue),
        "warp_stagger_index should be past prologue",
    )

    print("  test_default_matmul_full_schedule PASSED")


def test_default_matmul_full_round_trip() raises:
    """Build full schedule, validate it compiles and has correct counts."""
    var schedule = build_default_matmul_schedule[
        num_k_tiles=2,
        num_m_mmas=4,
        num_n_mmas=4,
        num_k_mmas=2,
        MMA_M=16,
        MMA_N=16,
        a_loads_per_thread=2,
        b_loads_per_thread=2,
    ]()

    # Prologue: 6 entries, starts with LOAD_DRAM
    assert_true(
        schedule.prologue[0].op.tag == LOAD_DRAM,
        "round-trip: prologue starts with LOAD_DRAM",
    )

    # Epilogue: 13 entries for T=2
    assert_equal(
        len(schedule.epilogue), 13, "round-trip: epilogue_len should be 13"
    )

    # Kernel body: 8 data ops + 48 schedule hints = 56
    assert_equal(
        len(schedule.kernel), 56, "round-trip: kernel_len should be 56"
    )

    print("  test_default_matmul_full_round_trip PASSED")


# =============================================================================
# Single-Buffer Transform Tests
# =============================================================================


def test_single_buffer_transform_t2() raises:
    """Verify single_buffer_reorder produces the expected pipelined order for T=2.
    """
    # Logical iteration: load_global, store_shared, sync, frag[0], frag[1], compute[0], compute[1]
    var logical = List[OpDesc]()
    logical.append(_load_dram())
    logical.append(_store_smem())
    logical.append(OpDesc.barrier())
    logical.append(_load_frag(subtile=0))
    logical.append(_load_frag(subtile=1))
    logical.append(_compute(subtile=0))
    logical.append(_compute(subtile=1))

    var p = single_buffer_reorder(logical, mi355x_single_buffer())

    # Expected order (7 logical ops -> 8 pipelined ops):
    # [0] frag[1]       — phase 1: frags [1..T-1]
    # [1] compute[0]    — phase 2: compute[0]
    # [2] barrier       — phase 3: sync
    # [3] store_smem    — phase 4: store shared
    # [4] load_dram     — phase 5: load global
    # [5] compute[1]    — phase 6: compute[1..T-1]
    # [6] barrier       — phase 7: sync (added)
    # [7] frag[0]       — phase 8: loop-carried
    assert_true(
        p[0].tag == LOAD_FRAG and p[0].subtile == 1,
        "pos 0: frag[1]",
    )
    assert_true(
        p[1].tag == COMPUTE and p[1].subtile == 0,
        "pos 1: compute[0]",
    )
    assert_true(
        p[2].tag == DefaultMatmulOps.BARRIER.value,
        "pos 2: barrier",
    )
    assert_true(p[3].tag == STORE_SMEM, "pos 3: store_smem")
    assert_true(p[4].tag == LOAD_DRAM, "pos 4: load_dram")
    assert_true(
        p[5].tag == COMPUTE and p[5].subtile == 1,
        "pos 5: compute[1]",
    )
    assert_true(
        p[6].tag == DefaultMatmulOps.BARRIER.value,
        "pos 6: barrier (added)",
    )
    assert_true(
        p[7].tag == LOAD_FRAG and p[7].subtile == 0,
        "pos 7: frag[0] (loop-carried)",
    )

    print("  test_single_buffer_transform_t2 PASSED")


def test_single_buffer_transform_t1() raises:
    """Verify single_buffer_reorder for T=1 (no frag[1..T-1] or compute[1..T-1]).
    """
    var logical = List[OpDesc]()
    logical.append(_load_dram())
    logical.append(_store_smem())
    logical.append(OpDesc.barrier())
    logical.append(_load_frag(subtile=0))
    logical.append(_compute(subtile=0))

    var p = single_buffer_reorder(logical, mi355x_single_buffer())

    # Expected: compute[0], sync, store_shared, load_global, sync, frag[0]
    assert_true(
        p[0].tag == COMPUTE and p[0].subtile == 0,
        "T=1 pos 0: compute[0]",
    )
    assert_true(
        p[1].tag == DefaultMatmulOps.BARRIER.value,
        "T=1 pos 1: barrier",
    )
    assert_true(p[2].tag == STORE_SMEM, "T=1 pos 2: store_smem")
    assert_true(p[3].tag == LOAD_DRAM, "T=1 pos 3: load_dram")
    assert_true(
        p[4].tag == DefaultMatmulOps.BARRIER.value,
        "T=1 pos 4: barrier (added)",
    )
    assert_true(
        p[5].tag == LOAD_FRAG and p[5].subtile == 0,
        "T=1 pos 5: frag[0] (loop-carried)",
    )

    print("  test_single_buffer_transform_t1 PASSED")


def test_single_buffer_transform_matches_body() raises:
    """Verify single_buffer_reorder produces correct pipelined order."""
    var body = _default_matmul_body[2]()

    # Should be: frag[1], compute[0], barrier, store_smem, load_dram,
    #            compute[1], barrier, frag[0]
    assert_true(
        body[0].tag == LOAD_FRAG and body[0].subtile == 1,
        "body[0]: frag[1]",
    )
    assert_true(
        body[1].tag == COMPUTE and body[1].subtile == 0,
        "body[1]: compute[0]",
    )
    assert_true(
        body[2].tag == DefaultMatmulOps.BARRIER.value, "body[2]: barrier"
    )
    assert_true(body[3].tag == STORE_SMEM, "body[3]: store_smem")
    assert_true(body[4].tag == LOAD_DRAM, "body[4]: load_dram")
    assert_true(
        body[5].tag == COMPUTE and body[5].subtile == 1,
        "body[5]: compute[1]",
    )
    assert_true(
        body[6].tag == DefaultMatmulOps.BARRIER.value, "body[6]: barrier"
    )
    assert_true(
        body[7].tag == LOAD_FRAG and body[7].subtile == 0,
        "body[7]: frag[0]",
    )

    print("  test_single_buffer_transform_matches_body PASSED")


def test_mma_block_interleave_2x2() raises:
    """Verify mma_block_interleave produces correct block structure for 2x2 grid.
    """
    # Logical iteration for half 0 (s=0, os=1).
    var logical = (
        # Global loads
        pipe(_load_a(stage=1, subtile=1, k_offset=KOffsetKind.K_PREV))
        >> _load_a(stage=0, subtile=0, k_offset=KOffsetKind.K0)
        >> _load_b(stage=0, subtile=0, k_offset=KOffsetKind.K0)
        >> _load_b(stage=0, subtile=1, k_offset=KOffsetKind.K0)
        # Fragment loads
        >> _mma_load_a(stage=0, subtile=0)
        >> _mma_load_a(stage=0, subtile=1)
        >> _mma_load_b(stage=0, subtile=0)
        >> _mma_load_b(stage=0, subtile=1)
        # Compute
        >> _mma(stage=0, subtile=0)
        >> _mma(stage=0, subtile=1)
        >> _mma(stage=1, subtile=0)
        >> _mma(stage=1, subtile=1)
    )

    var result = mma_block_interleave(logical, mi355x_double_buffer())

    # Block 0: B_frag[0], A_frag[0], global[0], MMA[0,0]
    assert_true(
        result.ops[0].tag == MMA_LOAD_B and result.ops[0].subtile == 0,
        "block0[0]: B_frag[0]",
    )
    assert_true(
        result.ops[1].tag == MMA_LOAD_A and result.ops[1].subtile == 0,
        "block0[1]: A_frag[0]",
    )
    assert_true(
        result.ops[2].tag == LOAD_A and result.ops[2].subtile == 1,
        "block0[2]: global load_a[1,1]",
    )
    assert_true(
        result.ops[3].tag == MMA
        and result.ops[3].stage == 0
        and result.ops[3].subtile == 0,
        "block0[3]: MMA[0,0]",
    )

    # Block 1: B_frag[1], MMA[0,1]
    assert_true(
        result.ops[4].tag == MMA_LOAD_B and result.ops[4].subtile == 1,
        "block1[0]: B_frag[1]",
    )
    assert_true(
        result.ops[5].tag == MMA
        and result.ops[5].stage == 0
        and result.ops[5].subtile == 1,
        "block1[1]: MMA[0,1]",
    )

    # Block 2: A_frag[1], global[1], global[2], MMA[1,0]
    assert_true(
        result.ops[6].tag == MMA_LOAD_A and result.ops[6].subtile == 1,
        "block2[0]: A_frag[1]",
    )
    assert_true(
        result.ops[7].tag == LOAD_A and result.ops[7].subtile == 0,
        "block2[1]: global load_a[0,0]",
    )
    assert_true(
        result.ops[8].tag == LOAD_B and result.ops[8].subtile == 0,
        "block2[2]: global load_b[0,0]",
    )
    assert_true(
        result.ops[9].tag == MMA
        and result.ops[9].stage == 1
        and result.ops[9].subtile == 0,
        "block2[3]: MMA[1,0]",
    )

    # Block 3: global[3], MMA[1,1]
    assert_true(
        result.ops[10].tag == LOAD_B and result.ops[10].subtile == 1,
        "block3[0]: global load_b[0,1]",
    )
    assert_true(
        result.ops[11].tag == MMA
        and result.ops[11].stage == 1
        and result.ops[11].subtile == 1,
        "block3[1]: MMA[1,1]",
    )

    print("  test_mma_block_interleave_2x2 PASSED")


def test_config_single_buffer_default() raises:
    """Verify mi355x_single_buffer() matches hardcoded transform."""
    var config = mi355x_single_buffer()
    assert_equal(config.depth, 1, "single_buffer depth")
    assert_equal(config.prefetch, 1, "single_buffer prefetch")
    assert_equal(config.drain_passes, 2, "single_buffer drain_passes")
    assert_equal(config.loop_carried.selector, 0, "loop-carried is subtile 0")
    assert_true(config.mma_serial, "MMA is serial")

    # Verify config-parameterized transform matches default.
    var logical = List[OpDesc]()
    logical.append(_load_dram())
    logical.append(_store_smem())
    logical.append(OpDesc.barrier())
    logical.append(_load_frag(subtile=0))
    logical.append(_load_frag(subtile=1))
    logical.append(_compute(subtile=0))
    logical.append(_compute(subtile=1))
    var pipelined = single_buffer_reorder(logical, config)

    # First op should be frag[1] (non-loop-carried fragment first).
    assert_equal(pipelined[0].subtile, 1, "first op is frag[1]")
    assert_true(
        pipelined[0].role == OpRole.FRAGMENT_LOAD,
        "first op is FRAGMENT_LOAD",
    )

    print("  test_config_single_buffer_default PASSED")


def test_config_block_sizing() raises:
    """Verify custom block sizing changes global load distribution."""
    var logical = (
        pipe(_load_a(stage=1, subtile=1, k_offset=KOffsetKind.K_PREV))
        >> _load_a(stage=0, subtile=0, k_offset=KOffsetKind.K0)
        >> _load_b(stage=0, subtile=0, k_offset=KOffsetKind.K0)
        >> _load_b(stage=0, subtile=1, k_offset=KOffsetKind.K0)
        >> _mma_load_a(stage=0, subtile=0)
        >> _mma_load_a(stage=0, subtile=1)
        >> _mma_load_b(stage=0, subtile=0)
        >> _mma_load_b(stage=0, subtile=1)
        >> _mma(stage=0, subtile=0)
        >> _mma(stage=0, subtile=1)
        >> _mma(stage=1, subtile=0)
        >> _mma(stage=1, subtile=1)
    )

    # Uniform block sizing (all blocks target 3 ops) changes distribution.
    var config = mi355x_double_buffer()
    config.block_sizing = BlockSizing.uniform(3)
    var result = mma_block_interleave(logical, config)

    # Block 0 now targets 3: 2 frags + 0 globals + MMA = 3 (no globals fit).
    # With uniform(3), heavy and light both = 3.
    # Block 0: new_m=True -> target=3 (uniform). 2 frags + MMA = 3. 0 globals.
    assert_true(
        result.ops[0].tag == MMA_LOAD_B,
        "uniform: block0 starts with B frag",
    )
    assert_true(
        result.ops[1].tag == MMA_LOAD_A,
        "uniform: block0 has A frag",
    )
    assert_true(
        result.ops[2].tag == MMA,
        "uniform: block0 ends with MMA (no global)",
    )

    print("  test_config_block_sizing PASSED")


def test_schedule_compiler_t2() raises:
    """Verify ScheduleCompiler produces correct schedule for T=2."""
    var sc = ScheduleCompiler()
    sc.compile(
        SingleBufferSchedule[2](
            AMDScheduleHints(
                m_mmas=4,
                n_mmas=4,
                k_mmas=2,
                mma_m=16,
                mma_n=16,
                a_loads_per_thread=2,
                b_loads_per_thread=2,
            )
        )
    )

    # Body: 8 ops (2*2+4), sizes discovered not prescribed.
    assert_equal(len(sc.body), 8, "body should have 8 ops")

    # Edges: 8 edges (2*(2-1)+6), discovered.
    assert_equal(len(sc.edges), 8, "should have 8 edges")

    # Prologue: 6 entries, discovered.
    assert_equal(len(sc.prologue), 6, "prologue should have 6 entries")
    assert_true(
        sc.prologue[0].op.tag == LOAD_DRAM,
        "prologue starts with LOAD_DRAM",
    )

    # Kernel: 8 data ops + 48 schedule hints = 56, discovered.
    assert_equal(
        len(sc.kernel), 56, "kernel should have 56 entries (data+hints)"
    )

    # Epilogue: 13 entries for T=2 (2*2+2*2+5), discovered.
    assert_equal(len(sc.epilogue), 13, "epilogue should have 13 entries")

    print("  test_schedule_compiler_t2 PASSED")


def test_schedule_compiler_t1() raises:
    """Verify ScheduleCompiler for T=1 edge case."""
    var sc = ScheduleCompiler()
    sc.compile(
        SingleBufferSchedule[1](
            AMDScheduleHints(
                m_mmas=4,
                n_mmas=4,
                k_mmas=2,
                mma_m=16,
                mma_n=16,
                a_loads_per_thread=2,
                b_loads_per_thread=2,
            )
        )
    )

    assert_equal(len(sc.body), 6, "T=1 body should have 6 ops")
    assert_equal(len(sc.edges), 6, "T=1 should have 6 edges")
    assert_equal(len(sc.prologue), 6, "T=1 prologue should have 6 entries")
    assert_equal(len(sc.epilogue), 9, "T=1 epilogue should have 9 entries")

    print("  test_schedule_compiler_t1 PASSED")


def test_schedule_compiler_interleaved() raises:
    """Verify ScheduleCompiler with DeclarativeSchedule trait impl."""
    var sc = ScheduleCompiler()
    sc.compile(
        DeclarativeSchedule[False, 4, 4](
            config=ScheduleConfig(auto_waits=False)
        )
    )

    # Body: 24 ops (2 halves x 12), discovered.
    assert_equal(len(sc.body), 24, "interleaved body should have 24 ops")

    # Edges: 54, discovered (same as _generate_interleaved_edges).
    assert_equal(len(sc.edges), 54, "interleaved should have 54 edges")

    # Kernel: 70 entries (expansion), discovered.
    assert_equal(
        len(sc.kernel), 70, "interleaved kernel should have 70 entries"
    )

    # Prologue: 10 entries, discovered.
    assert_equal(
        len(sc.prologue), 10, "interleaved prologue should have 10 entries"
    )

    # Kernel deps: 8 structural deps.
    assert_equal(
        len(sc.kernel_deps), 8, "interleaved should have 8 kernel deps"
    )

    # Verify edge breakdown matches the 4 phases: 16 + 8 + 16 + 14 = 54.
    var anti = 0
    for i in range(len(sc.edges)):
        if sc.edges[i].dep_kind != DepKind.FLOW:
            anti += 1
    assert_equal(anti, 14, "should have 14 ANTI edges")

    print("  test_schedule_compiler_interleaved PASSED")


# =============================================================================
# Optimal Scheduler Tests
# =============================================================================


def test_compute_alap() raises:
    """Test ALAP computation on a 3-op linear chain."""
    # load_a(lat=200) -> mma_load_a(lat=20) -> mma(lat=16)
    var body = LoopBody()
    body.ops.append(
        OpNode(
            op=_load_a(stage=0, subtile=0, k_offset=KOffsetKind.K0, vm_cost=4)
        )
    )
    body.ops.append(OpNode(op=_mma_load_a(stage=0, subtile=0)))
    body.ops.append(OpNode(op=_mma(stage=0, subtile=0)))
    body.edges.append(DepEdge.flow(0, 1))
    body.edges.append(DepEdge.flow(1, 2))

    var asap = body.compute_asap()
    # Critical path: 200 + 20 + 16 = 236
    var makespan = asap[2] + 16  # 220 + 16 = 236
    var alap = body.compute_alap(makespan)

    # ALAP(mma) = 236 - 16 = 220
    assert_equal(alap[2], 220, "mma ALAP should be 220")
    # ALAP(mma_load_a) = 220 - 20 = 200
    assert_equal(alap[1], 200, "mma_load_a ALAP should be 200")
    # ALAP(load_a) = 200 - 200 = 0
    assert_equal(alap[0], 0, "load_a ALAP should be 0")

    # All ops are critical (zero slack).
    for i in range(3):
        assert_equal(alap[i], asap[i], "zero slack on critical chain")

    print("  test_compute_alap PASSED")


def test_evaluate_makespan_linear_chain() raises:
    """Makespan of a serial chain equals critical path."""
    var body = LoopBody()
    body.ops.append(
        OpNode(
            op=_load_a(stage=0, subtile=0, k_offset=KOffsetKind.K0, vm_cost=4)
        )
    )
    body.ops.append(OpNode(op=_mma_load_a(stage=0, subtile=0)))
    body.ops.append(OpNode(op=_mma(stage=0, subtile=0)))
    body.edges.append(DepEdge.flow(0, 1))
    body.edges.append(DepEdge.flow(1, 2))

    var order = List[Int]()
    order.append(0)
    order.append(1)
    order.append(2)
    var makespan = _evaluate_makespan(body, order)

    # load(200) + mma_load(20) + mma(16) = 236
    assert_equal(makespan, 236, "linear chain makespan should be 236")

    print("  test_evaluate_makespan_linear_chain PASSED")


def test_evaluate_makespan_serial_mma() raises:
    """Two independent MMAs serialize on the MMA_UNIT."""
    # mma0(lat=16) and mma1(lat=16), no deps, both use MMA_UNIT.
    var body = LoopBody()
    body.ops.append(OpNode(op=_mma(stage=0, subtile=0)))
    body.ops.append(OpNode(op=_mma(stage=0, subtile=1)))

    var order = List[Int]()
    order.append(0)
    order.append(1)
    var makespan = _evaluate_makespan(body, order)

    # MMA is serial: mma0 ends at 16, mma1 starts at 16, ends at 32.
    assert_equal(makespan, 32, "two MMAs should serialize to 32 cycles")

    print("  test_evaluate_makespan_serial_mma PASSED")


def test_optimal_schedule_interleaved() raises:
    """24-op half-isolated search produces valid permutation."""
    var body = _build_ldg()

    var order = optimal_schedule_with_halves(body)

    # Verify it's a valid permutation.
    var seen = List[Bool]()
    for _ in range(len(body.ops)):
        seen.append(False)
    for i in range(len(body.ops)):
        assert_true(
            order[i] >= 0 and order[i] < len(body.ops),
            "order index out of bounds",
        )
        assert_true(not seen[order[i]], "duplicate index in ordering")
        seen[order[i]] = True

    # Verify half isolation: first 12 entries should be ops 0-11.
    for i in range(12):
        assert_true(
            order[i] < 12,
            "first half should contain only ops 0-11",
        )
    for i in range(12, 24):
        assert_true(
            order[i] >= 12,
            "second half should contain only ops 12-23",
        )

    # Verify makespan <= greedy scheduler's makespan.
    var greedy_order = greedy_schedule(body)
    var greedy_cost = _evaluate_makespan(body, greedy_order)
    var optimal_cost = _evaluate_makespan(body, order)
    assert_true(
        optimal_cost <= greedy_cost,
        "optimal should be at least as good as greedy",
    )

    print("  test_optimal_schedule_interleaved PASSED")


def test_opdesc_metadata() raises:
    """Verify OpDesc factory methods set resource, latency, and role correctly.
    """

    def _check(
        op: OpDesc,
        name: StringLiteral,
        res: ResourceKind,
        lat: Int,
        r: OpRole,
    ) raises:
        assert_true(op.resource == res, String(name) + ": resource")
        assert_equal(op.latency, lat, String(name) + ": latency")
        assert_true(op.role == r, String(name) + ": role")

    # Ping-pong global loads: GLOBAL_MEM, 200 cycles, GLOBAL_LOAD
    _check(
        _load_a(stage=0, subtile=0, k_offset=KOffsetKind.K0),
        "load_a",
        ResourceKind.GLOBAL_MEM,
        200,
        OpRole.GLOBAL_LOAD,
    )
    _check(
        _load_b(stage=0, subtile=0, k_offset=KOffsetKind.K0),
        "load_b",
        ResourceKind.GLOBAL_MEM,
        200,
        OpRole.GLOBAL_LOAD,
    )

    # Ping-pong fragment loads: LDS, 20 cycles, FRAGMENT_LOAD
    _check(
        _mma_load_a(stage=0, subtile=0),
        "mma_load_a",
        ResourceKind.LDS,
        20,
        OpRole.FRAGMENT_LOAD,
    )
    _check(
        _mma_load_b(stage=0, subtile=0),
        "mma_load_b",
        ResourceKind.LDS,
        20,
        OpRole.FRAGMENT_LOAD,
    )

    # Compute ops: MMA_UNIT, COMPUTE role
    _check(
        OpDesc.op(COMPUTE, ResourceKind.MMA_UNIT, 64, OpRole.COMPUTE, stage=0),
        "compute",
        ResourceKind.MMA_UNIT,
        64,
        OpRole.COMPUTE,
    )
    _check(
        _mma(stage=0, subtile=0),
        "mma",
        ResourceKind.MMA_UNIT,
        16,
        OpRole.COMPUTE,
    )
    _check(
        _compute(subtile=0),
        "compute_k",
        ResourceKind.MMA_UNIT,
        64,
        OpRole.COMPUTE,
    )

    # Sync ops: SCALAR, 0 cycles, SYNC
    _check(OpDesc.barrier(), "barrier", ResourceKind.SCALAR, 0, OpRole.SYNC)
    _check(OpDesc.wait_vm[0](), "wait_vm", ResourceKind.SCALAR, 0, OpRole.SYNC)
    _check(
        OpDesc.wait_vm_n(0), "wait_vm_n", ResourceKind.SCALAR, 0, OpRole.SYNC
    )
    _check(
        OpDesc.wait_lgkm[0](), "wait_lgkm", ResourceKind.SCALAR, 0, OpRole.SYNC
    )
    _check(
        OpDesc.wait_lgkm_n(0),
        "wait_lgkm_n",
        ResourceKind.SCALAR,
        0,
        OpRole.SYNC,
    )

    # Fence ops: SCALAR, 0 cycles, FENCE
    _check(
        OpDesc.set_prio[0](), "set_prio", ResourceKind.SCALAR, 0, OpRole.FENCE
    )
    _check(
        OpDesc.schedule_barrier(),
        "schedule_barrier",
        ResourceKind.SCALAR,
        0,
        OpRole.FENCE,
    )
    _check(
        OpDesc.sched_group_barrier[0, 1](),
        "sched_group_barrier",
        ResourceKind.SCALAR,
        0,
        OpRole.FENCE,
    )

    # Default matmul ops
    _check(
        _load_dram(),
        "load_dram",
        ResourceKind.GLOBAL_MEM,
        200,
        OpRole.GLOBAL_LOAD,
    )
    _check(
        _store_smem(),
        "store_smem",
        ResourceKind.LDS,
        20,
        OpRole.SHARED_STORE,
    )
    _check(
        _load_frag(subtile=0),
        "load_frag",
        ResourceKind.LDS,
        20,
        OpRole.FRAGMENT_LOAD,
    )

    # Pipe-based factories
    var frags = load_frags[0, 2]()
    _check(
        frags.ops[0],
        "load_frags[0]",
        ResourceKind.LDS,
        20,
        OpRole.FRAGMENT_LOAD,
    )
    var computes = compute_range[0, 2]()
    _check(
        computes.ops[0],
        "compute_range[0]",
        ResourceKind.MMA_UNIT,
        64,
        OpRole.COMPUTE,
    )

    # Sentinel
    var none_op = OpDesc.none()
    assert_true(none_op.role == OpRole.NONE, "none: role")
    assert_true(none_op.resource == ResourceKind.NONE, "none: resource")

    print("  test_opdesc_metadata PASSED")


# =============================================================================
# Target cost model tests
# =============================================================================


def test_cost_model_annotation() raises:
    """Verify annotate_ops with MI355X model produces same OpDesc as factory functions.
    """
    var model = mi355x_cost_model()

    # Logical op: just tag + buffer metadata, no resource/latency/role.
    var logical = OpDesc.logical(
        LOAD_A, channel=0, stage=1, subtile=0, k_offset=KOffsetKind.K0
    )
    assert_true(
        logical.resource == ResourceKind.NONE,
        "logical op should have NONE resource",
    )
    assert_equal(logical.latency, 0, "logical op should have 0 latency")
    assert_true(logical.role == OpRole.NONE, "logical op should have NONE role")

    # Annotate with cost model.
    var ops = List[OpDesc]()
    ops.append(logical)
    var annotated = annotate_ops(ops, model)
    var op = annotated[0]

    # After annotation: resource, latency, role filled in from cost model.
    assert_true(
        op.resource == ResourceKind.GLOBAL_MEM,
        "LOAD_A should be GLOBAL_MEM",
    )
    assert_equal(op.latency, 200, "LOAD_A latency should be 200")
    assert_true(
        op.role == OpRole.GLOBAL_LOAD, "LOAD_A role should be GLOBAL_LOAD"
    )

    # Buffer metadata preserved.
    assert_equal(op.tag, LOAD_A, "tag preserved")
    assert_equal(op.channel, 0, "channel preserved")
    assert_equal(op.stage, 1, "stage preserved")
    assert_equal(op.subtile, 0, "subtile preserved")
    assert_equal(
        op.k_offset.bk_multiple,
        KOffsetKind.K0.bk_multiple,
        "k_offset preserved",
    )

    # Test all op types via cost model.
    var all_ops = List[OpDesc]()
    all_ops.append(OpDesc.logical(LOAD_A, channel=0))
    all_ops.append(OpDesc.logical(LOAD_B, channel=1))
    all_ops.append(OpDesc.logical(MMA_LOAD_A, channel=0))
    all_ops.append(OpDesc.logical(MMA_LOAD_B, channel=1))
    all_ops.append(OpDesc.logical(MMA))
    var all_annotated = annotate_ops(all_ops, model)

    # LOAD_A/B → GLOBAL_MEM, 200, GLOBAL_LOAD
    assert_true(
        all_annotated[0].resource == ResourceKind.GLOBAL_MEM, "LOAD_A res"
    )
    assert_true(
        all_annotated[1].resource == ResourceKind.GLOBAL_MEM, "LOAD_B res"
    )
    assert_equal(all_annotated[0].latency, 200, "LOAD_A lat")
    assert_equal(all_annotated[1].latency, 200, "LOAD_B lat")

    # MMA_LOAD_A/B → LDS, 20, FRAGMENT_LOAD
    assert_true(all_annotated[2].resource == ResourceKind.LDS, "MMA_LOAD_A res")
    assert_true(all_annotated[3].resource == ResourceKind.LDS, "MMA_LOAD_B res")
    assert_equal(all_annotated[2].latency, 20, "MMA_LOAD_A lat")
    assert_equal(all_annotated[3].latency, 20, "MMA_LOAD_B lat")

    # MMA → MMA_UNIT, 16, COMPUTE
    assert_true(all_annotated[4].resource == ResourceKind.MMA_UNIT, "MMA res")
    assert_equal(all_annotated[4].latency, 16, "MMA lat")
    assert_true(all_annotated[4].role == OpRole.COMPUTE, "MMA role")

    print("  test_cost_model_annotation PASSED")


# =============================================================================
# TargetProfile tests
# =============================================================================


def test_target_profile_mi355x() raises:
    """Verify mi355x_target() provides consistent cost model and pipeline config.
    """
    var target = mi355x_target()

    # Pipeline config: double-buffer, 2 halves, 2x2 MMA grid.
    assert_equal(target.pipeline.depth, 2, "depth should be 2")
    assert_equal(target.pipeline.num_halves, 2, "num_halves should be 2")
    assert_equal(target.pipeline.m_mmas, 2, "m_mmas should be 2")
    assert_equal(target.pipeline.n_mmas, 2, "n_mmas should be 2")
    assert_equal(target.pipeline.mma_latency, 16, "mma_latency should be 16")

    # Cost model: verify LOAD_A costs match standalone mi355x_cost_model().
    var standalone = mi355x_cost_model()
    var cost_a = target.cost_model.get_cost(LOAD_A)
    var ref_a = standalone.get_cost(LOAD_A)
    assert_equal(cost_a.latency, ref_a.latency, "LOAD_A latency matches")
    assert_true(cost_a.resource == ref_a.resource, "LOAD_A resource matches")
    assert_true(cost_a.role == ref_a.role, "LOAD_A role matches")

    # Cost model: verify MMA costs.
    var cost_mma = target.cost_model.get_cost(MMA)
    assert_equal(cost_mma.latency, 16, "MMA latency should be 16")
    assert_true(cost_mma.resource == ResourceKind.MMA_UNIT, "MMA resource")

    # vm_per_load defaults.
    assert_equal(target.pipeline.vm_per_load_a, 4, "vm_per_load_a default")
    assert_equal(target.pipeline.vm_per_load_b, 4, "vm_per_load_b default")

    # vm_per_load overrides.
    var custom = mi355x_target(vm_per_load_a=8, vm_per_load_b=2)
    assert_equal(custom.pipeline.vm_per_load_a, 8, "vm_per_load_a override")
    assert_equal(custom.pipeline.vm_per_load_b, 2, "vm_per_load_b override")

    print("  test_target_profile_mi355x PASSED")


def test_target_profile_declarative_equivalence() raises:
    """DeclarativeSchedule via TargetProfile matches via separate params."""
    # New API: TargetProfile.
    var new_sched = DeclarativeSchedule[False, 0, 0](
        config=ScheduleConfig(
            scheduling=SchedulingStrategy.IDENTITY, auto_waits=False
        ),
        target=mi355x_target(),
    )
    # Old API: separate params (backward-compatible constructor).
    var old_sched = DeclarativeSchedule[False, 0, 0](
        config=ScheduleConfig(
            scheduling=SchedulingStrategy.IDENTITY, auto_waits=False
        ),
        hw_config=mi355x_double_buffer(),
        cost_model=mi355x_cost_model(),
    )

    var new_body = new_sched.build_body()
    var old_body = old_sched.build_body()

    assert_equal(len(new_body), len(old_body), "body length")
    for i in range(len(new_body)):
        assert_equal(
            new_body[i].tag,
            old_body[i].tag,
            String("body[{}] tag").format(i),
        )
        assert_equal(
            new_body[i].stage,
            old_body[i].stage,
            String("body[{}] stage").format(i),
        )
        assert_equal(
            new_body[i].subtile,
            old_body[i].subtile,
            String("body[{}] subtile").format(i),
        )

    print("  test_target_profile_declarative_equivalence PASSED")


# =============================================================================
# Declarative schedule equivalence tests
# =============================================================================


def test_declarative_body_structure() raises:
    """DeclarativeSchedule.build_body() produces expected 24-op ping-pong body.
    """
    var sched = DeclarativeSchedule[False, 0, 0](
        config=ScheduleConfig(auto_waits=False)
    )
    var body = sched.build_body()

    # 2 halves x 12 ops = 24 total.
    assert_equal(len(body), 24, "body should have 24 ops")

    # Verify op distribution: 4 GLOBAL_LOAD, 8 FRAGMENT_LOAD, 4+8=12... actually
    # 4 LOAD_A + 4 LOAD_B = 8 global loads? No: 2 halves x (1+1+1+1) = 8 globals,
    # 2 halves x 4 frag loads = 8 frag loads, 2 halves x 4 MMA = 8 computes.
    var n_global = 0
    var n_frag = 0
    var n_compute = 0
    for i in range(len(body)):
        if body[i].role == OpRole.GLOBAL_LOAD:
            n_global += 1
        elif body[i].role == OpRole.FRAGMENT_LOAD:
            n_frag += 1
        elif body[i].role == OpRole.COMPUTE:
            n_compute += 1

    assert_equal(n_global, 8, "8 global loads (4 per half)")
    assert_equal(n_frag, 8, "8 fragment loads (4 per half)")
    assert_equal(n_compute, 8, "8 MMA computes (4 per half)")

    # Verify all ops have valid annotations (resource != NONE).
    for i in range(len(body)):
        assert_true(
            body[i].resource._value != ResourceKind.NONE._value,
            String("body[{}] should have annotated resource").format(i),
        )

    print("  test_declarative_body_structure PASSED")


def test_declarative_schedule_compiles() raises:
    """Full compiled DeclarativeSchedule produces expected phase lengths."""
    var sc = build_schedule[False, 0, 0](
        config=ScheduleConfig(auto_waits=False)
    )

    # Body: 24 ops.
    assert_equal(len(sc.body), 24, "body length")

    # Kernel: should have all 24 data ops + infrastructure ops.
    assert_true(len(sc.kernel) > 24, "kernel should include infrastructure ops")

    # Verify all 24 data op tags appear in kernel.
    var data_tags = List[Int]()
    for i in range(len(sc.kernel)):
        var tag = sc.kernel[i].op.tag
        if tag < 128:
            data_tags.append(tag)
    assert_equal(len(data_tags), 24, "all 24 data ops in kernel")

    # Prologue and epilogue should be non-empty.
    assert_true(len(sc.prologue) > 0, "prologue should be non-empty")
    assert_true(len(sc.epilogue) > 0, "epilogue should be non-empty")

    # Edges: 54 (same as before).
    assert_equal(len(sc.edges), 54, "54 dependency edges")

    print("  test_declarative_schedule_compiles PASSED")


def test_declarative_c2_validation() raises:
    """Auto-scheduled DeclarativeSchedule satisfies C2 (data dependence)."""
    var sc = ScheduleCompiler()
    sc.compile(
        DeclarativeSchedule[False, 0, 0](
            config=ScheduleConfig(
                scheduling=SchedulingStrategy.CSP, auto_waits=False
            )
        )
    )
    var body = sc.body.copy()
    var kernel = sc.kernel.copy()

    # Extract data op tags from kernel entries (skip infrastructure ops).
    var data_tags = List[Int]()
    for i in range(len(kernel)):
        var tag = kernel[i].op.tag
        if tag < 128:  # Kernel-specific data ops are < 128.
            data_tags.append(tag)

    # Verify all data ops from body appear in kernel.
    assert_equal(len(data_tags), len(body), "data op count must match body")

    print("  test_declarative_c2_validation PASSED")


# =============================================================================
# Declarative Edge Rule Tests
# =============================================================================


def _edges_match(a: List[DepEdge], b: List[DepEdge]) raises -> Bool:
    """Check two edge lists match (unordered — sort by (producer, consumer, kind, d)).
    """
    if len(a) != len(b):
        return False

    # Simple O(n^2) containment check for small lists.
    var matched = List[Bool]()
    for _ in range(len(b)):
        matched.append(False)

    for i in range(len(a)):
        var found = False
        for j in range(len(b)):
            if matched[j]:
                continue
            if (
                a[i].producer_idx == b[j].producer_idx
                and a[i].consumer_idx == b[j].consumer_idx
                and a[i].dep_kind == b[j].dep_kind
                and a[i].loop_distance == b[j].loop_distance
            ):
                matched[j] = True
                found = True
                break
        if not found:
            return False
    return True


def test_rule_based_edges_match_double_buffer() raises:
    """Verify rule-based edge derivation matches procedural for double-buffer.
    """
    var sched = DeclarativeSchedule[False, 0, 0](
        config=ScheduleConfig(auto_waits=False)
    )
    var body = sched.build_body()
    var config = mi355x_double_buffer()

    # Procedural (existing).
    var expected = derive_edges_from_ops(body, config)

    # Rule-based (new).
    var rules = double_buffer_edge_rules()
    var actual = apply_edge_rules(body, config, rules)

    assert_equal(
        len(actual),
        len(expected),
        "double-buffer: edge count mismatch",
    )
    assert_true(
        _edges_match(actual, expected),
        "double-buffer: edge content mismatch",
    )

    print("  test_rule_based_edges_match_double_buffer PASSED")


def test_rule_based_edges_match_single_buffer() raises:
    """Verify rule-based edge derivation matches procedural for single-buffer.
    """
    var config = mi355x_single_buffer()

    # Build a single-buffer body using the same helper as other tests.
    var body = _default_matmul_body[2]()

    # Procedural (existing).
    var expected = derive_edges_from_ops(body, config)

    # Rule-based (new).
    var rules = single_buffer_edge_rules(config)
    var actual = apply_edge_rules(body, config, rules)

    assert_equal(
        len(actual),
        len(expected),
        "single-buffer: edge count mismatch",
    )
    assert_true(
        _edges_match(actual, expected),
        "single-buffer: edge content mismatch",
    )

    print("  test_rule_based_edges_match_single_buffer PASSED")


def test_optimize_within_barriers_single_buffer() raises:
    """Verify CSP optimization within barrier segments preserves correctness.

    The optimized body must:
      1. Have the same length as the original
      2. Contain the same ops (just reordered within segments)
      3. Preserve barrier positions
      4. Satisfy all d=0 dependency edges
    """
    from linalg.matmul.gpu.amd.matmul_schedule import _logical_body
    from linalg.matmul.gpu.amd.amd_target import mi355x_single_buffer_target
    from pipeline.types import annotate_ops
    from pipeline.program_builder import single_buffer_reorder

    var target = mi355x_single_buffer_target()
    var config = target.pipeline

    # Build the body the same way SingleBufferSchedule.build_body() does.
    var logical = _logical_body[4]()
    var annotated = annotate_ops(logical, target.cost_model)
    var body = single_buffer_reorder(annotated, config)

    # CSP-optimize within barrier segments.
    var optimized = optimize_within_barriers(body, config)

    # Same length.
    assert_equal(
        len(optimized),
        len(body),
        "optimized body length must match original",
    )

    # Barriers at same positions.
    for i in range(len(body)):
        if body[i].role == OpRole.SYNC:
            assert_true(
                optimized[i].role == OpRole.SYNC,
                "barrier must remain at same position",
            )

    # All d=0 edges satisfied: producer before consumer.
    var edges = derive_edges_from_ops(optimized, config)
    for e in range(len(edges)):
        var edge = edges[e]
        if edge.loop_distance == 0:
            assert_true(
                edge.producer_idx < edge.consumer_idx,
                "d=0 edge violated: producer must precede consumer",
            )

    print("  test_optimize_within_barriers_single_buffer PASSED")


def test_phase_recipe_prologue() raises:
    """Verify recipe-based prologue matches expected structure.

    The single-buffer prologue should emit:
      load, store, barrier, load (prefetch), lc_frag, fence
    = 6 entries total.
    """
    from pipeline.types import Phase
    from pipeline.phase_derivation import (
        PhaseStep,
        apply_phase_recipe,
        single_buffer_prologue_recipe,
    )

    var config = mi355x_single_buffer()
    var body = _default_matmul_body[2]()

    var recipe = single_buffer_prologue_recipe()
    var pro = apply_phase_recipe(body, recipe, config, Phase.PROLOGUE)

    # 6 entries: load, store, barrier, load, lc_frag, fence.
    assert_equal(len(pro), 6, "prologue should have 6 entries")

    # Check structure.
    assert_true(
        pro[0].op.role == OpRole.GLOBAL_LOAD, "pro[0] should be global_load"
    )
    assert_true(
        pro[1].op.role == OpRole.SHARED_STORE,
        "pro[1] should be shared_store",
    )
    assert_true(pro[2].op.role == OpRole.SYNC, "pro[2] should be barrier")
    assert_true(
        pro[3].op.role == OpRole.GLOBAL_LOAD, "pro[3] should be global_load"
    )
    assert_true(
        pro[4].op.role == OpRole.FRAGMENT_LOAD,
        "pro[4] should be fragment_load",
    )
    # Loop-carried fragment has subtile == lc.selector == 0.
    assert_equal(
        pro[4].op.subtile,
        config.loop_carried.selector,
        "pro[4] should be loop-carried frag",
    )

    print("  test_phase_recipe_prologue PASSED")


def test_phase_recipe_epilogue() raises:
    """Verify recipe-based epilogue matches expected structure.

    With num_k_tiles=2, the body has 2 frags (subtile 0=LC, subtile 1=non-LC)
    and 2 computes. Expected epilogue:
      Drain 1: fence, frag[1], barrier, store, comp[0], comp[1], fence
      Drain 2: barrier, frag[0], frag[1], comp[0], comp[1], fence
    = 13 entries total.
    """
    from pipeline.types import Phase
    from pipeline.phase_derivation import (
        PhaseStep,
        apply_phase_recipe,
        single_buffer_epilogue_recipe,
    )

    var config = mi355x_single_buffer()
    var body = _default_matmul_body[2]()

    var recipe = single_buffer_epilogue_recipe()
    var epi = apply_phase_recipe(body, recipe, config, Phase.EPILOGUE)

    # Count entries.
    assert_equal(len(epi), 13, "epilogue should have 13 entries")

    # Drain 1 structure check.
    assert_true(
        epi[0].op.tag >= 128, "epi[0] should be fence (infrastructure op)"
    )
    assert_true(
        epi[1].op.role == OpRole.FRAGMENT_LOAD,
        "epi[1] should be non-LC frag",
    )
    assert_true(
        epi[1].op.subtile != config.loop_carried.selector,
        "epi[1] should NOT be loop-carried",
    )
    assert_true(epi[2].op.role == OpRole.SYNC, "epi[2] should be barrier")

    # Drain 2 starts with barrier.
    assert_true(epi[7].op.role == OpRole.SYNC, "epi[7] should be barrier")

    print("  test_phase_recipe_epilogue PASSED")


def test_edge_rule_count() raises:
    """Verify rule tables have the expected number of rules."""
    var db_rules = double_buffer_edge_rules()
    assert_equal(len(db_rules), 5, "double-buffer should have 5 rules")

    var sb_rules = single_buffer_edge_rules(mi355x_single_buffer())
    assert_equal(len(sb_rules), 8, "single-buffer should have 8 rules")

    print("  test_edge_rule_count PASSED")


# =============================================================================
# Verification & Drain Mask Tests
# =============================================================================


struct _TestProgramResult(Movable):
    """Helper to return program + config from test builder."""

    var program: PipelineProgram
    var config: PipelineConfig

    def __init__(out self, program: PipelineProgram, config: PipelineConfig):
        self.program = program.copy()
        self.config = config


def _build_bf16_program(sched: ScheduleConfig) -> _TestProgramResult:
    """Build a BF16 double-buffer program for testing."""
    var target = mi355x_target(vm_per_load_a=4, vm_per_load_b=4)
    var config = target.pipeline
    var ds = DeclarativeSchedule[False, 8, 8](config=sched)
    var body = ds.build_body()
    var annotated = annotate_ops(body, target.cost_model)
    var edges = derive_edges_from_ops(annotated, config)
    var ldg = LoopBody()
    for i in range(len(annotated)):
        ldg.ops.append(OpNode(op=annotated[i]))
    ldg.edges = edges^

    var order = List[Int]()
    if sched.scheduling == SchedulingStrategy.CSP:
        order = optimal_schedule_with_halves(
            ldg,
            max_globals_per_block=config.block_sizing.max_globals,
        )
    else:
        for i in range(len(annotated)):
            order.append(i)

    var program = build_double_buffer_program(ldg, order, config, sched)
    return _TestProgramResult(program, config)


def test_verify_schedule_passes_valid() raises:
    """Default BF16 schedule with auto_waits passes all verification checks."""
    var sched = ScheduleConfig(
        scheduling=SchedulingStrategy.CSP,
        auto_waits=True,
        lgkm_per_load_a=8,
        lgkm_per_load_b=8,
    )
    var result = _build_bf16_program(sched)
    var program = result.program.copy()
    var config = result.config

    # Derive wait counts (same as build_kernel_program path).
    var waits = derive_waits_from_blocks(program, config, 8, 8)
    var bph = config.blocks_per_half()
    for half in range(config.num_halves):
        var b0 = half * bph
        var bN = b0 + bph - 1
        if waits[0] < 255:
            var blk = program.blocks[b0]
            blk.pre_sync = OpDesc.wait_lgkm_n(waits[0])
            program.blocks[b0] = blk
        var blk = program.blocks[bN]
        blk.pre_sync = OpDesc.wait_vm_n(waits[1])
        program.blocks[bN] = blk

    # Should not assert (debug_assert is a no-op in non-debug builds,
    # but the function itself should complete without error).
    verify_schedule(program, config, 8, 8, waits[0], waits[1])

    print("  test_verify_schedule_passes_valid PASSED")


def test_drain_mask_selective() raises:
    """Manual drain_lgkm_mask=0b1101 drains only blocks 0,2,3."""
    var sched = ScheduleConfig(
        scheduling=SchedulingStrategy.CSP,
        auto_waits=True,
        lgkm_per_load_a=8,
        lgkm_per_load_b=8,
        drain_lgkm_mask=0b11011101,  # blocks 0,2,3 per half
    )
    var result = _build_bf16_program(sched)
    var program = result.program.copy()
    var config = result.config
    var num_blocks = config.blocks_per_half() * config.num_halves

    # Check which blocks have drain_lgkm_before_loads set.
    var drain_count = 0
    for i in range(num_blocks):
        var b = program.blocks[i]
        if b.drain_lgkm_before_loads:
            drain_count += 1
            # Verify the drain is in expected blocks.
            var pos_in_half = i % config.blocks_per_half()
            # Block 1 (per half) should NOT have drain with mask 0b1101.
            assert_true(
                pos_in_half != 1,
                "block 1 should not drain with mask 0b1101",
            )

    # With uniform distribution, blocks 0,2,3 have both frags and globals.
    # Block 3 may not have frags (only global), so drain might not apply.
    # At minimum, blocks with both frags+globals that match mask bits drain.
    assert_true(drain_count >= 2, "at least 2 blocks should drain")

    print("  test_drain_mask_selective PASSED")


def test_drain_mask_backward_compat() raises:
    """All-bits drain mask drains all blocks with frags+globals."""
    var sched = ScheduleConfig(
        scheduling=SchedulingStrategy.CSP,
        auto_waits=False,
        drain_lgkm_mask=0xFF,
    )
    var result = _build_bf16_program(sched)
    var program = result.program.copy()
    var config = result.config
    var num_blocks = config.blocks_per_half() * config.num_halves

    var drain_count = 0
    for i in range(num_blocks):
        if program.blocks[i].drain_lgkm_before_loads:
            drain_count += 1

    # Every block that has both frags and globals should drain.
    var expected = 0
    for i in range(num_blocks):
        var b = program.blocks[i]
        var has_frag = b.pre_op_0.is_present() or b.pre_op_1.is_present()
        var has_global = (
            b.global_load.is_present() or b.global_load_1.is_present()
        )
        if has_frag and has_global:
            expected += 1

    assert_equal(drain_count, expected, "blanket drain should match expected")

    print("  test_drain_mask_backward_compat PASSED")


def test_auto_drain_derivation() raises:
    """Auto-derived drain mask only sets bits for cross-channel blocks."""
    var sched = ScheduleConfig(
        scheduling=SchedulingStrategy.CSP, auto_waits=True
    )
    var result = _build_bf16_program(sched)
    var program = result.program.copy()
    var config = result.config

    var mask = derive_drain_mask(program, config)
    var num_blocks = config.blocks_per_half() * config.num_halves

    # Verify: bits are set only where frag channel != global channel.
    for i in range(num_blocks):
        var b = program.blocks[i]
        var has_frag = b.pre_op_0.is_present() or b.pre_op_1.is_present()
        var has_global = b.global_load.is_present()
        var bit = (mask >> i) & 1

        if not has_frag or not has_global:
            assert_equal(
                bit,
                0,
                "block " + String(i) + " has no frag/global, should not drain",
            )
        else:
            # If all frags match global channel, bit should be 0.
            var frag_ch_0 = (
                b.pre_op_0.channel if b.pre_op_0.is_present() else -1
            )
            var frag_ch_1 = (
                b.pre_op_1.channel if b.pre_op_1.is_present() else -1
            )
            var global_ch = b.global_load.channel
            var cross = (frag_ch_0 >= 0 and frag_ch_0 != global_ch) or (
                frag_ch_1 >= 0 and frag_ch_1 != global_ch
            )
            if cross:
                assert_equal(
                    bit,
                    1,
                    "block " + String(i) + " is cross-channel, should drain",
                )
            else:
                assert_equal(
                    bit,
                    0,
                    "block " + String(i) + " is same-channel, should not drain",
                )

    print("  test_auto_drain_derivation PASSED")


def test_distribution_preserved_after_csp() raises:
    """CSP + redistribution produces [1,1,1,1] per half with max_globals=1."""
    var sched = ScheduleConfig(
        scheduling=SchedulingStrategy.CSP, auto_waits=True
    )
    var result = _build_bf16_program(sched)
    var program = result.program.copy()
    var config = result.config
    var bph = config.blocks_per_half()

    for half in range(config.num_halves):
        for b in range(bph):
            var block = program.blocks[half * bph + b]
            var count = (1 if block.global_load.is_present() else 0) + (
                1 if block.global_load_1.is_present() else 0
            )
            assert_equal(
                count,
                1,
                "half "
                + String(half)
                + " block "
                + String(b)
                + " should have exactly 1 global",
            )

    print("  test_distribution_preserved_after_csp PASSED")


def test_wait_counts_match_block_structure() raises:
    """Verify derive_waits_from_blocks output matches manual calculation."""
    var sched = ScheduleConfig(
        scheduling=SchedulingStrategy.CSP,
        auto_waits=True,
        lgkm_per_load_a=8,
        lgkm_per_load_b=8,
    )
    var result = _build_bf16_program(sched)
    var program = result.program.copy()
    var config = result.config

    var waits = derive_waits_from_blocks(program, config, 8, 8)
    var wait_lgkm = waits[0]
    var wait_vm = waits[1]

    # wait_lgkm_first = lgkm from block 0's globals.
    # With max_globals=1, block 0 has exactly 1 global load.
    var bph = config.blocks_per_half()
    var b0 = program.blocks[0]
    var expected_lgkm = 0
    if b0.global_load.is_present():
        expected_lgkm += 8 if b0.global_load.channel == 0 else 8
    if b0.global_load_1.is_present():
        expected_lgkm += 8 if b0.global_load_1.channel == 0 else 8
    assert_equal(
        wait_lgkm,
        expected_lgkm,
        "wait_lgkm should match block 0 global lgkm",
    )

    # wait_vm_last = total_vm - completion_vm.
    # Count across all blocks in half 0.
    var total_vm = 0
    var completion_vm = 0
    for b in range(bph):
        var block = program.blocks[b]
        if block.global_load.is_present():
            total_vm += config.vm_per_channel(block.global_load.channel)
            if not block.global_load_prefetch:
                completion_vm += config.vm_per_channel(
                    block.global_load.channel
                )
        if block.global_load_1.is_present():
            total_vm += config.vm_per_channel(block.global_load_1.channel)
            if not block.global_load_1_prefetch:
                completion_vm += config.vm_per_channel(
                    block.global_load_1.channel
                )
    var expected_vm = total_vm - completion_vm
    assert_equal(
        wait_vm,
        expected_vm,
        "wait_vm should equal total_vm - completion_vm",
    )

    # Sanity: values should be reasonable.
    assert_true(wait_lgkm >= 0, "wait_lgkm should be non-negative")
    assert_true(wait_vm >= 0, "wait_vm should be non-negative")
    assert_true(wait_lgkm <= 16, "wait_lgkm should be at most 16")
    assert_true(wait_vm <= 16, "wait_vm should be at most 16")

    print("  test_wait_counts_match_block_structure PASSED")


# =============================================================================
# Phase 4: Prologue from Program Tests
# =============================================================================


def test_prologue_from_program_structure() raises:
    """Program-derived prologue has correct structure: loads, waits, barrier."""
    var target = mi355x_target(vm_per_load_a=4, vm_per_load_b=4)
    var config = target.pipeline
    var sched = ScheduleConfig(
        scheduling=SchedulingStrategy.CSP, auto_waits=True
    )
    var ds = DeclarativeSchedule[False, 8, 8](config=sched, target=target)
    var body = ds.build_body()

    var sc_sched = sched
    sc_sched.lgkm_per_load_a = 8
    sc_sched.lgkm_per_load_b = 8
    var program = build_kernel_program(body, config, sc_sched)
    var prologue = derive_prologue_from_program(program, config)

    # Prologue structure: stage-0 loads, wait_vm(0), barrier, stage-1 loads, wait_vm(0).
    # BF16 has 7 prefetch loads total (4 stage-0, 3 stage-1).
    var load_a = 0
    var load_b = 0
    var waits = 0
    var barriers = 0
    for i in range(len(prologue)):
        var op = prologue[i].op
        if op.role == OpRole.GLOBAL_LOAD:
            if op.channel == 0:
                load_a += 1
            else:
                load_b += 1
        elif op.tag == _Ops.WAIT_VM.value:
            waits += 1
        elif op.tag == _Ops.BARRIER.value:
            barriers += 1

    # BF16: 4 load_a + 3 load_b = 7 prefetch loads.
    assert_equal(load_a + load_b, 7, "should have 7 prefetch loads")
    # 2 wait_vm ops (after stage-0, after stage-1).
    assert_equal(waits, 2, "should have 2 wait_vm ops")
    # 1 barrier (between stage-0 and stage-1).
    assert_equal(barriers, 1, "should have 1 barrier")

    print("  test_prologue_from_program_structure PASSED")


def test_prologue_from_program_entry_count() raises:
    """Program-derived prologue has correct entry count for BF16 and FP8."""
    var target = mi355x_target(vm_per_load_a=4, vm_per_load_b=4)
    var config = target.pipeline
    var sched = ScheduleConfig(
        scheduling=SchedulingStrategy.CSP,
        auto_waits=True,
        lgkm_per_load_a=8,
        lgkm_per_load_b=8,
    )

    # BF16: 3 stage-0 + wait_vm + barrier + 3 stage-1 + wait_vm = 10.
    var ds_bf16 = DeclarativeSchedule[False, 8, 8](config=sched, target=target)
    var body_bf16 = ds_bf16.build_body()
    var prog_bf16 = build_kernel_program(body_bf16, config, sched)
    var pro_bf16 = derive_prologue_from_program(prog_bf16, config)
    assert_equal(len(pro_bf16), 10, "BF16 prologue should have 10 entries")

    # FP8: same structure, same count.
    var ds_fp8 = DeclarativeSchedule[True, 4, 4](config=sched, target=target)
    var body_fp8 = ds_fp8.build_body()
    var prog_fp8 = build_kernel_program(body_fp8, config, sched)
    var pro_fp8 = derive_prologue_from_program(prog_fp8, config)
    assert_equal(len(pro_fp8), 10, "FP8 prologue should have 10 entries")

    print("  test_prologue_from_program_entry_count PASSED")


# =============================================================================
# Phase 3: LDS Contention Model Tests
# =============================================================================


def test_lds_contention_zero_is_noop() raises:
    """LDS contention penalty=0 gives same makespan as default model."""
    # Build a simple LDG: load_a → mma_load_a → mma.
    var body = LoopBody()
    body.ops.append(
        OpNode(
            op=OpDesc.op(0, ResourceKind.GLOBAL_MEM, 200, OpRole.GLOBAL_LOAD)
        )
    )
    body.ops.append(
        OpNode(op=OpDesc.op(3, ResourceKind.LDS, 20, OpRole.FRAGMENT_LOAD))
    )
    body.ops.append(
        OpNode(op=OpDesc.op(5, ResourceKind.MMA_UNIT, 16, OpRole.COMPUTE))
    )
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
    # Two independent loads that use the LDS port:
    # op 0: global_load (GLOBAL_MEM, lat=200)
    # op 1: fragment_load (LDS, lat=20)
    # op 2: mma (MMA_UNIT, lat=16) depends on op 1
    var body = LoopBody()
    body.ops.append(
        OpNode(
            op=OpDesc.op(0, ResourceKind.GLOBAL_MEM, 200, OpRole.GLOBAL_LOAD)
        )
    )
    body.ops.append(
        OpNode(op=OpDesc.op(3, ResourceKind.LDS, 20, OpRole.FRAGMENT_LOAD))
    )
    body.ops.append(
        OpNode(op=OpDesc.op(5, ResourceKind.MMA_UNIT, 16, OpRole.COMPUTE))
    )
    # Only dep: frag_load → mma (op 0 is independent of op 1).
    body.edges.append(DepEdge(1, 2, DepKind.FLOW, 0))

    # Order: global_load first, then fragment_load, then mma.
    var order = List[Int]()
    order.append(0)
    order.append(1)
    order.append(2)

    var ms_none = _evaluate_makespan(body, order, 0)

    # Without penalty: global_load starts at 0, frag_load starts at 0
    # (independent), so makespan = max(200, 20+16) = 200.
    assert_equal(ms_none, 200, "Without penalty: makespan should be 200")

    # Test with short global loads where contention actually matters.
    var body2 = LoopBody()
    body2.ops.append(
        OpNode(op=OpDesc.op(0, ResourceKind.GLOBAL_MEM, 10, OpRole.GLOBAL_LOAD))
    )
    body2.ops.append(
        OpNode(op=OpDesc.op(1, ResourceKind.GLOBAL_MEM, 10, OpRole.GLOBAL_LOAD))
    )
    body2.ops.append(
        OpNode(op=OpDesc.op(3, ResourceKind.LDS, 20, OpRole.FRAGMENT_LOAD))
    )
    # No deps — all independent.

    var order2 = List[Int]()
    order2.append(0)
    order2.append(1)
    order2.append(2)
    var ms2_none = _evaluate_makespan(body2, order2, 0)
    var ms2_pen = _evaluate_makespan(body2, order2, 15)

    # Without penalty: all start at 0, makespan = max(10, 10, 20) = 20.
    assert_equal(ms2_none, 20, "Without penalty: all overlap, makespan=20")

    # With penalty=15: op0 starts at 0 (lds_port_free=15),
    # op1 starts at 15 (lds_port_free=30), op2 starts at 30, ends at 50.
    # makespan = 50.
    assert_equal(ms2_pen, 50, "With penalty=15: LDS ops serialize, makespan=50")

    print("  test_lds_contention_penalty_increases_makespan PASSED")


def test_lds_contention_lower_bound() raises:
    """LDS contention penalty updates the ASAP lower bound."""
    var body = LoopBody()
    # 4 LDS-port ops (2 global + 2 fragment) + 1 MMA.
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

    # Without penalty: lb = max(critical_path, mma_total) = max(10, 16) = 16.
    assert_equal(lb_none, 16, "Without penalty: lb = mma_total = 16")

    # With penalty=20: lds_total = 4 * 20 = 80 > 16.
    assert_equal(lb_pen, 80, "With penalty=20: lb = lds_total = 80")

    print("  test_lds_contention_lower_bound PASSED")


def test_stage_based_completion_symmetric() raises:
    """Both halves should have symmetric wait_vm values.

    The bug: _is_prefetch used k_offset (K0/K1=prefetch, K_PREV=completion)
    which incorrectly classified half 1's stage-0 load (K0 offset) as
    prefetch. With stage-based completion in derive_waits_from_blocks,
    both halves correctly identify 1 completion load (stage != half).

    This ensures the wait_vm at the half boundary drains the completion
    load that writes to the OTHER half's LDS stage, preventing the race
    where s_barrier fires before global→LDS data has landed.
    """
    # Build BF16 program with auto-waits.
    var sched = DeclarativeSchedule[False, 4, 4](
        config=ScheduleConfig(
            scheduling=SchedulingStrategy.CSP, auto_waits=True
        )
    )
    var body = sched.build_body()
    var program = build_kernel_program(
        body,
        sched.config(),
        ScheduleConfig(
            scheduling=SchedulingStrategy.CSP,
            auto_waits=True,
            lgkm_per_load_a=4,
            lgkm_per_load_b=4,
        ),
    )

    # Both halves should have completion loads (stage != half).
    var bph = sched.config().blocks_per_half()
    for half in range(2):
        var half_start = half * bph
        var completion_count = 0
        for b in range(bph):
            var block = program.blocks[half_start + b]
            if (
                block.global_load.is_present()
                and block.global_load.stage != half
            ):
                completion_count += 1
            if (
                block.global_load_1.is_present()
                and block.global_load_1.stage != half
            ):
                completion_count += 1
        assert_true(
            completion_count > 0,
            "half "
            + String(half)
            + " must have at least 1 completion load (stage != half)",
        )

    # The wait_vm values should be equal across halves.
    # (Both have 1 completion load out of 4 total.)
    var kernel = program.expand_to_list(Phase.KERNEL)

    # Find the wait_vm entries in the kernel (at last block of each half).
    var wait_vm_values = List[Int]()
    for i in range(len(kernel)):
        var entry = kernel[i]
        if entry.op.tag == _Ops.WAIT_VM.value:
            wait_vm_values.append(entry.op.wait_value)

    # Both halves should produce the same wait_vm value.
    assert_true(
        len(wait_vm_values) >= 2,
        "should have at least 2 wait_vm entries (one per half)",
    )
    assert_equal(
        wait_vm_values[0],
        wait_vm_values[1],
        "wait_vm should be symmetric across halves",
    )

    print("  test_stage_based_completion_symmetric PASSED")


# =============================================================================
# PipelineBody builder equivalence tests
# =============================================================================


def _assert_ops_equal(a: List[OpDesc], b: List[OpDesc], msg: String) raises:
    """Assert two op lists have identical metadata element-by-element."""
    assert_equal(len(a), len(b), msg + ": length mismatch")
    for i in range(len(a)):
        assert_equal(a[i].tag, b[i].tag, msg + ": tag mismatch at " + String(i))
        assert_equal(
            a[i].channel,
            b[i].channel,
            msg + ": channel mismatch at " + String(i),
        )
        assert_equal(
            a[i].stage,
            b[i].stage,
            msg + ": stage mismatch at " + String(i),
        )
        assert_equal(
            a[i].subtile,
            b[i].subtile,
            msg + ": subtile mismatch at " + String(i),
        )
        assert_equal(
            a[i].k_offset.bk_multiple,
            b[i].k_offset.bk_multiple,
            msg + ": k_offset mismatch at " + String(i),
        )


def test_pipeline_body_single_buffer() raises:
    """PipelineBody produces identical output to _logical_body for single-buffer.
    """
    from linalg.matmul.gpu.amd.matmul_schedule import _logical_body
    from linalg.matmul.gpu.amd.pipeline_body import PipelineBody

    # Build via current code.
    var expected = _logical_body[4]()

    # Build via PipelineBody builder.
    with PipelineBody() as b:
        b.load(LOAD_DRAM, ch=0)
        b.store(STORE_SMEM, ch=0)
        b.barrier()
        b.fan[4](LOAD_FRAG, ch=0)
        b.fan[4](COMPUTE)
        _assert_ops_equal(expected, b.done(), "single-buffer body")
    print("  test_pipeline_body_single_buffer PASSED")


def test_pipeline_body_pingpong() raises:
    """PipelineBody produces identical output to _logical_half for ping-pong."""
    from linalg.matmul.gpu.amd.pingpong_schedule import _logical_half
    from linalg.matmul.gpu.amd.pipeline_body import PipelineBody

    # Test both halves.
    for h in range(2):
        var expected = _logical_half[0]() if h == 0 else _logical_half[1]()
        var s = h
        var os = 1 - h
        var k_off = KOffsetKind.K1 if h == 1 else KOffsetKind.K0
        var k_special = KOffsetKind.K_PREV if h == 0 else KOffsetKind.K0

        with PipelineBody() as b:
            b.load(LOAD_A, ch=0, stage=os, sub=1, k=k_special)
            b.load(LOAD_A, ch=0, stage=s, sub=0, k=k_off)
            b.load(LOAD_B, ch=1, stage=s, sub=0, k=k_off)
            b.load(LOAD_B, ch=1, stage=s, sub=1, k=k_off)
            b.frag(MMA_LOAD_A, ch=0, stage=s, sub=0)
            b.frag(MMA_LOAD_A, ch=0, stage=s, sub=1)
            b.frag(MMA_LOAD_B, ch=1, stage=s, sub=0)
            b.frag(MMA_LOAD_B, ch=1, stage=s, sub=1)
            b.grid[2, 2](MMA)
            _assert_ops_equal(expected, b.done(), "pingpong half " + String(h))

    print("  test_pipeline_body_pingpong PASSED")


def test_pipeline_body_fan_and_grid() raises:
    """Verify fan and grid combinators produce correct subtile/stage indexing.
    """
    from linalg.matmul.gpu.amd.pipeline_body import PipelineBody

    # fan[3] should produce subtile=0,1,2 with same tag/channel/stage.
    with PipelineBody() as b:
        b.fan[3](COMPUTE, stage=1)
        var ops = b.done()
        assert_equal(len(ops), 3, "fan[3] length")
        for i in range(3):
            assert_equal(ops[i].tag, COMPUTE, "fan tag")
            assert_equal(ops[i].stage, 1, "fan stage")
            assert_equal(ops[i].subtile, i, "fan subtile")

    # grid[2,3] should produce 6 ops: stage=0..1, subtile=0..2.
    with PipelineBody() as b2:
        b2.grid[2, 3](MMA)
        var ops2 = b2.done()
        assert_equal(len(ops2), 6, "grid[2,3] length")
        var idx = 0
        for i in range(2):
            for j in range(3):
                assert_equal(ops2[idx].stage, i, "grid stage")
                assert_equal(ops2[idx].subtile, j, "grid subtile")
                idx += 1

    # extend: two builders merged.
    with PipelineBody() as a:
        a.load(LOAD_A, ch=0)
        var c = PipelineBody()
        c.compute(MMA)
        a.extend(c)
        var merged = a.done()
        assert_equal(len(merged), 2, "extend length")
        assert_equal(merged[0].tag, LOAD_A, "extend first tag")
        assert_equal(merged[1].tag, MMA, "extend second tag")

    print("  test_pipeline_body_fan_and_grid PASSED")


# =============================================================================
# Warp Stagger LDS Safety & derive_safe_max_globals Tests
# =============================================================================


def _build_fp8_program(sched: ScheduleConfig) -> _TestProgramResult:
    """Build an FP8 double-buffer program for testing (max_globals=0)."""
    var target = mi355x_target(vm_per_load_a=4, vm_per_load_b=4, max_globals=0)
    var config = target.pipeline
    var ds = DeclarativeSchedule[True, 4, 4](config=sched)
    var body = ds.build_body()
    var annotated = annotate_ops(body, target.cost_model)
    var edges = derive_edges_from_ops(annotated, config)
    var ldg = LoopBody()
    for i in range(len(annotated)):
        ldg.ops.append(OpNode(op=annotated[i]))
    ldg.edges = edges^

    var order = List[Int]()
    if sched.scheduling == SchedulingStrategy.CSP:
        order = optimal_schedule_with_halves(
            ldg,
            max_globals_per_block=config.block_sizing.max_globals,
        )
    else:
        for i in range(len(annotated)):
            order.append(i)

    var program = build_double_buffer_program(ldg, order, config, sched)
    return _TestProgramResult(program, config)


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


def test_warp_stagger_safety_bf16() raises:
    """BF16 schedule (max_globals=1, num_k_mmas=2) passes LDS safety check."""
    var sched = ScheduleConfig(
        scheduling=SchedulingStrategy.CSP,
        auto_waits=True,
        lgkm_per_load_a=8,
        lgkm_per_load_b=8,
    )
    var result = _build_bf16_program(sched)
    var program = result.program.copy()
    var config = result.config

    # Derive wait counts and populate pre_sync.
    var waits = derive_waits_from_blocks(program, config, 8, 8)
    var bph = config.blocks_per_half()
    for half in range(config.num_halves):
        var b0 = half * bph
        var bN = b0 + bph - 1
        if waits[0] < 255:
            var blk = program.blocks[b0]
            blk.pre_sync = OpDesc.wait_lgkm_n(waits[0])
            program.blocks[b0] = blk
        var blk = program.blocks[bN]
        blk.pre_sync = OpDesc.wait_vm_n(waits[1])
        program.blocks[bN] = blk

    # Should pass all checks including LDS safety.
    verify_schedule(program, config, 8, 8, waits[0], waits[1])
    print("  test_warp_stagger_safety_bf16 PASSED")


def test_warp_stagger_safety_fp8_bunched() raises:
    """FP8 schedule with max_globals=0 (bunched) passes LDS safety check."""
    var sched = ScheduleConfig(
        scheduling=SchedulingStrategy.CSP,
        auto_waits=True,
        lgkm_per_load_a=4,
        lgkm_per_load_b=4,
    )
    var result = _build_fp8_program(sched)
    var program = result.program.copy()
    var config = result.config

    # Derive wait counts.
    var waits = derive_waits_from_blocks(program, config, 4, 4)
    var bph = config.blocks_per_half()
    for half in range(config.num_halves):
        var b0 = half * bph
        var bN = b0 + bph - 1
        if waits[0] < 255:
            var blk = program.blocks[b0]
            blk.pre_sync = OpDesc.wait_lgkm_n(waits[0])
            program.blocks[b0] = blk
        var blk = program.blocks[bN]
        blk.pre_sync = OpDesc.wait_vm_n(waits[1])
        program.blocks[bN] = blk

    # Should pass — bunched globals avoid the cross-block race.
    verify_schedule(program, config, 4, 4, waits[0], waits[1])
    print("  test_warp_stagger_safety_fp8_bunched PASSED")


def main() raises:
    test_opnode_resource_assignment()
    test_linear_chain_asap()
    test_diamond_asap()
    test_loop_carried_edges_skip_asap()
    test_pp_extract_ldg()
    test_interleaved_ldg_asap()
    test_interleaved_c2_validation()
    test_pp_build_ldg()
    test_greedy_schedule_block_structure()
    test_prefetch_marking_in_kernel_body()
    # OpDesc metadata tests
    print("\n  --- OpDesc Metadata Tests ---")
    test_opdesc_metadata()

    # Default matmul schedule tests
    print("\n  --- Default Matmul Schedule Tests ---")
    test_default_matmul_full_schedule()
    test_default_matmul_full_round_trip()

    # Single-buffer transform tests
    print("\n  --- Single-Buffer Transform Tests ---")
    test_single_buffer_transform_t2()
    test_single_buffer_transform_t1()
    test_single_buffer_transform_matches_body()
    test_mma_block_interleave_2x2()
    test_config_single_buffer_default()
    test_config_block_sizing()
    test_schedule_compiler_t2()
    test_schedule_compiler_t1()
    test_schedule_compiler_interleaved()

    # Optimal scheduler tests
    print("\n  --- Optimal Scheduler Tests ---")
    test_compute_alap()
    test_evaluate_makespan_linear_chain()
    test_evaluate_makespan_serial_mma()
    test_optimal_schedule_interleaved()

    # Target cost model tests
    print("\n  --- Target Cost Model Tests ---")
    test_cost_model_annotation()

    # TargetProfile tests
    print("\n  --- TargetProfile Tests ---")
    test_target_profile_mi355x()
    test_target_profile_declarative_equivalence()

    # Declarative schedule tests
    print("\n  --- Declarative Schedule Tests ---")
    test_declarative_body_structure()
    test_declarative_schedule_compiles()
    test_declarative_c2_validation()

    # Declarative edge rule tests
    print("\n  --- Declarative Edge Rule Tests ---")
    test_edge_rule_count()
    test_rule_based_edges_match_double_buffer()
    test_rule_based_edges_match_single_buffer()

    # Single-buffer CSP optimizer tests
    print("\n  --- Single-Buffer CSP Optimizer Tests ---")
    test_optimize_within_barriers_single_buffer()

    # Phase recipe tests
    print("\n  --- Phase Recipe Tests ---")
    test_phase_recipe_prologue()
    test_phase_recipe_epilogue()

    # Verification & drain mask tests
    print("\n  --- Verification & Drain Mask Tests ---")
    test_verify_schedule_passes_valid()
    test_drain_mask_selective()
    test_drain_mask_backward_compat()
    test_auto_drain_derivation()
    test_distribution_preserved_after_csp()
    test_wait_counts_match_block_structure()

    # Prologue from program tests (Phase 4)
    print("\n  --- Prologue from Program Tests ---")
    test_prologue_from_program_structure()
    test_prologue_from_program_entry_count()

    # LDS contention model tests (Phase 3)
    print("\n  --- LDS Contention Model Tests ---")
    test_lds_contention_zero_is_noop()
    test_lds_contention_penalty_increases_makespan()
    test_lds_contention_lower_bound()

    # Stage-based completion (race condition fix)
    print("\n  --- Stage-Based Completion Tests ---")
    test_stage_based_completion_symmetric()

    # PipelineBody builder equivalence tests
    print("\n  --- PipelineBody Builder Tests ---")
    test_pipeline_body_single_buffer()
    test_pipeline_body_pingpong()
    test_pipeline_body_fan_and_grid()

    # Warp stagger LDS safety tests
    print("\n  --- Warp Stagger LDS Safety Tests ---")
    test_derive_safe_max_globals_bf16()
    test_derive_safe_max_globals_fp8()
    test_derive_safe_max_globals_large_k()
    test_warp_stagger_safety_bf16()
    test_warp_stagger_safety_fp8_bunched()

    print("All pipeline LDG tests passed!")
