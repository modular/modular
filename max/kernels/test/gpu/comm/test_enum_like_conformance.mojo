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
"""Exercises runtime `__match` on the kernels' enum-like structs.

Each helper names every case so the match fails to compile if
`_enum_case_names` drifts from the comptime case constants. All cases are
materialized into runtime values so the `__match` dispatch is exercised at
runtime, not just at compile time.
"""

from std.testing import TestSuite, assert_equal

from comm.allreduce import AllReduceAlgorithm
from layout.layout_tensor import ThreadScope
from linalg.fp6_utils import FP6Format
from linalg.mx_format import MXFormat
from linalg.matmul.gpu.tile_scheduler import MatmulSchedule, RasterOrder
from linalg.matmul.gpu.tile_scheduler_splitk import ReductionMode
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    GEMMKind,
)
from linalg.matmul.vendor.blas import Backend
from nn.attention.mha_utils import FlashAttentionAlgorithm
from nn.attention.gpu.amd_structured.iglp import AMDIGLPStrategy
from nn.attention.gpu.nvidia.mha_tile_scheduler import MHASchedule
from nn.gather_scatter import ScatterOobIndexStrategy
from pipeline.config import SchedulingStrategy
from pipeline.phase_derivation import PhaseAction
from pipeline.types import Phase


def _allreduce_rank(algorithm: AllReduceAlgorithm) -> Int:
    __match algorithm:
        case .ONE_STAGE:
            return 0
        case .TWO_STAGE:
            return 1
        case .LAMPORT:
            return 2


def _thread_scope_rank(scope: ThreadScope) -> Int:
    __match scope:
        case .BLOCK:
            return 0
        case .WARP:
            return 1


def _fp6_rank(format: FP6Format) -> Int:
    __match format:
        case .E2M3:
            return 0
        case .E3M2:
            return 1


def _mx_rank(format: MXFormat) -> Int:
    """Returns -1 when no named case matches (raw backing values).

    `MXFormat` is non-exhaustive: its public `__init__(value: Int)` accepts
    raw backing values, and those must not select any named case.
    """
    __match format:
        case .FP8_E4M3:
            return 0
        case .FP8_E5M2:
            return 1
        case .FP6_E2M3:
            return 2
        case .FP6_E3M2:
            return 3
        case .FP4_E2M1:
            return 4
        case _:
            return -1


def _matmul_schedule_rank(schedule: MatmulSchedule) -> Int:
    __match schedule:
        case .NONE:
            return 0
        case .TILE1D:
            return 1
        case .TILE2D:
            return 2
        case .DS_SCHEDULER:
            return 3


def _raster_rank(order: RasterOrder) -> Int:
    __match order:
        case .AlongN:
            return 0
        case .AlongM:
            return 1


def _reduction_mode_rank(mode: ReductionMode) -> Int:
    __match mode:
        case .Deterministic:
            return 0
        case .Nondeterministic:
            return 1


def _gemm_kind_rank(kind: GEMMKind) -> Int:
    """Returns -1 when no named case matches (raw invalid values).

    `GEMMKind` is non-exhaustive: its implicit fieldwise initializer permits
    raw backing values, and those must not select any named case.
    """
    __match kind:
        case .GEMM:
            return 0
        case .BMM:
            return 1
        case .GMM:
            return 2
        case .BLOCK_SCALED_1D2D_FP8:
            return 3
        case _:
            return -1


def _backend_rank(backend: Backend) -> Int:
    __match backend:
        case .AUTOMATIC:
            return 0
        case .CUBLAS:
            return 1
        case .CUBLASLT:
            return 2
        case .ROCBLAS:
            return 3
        case .HIPBLASLT:
            return 4


def _fa_algorithm_rank(algorithm: FlashAttentionAlgorithm) -> Int:
    """Returns -1 when no named case matches (the `unspecified` state).

    The enum is non-exhaustive: `__init__(-1)` is the runtime "unspecified"
    value `init()` resolves, and it must not select any named case.
    """
    __match algorithm:
        case .NAIVE:
            return 0
        case .FLASH_ATTENTION_1:
            return 1
        case .FLASH_ATTENTION_2:
            return 2
        case .FLASH_ATTENTION_3:
            return 3
        case _:
            return -1


def _iglp_rank(strategy: AMDIGLPStrategy) -> Int:
    __match strategy:
        case .MFMA_SMALL_GEMM:
            return 0
        case .MFMA_SMALL_GEMM_SINGLE_WAVE:
            return 1
        case .MFMA_EXP_INTERLEAVE:
            return 2
        case .MFMA_EXP_SIMPLE_INTERLEAVE:
            return 3


def _mha_schedule_rank(schedule: MHASchedule) -> Int:
    __match schedule:
        case .DEFAULT:
            return 0
        case .PROMPT_ROTATE:
            return 1


def _scatter_strategy_rank(strategy: ScatterOobIndexStrategy) -> Int:
    __match strategy:
        case .UNDEFINED:
            return 0
        case .SKIP:
            return 1


def _scheduling_strategy_rank(strategy: SchedulingStrategy) -> Int:
    __match strategy:
        case .IDENTITY:
            return 0
        case .GREEDY:
            return 1
        case .CSP:
            return 2


def _phase_rank(phase: Phase) -> Int:
    __match phase:
        case .PROLOGUE:
            return 0
        case .KERNEL:
            return 1
        case .EPILOGUE:
            return 2


def _phase_action_rank(action: PhaseAction) -> Int:
    __match action:
        case .EMIT:
            return 0
        case .BARRIER:
            return 1
        case .FENCE:
            return 2


def test_allreduce_algorithm_match() raises:
    var algorithms = List[AllReduceAlgorithm]()
    algorithms.append(AllReduceAlgorithm.ONE_STAGE)
    algorithms.append(AllReduceAlgorithm.TWO_STAGE)
    algorithms.append(AllReduceAlgorithm.LAMPORT)
    for i in range(len(algorithms)):
        assert_equal(
            _allreduce_rank(algorithms[i]), i, "wrong AllReduceAlgorithm case"
        )


def test_thread_scope_match() raises:
    var scopes = List[ThreadScope]()
    scopes.append(ThreadScope.BLOCK)
    scopes.append(ThreadScope.WARP)
    for i in range(len(scopes)):
        assert_equal(_thread_scope_rank(scopes[i]), i, "wrong ThreadScope case")


def test_fp6_format_match() raises:
    var formats = List[FP6Format]()
    formats.append(FP6Format.E2M3)
    formats.append(FP6Format.E3M2)
    for i in range(len(formats)):
        assert_equal(_fp6_rank(formats[i]), i, "wrong FP6Format case")


def test_mx_format_match() raises:
    var formats = List[MXFormat]()
    formats.append(MXFormat.FP8_E4M3)
    formats.append(MXFormat.FP8_E5M2)
    formats.append(MXFormat.FP6_E2M3)
    formats.append(MXFormat.FP6_E3M2)
    formats.append(MXFormat.FP4_E2M1)
    for i in range(len(formats)):
        assert_equal(_mx_rank(formats[i]), i, "wrong MXFormat case")

    # A raw backing value outside the named cases must not select any of
    # them (non-exhaustive enum).
    var invalid = MXFormat(99)
    assert_equal(_mx_rank(invalid), -1, "raw MXFormat(99) must match no case")


def test_matmul_schedule_match() raises:
    var schedules = List[MatmulSchedule]()
    schedules.append(MatmulSchedule.NONE)
    schedules.append(MatmulSchedule.TILE1D)
    schedules.append(MatmulSchedule.TILE2D)
    schedules.append(MatmulSchedule.DS_SCHEDULER)
    for i in range(len(schedules)):
        assert_equal(
            _matmul_schedule_rank(schedules[i]), i, "wrong MatmulSchedule case"
        )


def test_raster_order_match() raises:
    var orders = List[RasterOrder]()
    orders.append(RasterOrder.AlongN)
    orders.append(RasterOrder.AlongM)
    for i in range(len(orders)):
        assert_equal(_raster_rank(orders[i]), i, "wrong RasterOrder case")


def test_reduction_mode_match() raises:
    var modes = List[ReductionMode]()
    modes.append(ReductionMode.Deterministic)
    modes.append(ReductionMode.Nondeterministic)
    for i in range(len(modes)):
        assert_equal(
            _reduction_mode_rank(modes[i]), i, "wrong ReductionMode case"
        )


def test_gemm_kind_match() raises:
    var kinds = List[GEMMKind]()
    kinds.append(GEMMKind.GEMM)
    kinds.append(GEMMKind.BMM)
    kinds.append(GEMMKind.GMM)
    kinds.append(GEMMKind.BLOCK_SCALED_1D2D_FP8)
    for i in range(len(kinds)):
        assert_equal(_gemm_kind_rank(kinds[i]), i, "wrong GEMMKind case")

    # A raw backing value outside the named cases must not select any of
    # them (non-exhaustive enum).
    var invalid = GEMMKind(99)
    assert_equal(
        _gemm_kind_rank(invalid), -1, "raw GEMMKind(99) must match no case"
    )


def test_backend_match() raises:
    var backends = List[Backend]()
    backends.append(Backend.AUTOMATIC)
    backends.append(Backend.CUBLAS)
    backends.append(Backend.CUBLASLT)
    backends.append(Backend.ROCBLAS)
    backends.append(Backend.HIPBLASLT)
    for i in range(len(backends)):
        assert_equal(_backend_rank(backends[i]), i, "wrong Backend case")


def test_flash_attention_algorithm_match() raises:
    var algorithms = List[FlashAttentionAlgorithm]()
    algorithms.append(FlashAttentionAlgorithm.NAIVE)
    algorithms.append(FlashAttentionAlgorithm.FLASH_ATTENTION_1)
    algorithms.append(FlashAttentionAlgorithm.FLASH_ATTENTION_2)
    algorithms.append(FlashAttentionAlgorithm.FLASH_ATTENTION_3)
    for i in range(len(algorithms)):
        assert_equal(
            _fa_algorithm_rank(algorithms[i]),
            i,
            "wrong FlashAttentionAlgorithm case",
        )

    # The `-1` "unspecified" value is outside the named cases; it must not
    # select any of them (non-exhaustive enum).
    var unspecified = FlashAttentionAlgorithm(-1)
    assert_equal(
        _fa_algorithm_rank(unspecified),
        -1,
        "unspecified (-1) must match no case",
    )


def test_iglp_strategy_match() raises:
    var strategies = List[AMDIGLPStrategy]()
    strategies.append(AMDIGLPStrategy.MFMA_SMALL_GEMM)
    strategies.append(AMDIGLPStrategy.MFMA_SMALL_GEMM_SINGLE_WAVE)
    strategies.append(AMDIGLPStrategy.MFMA_EXP_INTERLEAVE)
    strategies.append(AMDIGLPStrategy.MFMA_EXP_SIMPLE_INTERLEAVE)
    for i in range(len(strategies)):
        assert_equal(_iglp_rank(strategies[i]), i, "wrong AMDIGLPStrategy case")


def test_mha_schedule_match() raises:
    var schedules = List[MHASchedule]()
    schedules.append(MHASchedule.DEFAULT)
    schedules.append(MHASchedule.PROMPT_ROTATE)
    for i in range(len(schedules)):
        assert_equal(
            _mha_schedule_rank(schedules[i]), i, "wrong MHASchedule case"
        )


def test_scatter_strategy_match() raises:
    var strategies = List[ScatterOobIndexStrategy]()
    strategies.append(ScatterOobIndexStrategy.UNDEFINED)
    strategies.append(ScatterOobIndexStrategy.SKIP)
    for i in range(len(strategies)):
        assert_equal(
            _scatter_strategy_rank(strategies[i]),
            i,
            "wrong ScatterOobIndexStrategy case",
        )


def test_scheduling_strategy_match() raises:
    var strategies = List[SchedulingStrategy]()
    strategies.append(SchedulingStrategy.IDENTITY)
    strategies.append(SchedulingStrategy.GREEDY)
    strategies.append(SchedulingStrategy.CSP)
    for i in range(len(strategies)):
        assert_equal(
            _scheduling_strategy_rank(strategies[i]),
            i,
            "wrong SchedulingStrategy case",
        )


def test_phase_match() raises:
    var phases = List[Phase]()
    phases.append(Phase.PROLOGUE)
    phases.append(Phase.KERNEL)
    phases.append(Phase.EPILOGUE)
    for i in range(len(phases)):
        assert_equal(_phase_rank(phases[i]), i, "wrong Phase case")


def test_phase_action_match() raises:
    var actions = List[PhaseAction]()
    actions.append(PhaseAction.EMIT)
    actions.append(PhaseAction.BARRIER)
    actions.append(PhaseAction.FENCE)
    for i in range(len(actions)):
        assert_equal(
            _phase_action_rank(actions[i]), i, "wrong PhaseAction case"
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
