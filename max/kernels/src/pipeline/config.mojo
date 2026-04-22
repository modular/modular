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
"""Pipeline configuration structs: LoopCarriedSpec, BlockSizing, FragOrder,
SchedulingStrategy, ScheduleConfig, WarpStaggerRule, PipelineConfig, and
TargetProfile.
"""

from .types import OpDesc, OpRole, TargetCostModel

# =============================================================================
# Pipeline Configuration
# =============================================================================


@fieldwise_init
struct LoopCarriedSpec(ImplicitlyCopyable, Movable):
    """Which ops are loop-carried (loaded at end of iter, consumed at start).

    For single-buffer pipelines, fragment[0] is typically loop-carried:
    it's loaded at the end of iteration N and consumed by compute[0] at
    the start of iteration N+1.
    """

    var role: OpRole  # Which role is loop-carried (e.g., FRAGMENT_LOAD)
    var selector: Int  # Which subtile (e.g., 0 for fragment[0])

    @staticmethod
    def fragment_zero() -> Self:
        """Fragment[0] is loop-carried (default matmul pattern)."""
        return Self(role=OpRole.FRAGMENT_LOAD, selector=0)

    @staticmethod
    def none() -> Self:
        """No loop-carried ops (double-buffered pipelines)."""
        return Self(role=OpRole.NONE, selector=0)


@fieldwise_init
struct BlockSizing(ImplicitlyCopyable, Movable):
    """Target op count per MMA block for latency hiding.

    Heavy blocks (introducing a new M-tile row) get more ops to fill the
    MMA latency window. Light blocks (continuation of same M-tile) get
    fewer ops since only one new fragment is needed.
    """

    var heavy: Int  # Blocks introducing a new M-tile row
    var light: Int  # Continuation blocks (same M-tile)
    var max_globals: Int  # Max global loads per block (0=unlimited)

    @staticmethod
    def default_2x2() -> Self:
        """2x2 MMA grid default: 4 ops for new M-tile, 2 for continuation."""
        return Self(heavy=4, light=2, max_globals=0)

    @staticmethod
    def uniform(n: Int) -> Self:
        """All blocks get the same target size."""
        return Self(heavy=n, light=n, max_globals=0)


@fieldwise_init
struct FragOrder(ImplicitlyCopyable, Movable):
    """Fragment load ordering within an MMA block."""

    var b_before_a: Bool  # True: B-frag first. False: A-frag first.

    @staticmethod
    def b_first() -> Self:
        """B-fragment before A-fragment."""
        return Self(b_before_a=True)

    @staticmethod
    def a_first() -> Self:
        """A-fragment before B-fragment."""
        return Self(b_before_a=False)


@fieldwise_init
struct SchedulingStrategy(Equatable, ImplicitlyCopyable, Movable):
    """Scheduling strategy for pipeline op ordering.

    IDENTITY: Declaration order (no scheduling).
    GREEDY: Greedy greedy_schedule() heuristic.
    CSP: Backtracking CSP solver (provably optimal).
    """

    var _value: Int

    comptime IDENTITY = Self(0)
    comptime GREEDY = Self(1)
    comptime CSP = Self(2)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return self._value != other._value


struct ScheduleConfig(ImplicitlyCopyable, Movable):
    """Tunable parameters for schedule generation.

    Controls the structural decisions in program builders:
    scheduling strategy, barrier placement, and drain behavior.

    The default configuration auto-derives wait counts from the program
    structure (Halide-inspired: declare intent, derive consequences).
    Manual wait overrides are available for testing and experimentation
    but should not be needed for correct operation.

    Fields:
        scheduling: Strategy for op ordering (IDENTITY, GREEDY, or CSP).
        sched_barrier_mask: Bitmask of which blocks get trailing
            schedule_barriers. Default: 0b01010101 (blocks 0,2,4,6).
        auto_waits: Auto-derive wait counts from schedule order (default: True).
        drain_lgkm_mask: Per-block bitmask for selective LDS drains.
        auto_drain: Auto-derive drain mask from channel analysis.
        lds_contention_penalty: CSP solver penalty for LDS port overlap.
        wait_lgkm_first: Manual wait_lgkm(N) override (used when auto_waits=False).
        wait_vm_last: Manual wait_vm(N) override (used when auto_waits=False).
        lgkm_per_load_a: lgkmcnt ops per load_a (for wait derivation).
        lgkm_per_load_b: lgkmcnt ops per load_b (for wait derivation).
        lgkm_after_last: Insert wait_lgkm(0) after last block barrier.
    """

    var scheduling: SchedulingStrategy
    var sched_barrier_mask: Int
    var auto_waits: Bool
    var drain_lgkm_mask: Int
    var auto_drain: Bool
    var lds_contention_penalty: Int
    # Manual wait overrides (only used when auto_waits=False).
    var wait_lgkm_first: Int
    var wait_vm_last: Int
    var lgkm_per_load_a: Int
    var lgkm_per_load_b: Int
    var lgkm_after_last: Bool

    @always_inline
    def __init__(
        out self,
        *,
        scheduling: SchedulingStrategy = SchedulingStrategy.IDENTITY,
        sched_barrier_mask: Int = 0b01010101,
        auto_waits: Bool = True,
        drain_lgkm_mask: Int = 0,
        auto_drain: Bool = False,
        lds_contention_penalty: Int = 0,
        wait_lgkm_first: Int = 8,
        wait_vm_last: Int = 6,
        lgkm_per_load_a: Int = 0,
        lgkm_per_load_b: Int = 0,
        lgkm_after_last: Bool = False,
    ):
        self.scheduling = scheduling
        self.sched_barrier_mask = sched_barrier_mask
        self.auto_waits = auto_waits
        self.drain_lgkm_mask = drain_lgkm_mask
        self.auto_drain = auto_drain
        self.lds_contention_penalty = lds_contention_penalty
        self.wait_lgkm_first = wait_lgkm_first
        self.wait_vm_last = wait_vm_last
        self.lgkm_per_load_a = lgkm_per_load_a
        self.lgkm_per_load_b = lgkm_per_load_b
        self.lgkm_after_last = lgkm_after_last


@fieldwise_init
struct WarpStaggerRule(ImplicitlyCopyable, Movable):
    """Declarative warp stagger configuration.

    Controls how warp groups are staggered in the prologue to hide
    latency. The stagger index determines how many prologue ops G0
    executes before G1 starts.

    Modes:
      - enabled=False: no stagger (single-buffer default)
      - enabled=True, compute_from_body=True: derive stagger by counting
        stage-0 prefetch loads in the body (double-buffer default)
      - enabled=True, compute_from_body=False: use fixed_amount directly
    """

    var enabled: Bool  # Whether to apply warp stagger
    var compute_from_body: Bool  # Derive amount from body analysis
    var fixed_amount: Int  # Used when compute_from_body is False

    @staticmethod
    def none() -> Self:
        """No stagger (single-buffer default)."""
        return Self(enabled=False, compute_from_body=False, fixed_amount=0)

    @staticmethod
    def auto() -> Self:
        """Auto-derive stagger from body (double-buffer default)."""
        return Self(enabled=True, compute_from_body=True, fixed_amount=0)

    @staticmethod
    def fixed(amount: Int) -> Self:
        """Fixed stagger amount."""
        return Self(enabled=True, compute_from_body=False, fixed_amount=amount)


@fieldwise_init
struct PipelineConfig(ImplicitlyCopyable, Movable):
    """Declarative pipeline strategy.

    Captures all the knowledge needed to transform a logical loop body
    into a pipelined schedule: buffer depth, prefetch distance, loop-carried
    edges, MMA block sizing, and hardware model.

    Platform-specific factories (e.g., mi355x_double_buffer() in
    amd_target.mojo) provide tuned configurations.
    """

    # --- Buffer strategy ---
    var depth: Int  # 1 = single-buffer, 2 = double-buffer
    var prefetch: Int  # Iterations ahead for DRAM loads (typically 1)

    # --- Phase derivation ---
    var drain_passes: Int  # Epilogue drain iterations
    var prologue_fill: Int  # Extra load iterations in prologue (depth - 1)

    # --- Execution ordering ---
    var loop_carried: LoopCarriedSpec  # Ops crossing iteration boundaries
    var block_sizing: BlockSizing  # MMA block op targets
    var frag_order: FragOrder  # Fragment ordering within a block

    # --- MMA grid ---
    var m_mmas: Int  # M-dimension MMA tiles (rows in spatial grid)
    var n_mmas: Int  # N-dimension MMA tiles (cols in spatial grid)
    var num_halves: Int  # Number of warp groups (1 for single, 2 for ping-pong)

    # --- Hardware model ---
    var mma_serial: Bool  # MMA unit is serial (capacity 1)
    var mma_latency: Int  # MMA latency in cycles
    var vm_per_load_a: Int  # vmcnt ops per global load (channel 0 / A)
    var vm_per_load_b: Int  # vmcnt ops per global load (channel 1 / B)

    # --- Channel configuration ---
    # For register FLOW edges: which compute field does each channel's
    # fragment subtile match against?
    #   0 = match compute.stage (row), 1 = match compute.subtile (col)
    # channel 0 (A): frag.subtile matches compute.stage (row)
    # channel 1 (B): frag.subtile matches compute.subtile (col)
    var ch0_match_field: Int  # 0 = stage, 1 = subtile
    var ch1_match_field: Int  # 0 = stage, 1 = subtile

    # --- Warp stagger ---
    var warp_stagger: WarpStaggerRule  # Warp group stagger configuration

    # --- Derived counts (no magic numbers) ---

    def mmas_per_half(self) -> Int:
        """MMA ops per warp group: m_mmas × n_mmas."""
        return self.m_mmas * self.n_mmas

    def globals_per_half(self) -> Int:
        """Global loads per warp group: m_mmas + n_mmas (A + B tiles)."""
        return self.m_mmas + self.n_mmas

    def frags_per_half(self) -> Int:
        """Fragment loads per warp group: m_mmas + n_mmas (A + B frags)."""
        return self.m_mmas + self.n_mmas

    def ops_per_half(self) -> Int:
        """Total ops per warp group."""
        return (
            self.globals_per_half()
            + self.frags_per_half()
            + self.mmas_per_half()
        )

    def total_ops(self) -> Int:
        """Total ops across all warp groups."""
        return self.num_halves * self.ops_per_half()

    def blocks_per_half(self) -> Int:
        """MMA blocks per warp group (one block per MMA)."""
        return self.mmas_per_half()

    def total_blocks(self) -> Int:
        """Total MMA blocks."""
        return self.num_halves * self.blocks_per_half()

    def compute_match_key(self, compute_op: OpDesc, channel: Int) -> Int:
        """Extract the compute field that a fragment on `channel` matches.

        For channel 0 (A): returns compute.stage (row).
        For channel 1 (B): returns compute.subtile (col).
        """
        var field = (
            self.ch0_match_field if channel == 0 else self.ch1_match_field
        )
        return compute_op.stage if field == 0 else compute_op.subtile

    def vm_per_channel(self, channel: Int) -> Int:
        """Return vmcnt cost for a global load on the given channel."""
        return self.vm_per_load_a if channel == 0 else self.vm_per_load_b

    def total_edges(self) -> Int:
        """Total dependency edges for double-buffer pipeline.

        Four phases of edges connect ops within and across iterations:

        - reg_flow: fragment_load → compute (register FLOW).
          Each half has 2 channels (A, B), each channel's frag feeds
          m*n compute ops.
        - accum: compute → compute (accumulator forwarding).
          m*n accumulator tiles forwarded between halves, twice
          (half0→half1 at d=0, half1→half0 at d=1).
        - lds_flow: global_load → fragment_load (LDS FLOW).
          Each half has g global loads, each feeds one frag per buffer
          stage (×2 for double-buffering).
        - lds_anti: fragment_load → global_load (LDS ANTI).
          Prevents a prefetch write from overwriting data a frag still
          needs. 2*g frags total minus 1 (last frag has no successor
          load), doubled for both channels.
        """
        var m = self.m_mmas
        var n = self.n_mmas
        var h = self.num_halves
        var g = self.globals_per_half()
        var reg_flow = h * 2 * m * n
        var accum = m * n * 2
        var lds_flow = h * g * 2
        var lds_anti = (2 * g - 1) * 2
        return reg_flow + accum + lds_flow + lds_anti


# =============================================================================
# Target Profile — unified hardware description
# =============================================================================


struct TargetProfile(ImplicitlyCopyable, Movable):
    """Unified hardware target description.

    Bundles everything the framework needs to know about a specific GPU
    target into a single struct:
      - cost_model: per-op costs (resource, latency, role)
      - pipeline: pipeline structure (depth, MMA grid, buffer strategy)

    The algorithm declares WHAT ops exist (logical op table). The
    TargetProfile describes HOW the hardware executes them. Platform-
    specific factories (e.g., mi355x_target() in amd_target.mojo) provide
    both from a single call — no redundant configuration.

    Usage:
        # Platform-specific factory (from amd_target):
        comptime target = mi355x_target()
        var annotated = annotate_ops(logical_ops, target.cost_model)
        var config = target.pipeline
    """

    var cost_model: TargetCostModel
    var pipeline: PipelineConfig

    def __init__(
        out self,
        cost_model: TargetCostModel,
        pipeline: PipelineConfig,
    ):
        self.cost_model = cost_model
        self.pipeline = pipeline
