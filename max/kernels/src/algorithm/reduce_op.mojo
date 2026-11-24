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
"""Reduction monoid traits — the device-agnostic author surface.

A reduction algorithm is authored as a struct conforming to `ReduceOp`:
inlined state fields, `__init__` (identity), `accumulate[w]` (SIMD-tile
fold), `join` (sequential combine), and optionally
`join_parallel[R: Reducer]` (parallel combine). How participants
cooperate and how the parallel scope is defined live in the `Reducer`
impls.
"""

from std.math import log
from std.utils.static_tuple import StaticTuple


# ===-----------------------------------------------------------------------===#
# Reducer trait
# ===-----------------------------------------------------------------------===#


trait Reducer:
    """Parallel scalar reducer over the closed set of associative ops.

    Reduces a scalar across all participants in the current parallel
    scope; every participant receives the reduced value (like
    `MPI_Allreduce`).

    Concrete impls provide hardware-fast `sum` / `max` / `min` over a
    closed set of dtypes, plus a `generic` fallback that combines any
    `ReduceOp` via its `join` (slower; dtype-agnostic).
    """

    def sum[dtype: DType](self, val: Scalar[dtype]) -> Scalar[dtype]:
        """Returns the sum of `val` across all participants.

        Parameters:
            dtype: The scalar dtype.

        Args:
            val: The per-participant value.

        Returns:
            The reduced sum, broadcast to every participant.
        """
        ...

    def max[dtype: DType](self, val: Scalar[dtype]) -> Scalar[dtype]:
        """Returns the maximum of `val` across all participants.

        Parameters:
            dtype: The scalar dtype.

        Args:
            val: The per-participant value.

        Returns:
            The reduced maximum, broadcast to every participant.
        """
        ...

    def min[dtype: DType](self, val: Scalar[dtype]) -> Scalar[dtype]:
        """Returns the minimum of `val` across all participants.

        Parameters:
            dtype: The scalar dtype.

        Args:
            val: The per-participant value.

        Returns:
            The reduced minimum, broadcast to every participant.
        """
        ...

    def generic[S: ReduceOp](self, mut state: S):
        """Combines `state` across all participants using only `join` —
        the multi-field path for monoids whose state isn't decomposable
        into scalar `sum`/`max`/`min` (Welford, ArgMax, ArgMin, …).

        Impls pick by state size: small states (a few uint32 words) use
        a register-only warp-shuffle butterfly, larger states a shmem
        tree — both valid since `ReduceOp` is `TrivialRegisterPassable`.
        Monoids that *can* decompose (ReduceSum, ReduceMax,
        OnlineLogSumExp, …) instead call `reducer.sum`/`max`/`min`
        directly, which compile to one hardware `redux.sync` per field
        and are faster.

        Parameters:
            S: The monoid type being combined.

        Args:
            state: Per-participant state; on return, holds the combined
                value on every participant.
        """
        ...


# ===-----------------------------------------------------------------------===#
# ReduceOp trait
# ===-----------------------------------------------------------------------===#


trait ReduceOp(TrivialRegisterPassable):
    """Trait for an associative reduction monoid (identity +
    associative `join`).

    Required: `__init__` (identity), `accumulate[w]` (SIMD-tile fold),
    `join` (sequential combine). Optional: `join_parallel[R: Reducer]`
    (parallel combine across participants). The trait's default
    delegates to `reducer.generic` (only needs `join`); override to use
    a `Reducer`'s hardware-fast scalar primitives, or a field-wise
    algebraic decomposition that collapses many `join`s into a few
    scalar reductions.

    `TrivialRegisterPassable` because accumulators pass by value through
    the parallel reducer.

    `accumulate` carries a free `val_dtype` parameter because Mojo trait
    conformance treats a struct-param `dtype` and a trait-declared
    `comptime dtype` as distinct types; impls cast `val` to the
    accumulator dtype inline.

    Constraints:
        `join` must be associative. The default `join_parallel` also
        assumes commutativity (participants reduce in unspecified
        order); non-commutative monoids must override it.

        A monoid registered for the graph compiler must also follow these
        rules, which `mogg-verify-kernels` checks:

        - `width` names one of the struct's parameters directly
          (`comptime width = Self.W`).
        - Its outputs are its `SIMD` fields whose width is that parameter;
          it has at least one. Other fields are internal state.
        - Every other parameter is an output's dtype (inferred from the
          reduction's result types) or has a default.
        - It defines no kernel entry point (`execute`, `elementwise`, ...).
    """

    comptime Single: ReduceOp
    """The width-1 form of this monoid (`Self` at `W == 1`), produced by
    `reduce` and consumed by the scalar broadcast constructor."""

    comptime width: Int
    """The SIMD width `W` of the state, exposed so the trait's default
    `reduce` / broadcast iterate the lanes generically."""

    def __init__(out self):
        """Returns the identity element."""
        ...

    def __init__(out self, s: Self.Single):
        """Broadcasts a width-1 result across all `W` lanes — the inverse of
        `reduce`, so the body can read `result.slice[w]` uniformly. Default:
        write every lane via `__setitem__`."""
        self = Self()
        comptime for j in range(Self.width):
            self[j] = s

    def accumulate[
        val_dtype: DType, w: Int
    ](
        mut self,
        val: SIMD[val_dtype, w],
        idx: SIMD[.int64, w] = SIMD[.int64, w](0),
    ):
        """Folds a SIMD tile into `self` **elementwise** — lane `j`
        folds `(val[j], idx[j])` — casting to the accumulator dtype
        internally.

        `idx[j]` is where `val[j]` sits along the reduce axis. The
        scaffolder builds it per tier — consecutive positions
        (`base + iota`) when the lanes are partials of one output, or a
        constant (`base`) when the lanes are independent output columns
        — so the monoid never learns its tier. Value-only monoids
        (`ReduceSum`, `ReduceMax`, `ReduceMin`, `ReduceProduct`,
        `OnlineLogSumExp`) ignore `idx`; index-tracking monoids
        (`ArgMax`, `ArgMin`) fold it directly. Defaults to a zero vector
        for callers that don't track positions.

        Parameters:
            val_dtype: The tile's source dtype.
            w: SIMD width of the tile.

        Args:
            val: The SIMD tile to fold.
            idx: Per-lane axis positions of `val`'s lanes (defaults to
                a zero vector).
        """
        ...

    def join(mut self, other: Self):
        """Combines `self` with `other` associatively (sequential).

        Args:
            other: The state to combine into `self`.
        """
        ...

    def __getitem__(self, j: Int) -> Self.Single:
        """Returns lane `j` of the state as a width-1 monoid."""
        ...

    def __setitem__(mut self, j: Int, s: Self.Single):
        """Writes the width-1 monoid `s` into lane `j` of the state."""
        ...

    def reduce(self) -> Self.Single:
        """Collapses the `W` lane-wise partials of one output into a single
        width-1 result.

        The scaffolder calls it once, immediately before `join_parallel`, and
        **only on cooperative tiers** (`W` lanes are partials of one output).
        On the tiled tier the `W` lanes are independent outputs, so both
        `reduce` and `join_parallel` are skipped and the per-lane results
        stand.

        Default: fold the lanes through `join` (the loop is empty at
        `W == 1`, so this is a branchless identity there). Override for a
        faster horizontal reduce (e.g. `acc.reduce_add()` for `ReduceSum`).
        """
        comptime assert Self.width > 0, "ReduceOp width must be positive"
        var acc = self[0]
        comptime for j in range(1, Self.width):
            acc.join(self[j])
        return acc

    @inline(.always)
    def join_parallel[R: Reducer](mut self, reducer: R):
        """Combines `self` across all participants via `reducer`,
        leaving the combined value on every participant (and, for SIMD
        states, every lane).

        Runs **after** `reduce`, so `self`'s lanes already hold a
        single within-participant result; this only does the
        cross-participant step. Default: `reducer.generic(self)` —
        correct for any commutative + associative monoid, and since
        `reduce` left all lanes equal, the generic field-wise combine
        yields the result on every lane (no separate broadcast).
        Override to dispatch to `reducer.sum`/`max`/`min` for a faster
        per-dtype path, splatting the scalar result back across lanes.

        Parameters:
            R: The parallel scalar reducer.

        Args:
            reducer: The reducer instance.
        """
        reducer.generic(self)

    @staticmethod
    @inline(.always)
    def pad[
        Wout: Int,
        Tout: DType,
        identity: Scalar[Tout],
        val_dtype: DType,
        w_in: Int,
    ](val: SIMD[val_dtype, w_in]) -> SIMD[Tout, Wout]:
        """Pads `val` to width `Wout`, casting `val_dtype -> Tout`
        and filling lanes `[w_in, Wout)` with `identity`.

        Lets a monoid's `accumulate[w]` body stay a single expression —
        no `comptime if w == Self.W` branching. For `w_in == Wout`
        returns the cast directly; for `w_in < Wout` builds an
        identity-filled vector and overwrites lanes `[0, w_in)` with the
        cast `val`.

        Precondition: `w_in <= Wout`. Bodies arrange this by computing
        `W = rowwise.pick_simd_width[...]` once across all monoids and
        dtypes involved, so the scaffolder's tile width never exceeds
        any monoid's `Self.W`.

        The default suffices for single-field SIMD monoids. Multi-field
        monoids (`ArgMax`, `ArgMin`) call `pad` twice — once per field —
        with each field's own identity.

        Parameters:
            Wout: Output SIMD width.
            Tout: Output element dtype.
            identity: Padding value for tail lanes.
            val_dtype: Input element dtype.
            w_in: Input SIMD width.

        Args:
            val: The partial-width input.

        Returns:
            A `SIMD[Tout, Wout]` with `val` in lanes `[0, w_in)`
            and `identity` in lanes `[w_in, Wout)`.
        """
        comptime if w_in == Wout:
            return rebind[SIMD[Tout, Wout]](val.cast[Tout]())
        var out = SIMD[Tout, Wout](identity)

        comptime for j in range(w_in):
            out[j] = val[j].cast[Tout]()
        return out
