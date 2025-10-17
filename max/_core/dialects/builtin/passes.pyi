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
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

import enum
from collections.abc import Callable, Sequence

import max._core

# Many of the generated overloads for constructors are more specialized in
# C++ than they are in Python. For example, `int32_t` and `int64_t` and `size_t`
# all map to `int` in Python typing. It may not always be clear which of these
# overloads will be run for a given set of inputs (though in most cases it's the first one)
# but we disable mypy errors for shadowed overloads.
#
# mypy: disable-error-code="overload-cannot-match"

# DiagnosticHandlers aren't a thing that Python can reasonably provided. In most cases
# these are automatically provided, but there are a few custom verifiers not covered yet.
# This binding prevents errors in those cases.
DiagnosticHandler = Callable

class GreedySimplifyRegionLevel(enum.Enum):
    normal = 1

    disabed = 0

    aggressive = 2

def BubbleDownMemorySpaceCasts() -> max._core.Pass:
    """
    This pass tries to iteratively bubble down all possible memory-space cast
    operations. It is important to note that the determination of which casts
    are bubbled down is based on the interfaces
    `MemorySpaceCastConsumerOpInterface`, and `MemorySpaceCastOpInterface`, and
    not the pass. The pass only looks for operations implementing the
    `MemorySpaceCastConsumerOpInterface` interface, and invoking the interface
    methods to perform the bubbling down.

    Example:

    ```mlir
    func.func @op_with_cast_sequence(%arg0: memref<4x4xf32, 1>, %arg1: index, %arg2: f32) -> memref<16xf32> {
      %memspacecast = memref.memory_space_cast %arg0 : memref<4x4xf32, 1> to memref<4x4xf32>
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %expanded = memref.expand_shape %memspacecast [[0], [1, 2]] output_shape [4, 2, 2] : memref<4x4xf32> into memref<4x2x2xf32>
      %collapsed = memref.collapse_shape %expanded [[0, 1, 2]] : memref<4x2x2xf32> into memref<16xf32>
      %loaded = memref.load %collapsed[%c0] : memref<16xf32>
      %added = arith.addf %loaded, %arg2 : f32
      memref.store %added, %collapsed[%c0] : memref<16xf32>
      %atomic_result = memref.atomic_rmw addf %arg2, %collapsed[%c4] : (f32, memref<16xf32>) -> f32
      return %collapsed : memref<16xf32>
    }
    // mlir-opt --bubble-down-memory-space-casts
    func.func @op_with_cast_sequence(%arg0: memref<4x4xf32, 1>, %arg1: index, %arg2: f32) -> memref<16xf32> {
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %expand_shape = memref.expand_shape %arg0 [[0], [1, 2]] output_shape [4, 2, 2] : memref<4x4xf32, 1> into memref<4x2x2xf32, 1>
      %collapse_shape = memref.collapse_shape %expand_shape [[0, 1, 2]] : memref<4x2x2xf32, 1> into memref<16xf32, 1>
      %memspacecast = memref.memory_space_cast %collapse_shape : memref<16xf32, 1> to memref<16xf32>
      %0 = memref.load %collapse_shape[%c0] : memref<16xf32, 1>
      %1 = arith.addf %0, %arg2 : f32
      memref.store %1, %collapse_shape[%c0] : memref<16xf32, 1>
      %2 = memref.atomic_rmw addf %arg2, %collapse_shape[%c4] : (f32, memref<16xf32, 1>) -> f32
      return %memspacecast : memref<16xf32>
    }
    ```
    """

def CSE() -> max._core.Pass:
    """
    This pass implements a generalized algorithm for common sub-expression
    elimination. This pass relies on information provided by the
    `Memory SideEffect` interface to identify when it is safe to eliminate
    operations. See [Common subexpression elimination](https://en.wikipedia.org/wiki/Common_subexpression_elimination)
    for more general details on this optimization.
    """

def Canonicalizer(
    top_down_processing_enabled: bool = True,
    region_simplify_level: GreedySimplifyRegionLevel = GreedySimplifyRegionLevel.normal,
    max_iterations: int = 10,
    max_num_rewrites: int = -1,
    test_convergence: bool = False,
    disabled_patterns: Sequence[str] = [],
    enabled_patterns: Sequence[str] = [],
) -> max._core.Pass:
    """
    This pass performs various types of canonicalizations over a set of
    operations by iteratively applying the canonicalization patterns of all
    loaded dialects until either a fixpoint is reached or the maximum number of
    iterations/rewrites is exhausted. Canonicalization is best-effort and does
    not guarantee that the entire IR is in a canonical form after running this
    pass. See [Operation Canonicalization](Canonicalization.md) for more
    details.
    """

def CompositeFixedPointPass(
    name: str = "CompositeFixedPointPass",
    pipeline_str: str = "",
    max_iter: int = 10,
) -> max._core.Pass:
    """
    Composite pass runs provided set of passes until fixed point or maximum
    number of iterations reached.
    """

def ControlFlowSink() -> max._core.Pass:
    """
    This pass implements control-flow sink on operations that implement
    `RegionBranchOpInterface` by moving dominating operations whose only uses
    are in a conditionally-executed regions into those regions so that
    executions paths where their results are not needed do not perform
    unnecessary computations.

    This is similar (but opposite) to loop-invariant code motion, which hoists
    operations out of regions executed more than once. The implementation of
    control-flow sink uses a simple and conversative cost model: operations are
    never duplicated and are only moved into singly-executed regions.

    It is recommended to run canonicalization first to remove unreachable
    blocks: ops in unreachable blocks may prevent other operations from being
    sunk as they may contain uses of their results
    """

def GenerateRuntimeVerification() -> max._core.Pass:
    """
    This pass generates op-specific runtime checks using the
    `RuntimeVerifiableOpInterface`. It can be run for debugging purposes after
    passes that are suspected to introduce faulty IR.
    """

def Inliner(
    default_pipeline_str: str = "canonicalize",
    op_pipeline_list: Sequence[max._core.OpPassManager] = [],
    max_inlining_iterations: int = 4,
    inlining_threshold: int = 4294967295,
) -> max._core.Pass: ...
def LocationSnapshot(
    file_name: str = "",
    tag: str = "",
    enable_debug_info: bool = False,
    print_generic_op_form: bool = False,
    use_local_scope: bool = False,
    print_pretty_debug_info: bool = False,
) -> max._core.Pass:
    """
    This pass allows for generating new locations from the IR during any stage
    of compilation, by snapshotting the IR to a file and using that file to
    generate new locations for the operations.

    Depending on the value of the `tag` option, different resulting locations
    may be generated:

    * If unset, the original location of the operation is replaced.

    Example:

    ```mlir
    // old:
    ... loc("original_source.cpp":1:1)

    // new:
    ... loc("snapshot_source.mlir":10:10)
    ```

    * If set, the new location is fused with the original location in the form
    of a [`Name Location`](Dialects/Builtin.md/#nameloc) with the specified tag.

    Example:

    ```mlir
    // old:
    ... loc("original_source.cpp":1:1)

    // new:
    ... loc(fused["original_source.cpp":1:1, "snapshot"("snapshot_source.mlir":10:10)])
    ```
    """

def LoopInvariantCodeMotion() -> max._core.Pass: ...
def LoopInvariantSubsetHoisting() -> max._core.Pass: ...
def Mem2Reg(enable_region_simplification: bool = True) -> max._core.Pass:
    """
    This pass removes loads out of and stores into a memory slot, and turns
    them into direct uses of SSA values. This is done generically using the
    `PromotableAllocationOpInterface`, `PromotableOpInterface` and
    `PromotableMemOpInterface` interfaces.

    This pass will attempt to compute which definitions of the content of
    the memory slot reach operations that use the memory slot pointer. It
    will rewire or remove operations that use the slot pointer so they no
    longer use it. If any of this is not possible, the IR will be left
    without mutation.

    This pass only supports unstructured control-flow. Promotion of operations
    within subregions will not happen.
    """

def PrintIRPass(label: str = "") -> max._core.Pass:
    """
    Print the entire IR on the debug stream. This is meant for debugging
    purposes to inspect the IR at a specific point in the pipeline.
    """

def PrintOpStats(print_as_json: bool = False) -> max._core.Pass: ...
def RemoveDeadValues() -> max._core.Pass:
    """
    The goal of this pass is optimization (reducing runtime) by removing
    unnecessary instructions. Unlike other passes that rely on local information
    gathered from patterns to accomplish optimization, this pass uses a full
    analysis of the IR, specifically, liveness analysis, and is thus more
    powerful.

    Currently, this pass performs the following optimizations:
    (A) Removes function arguments that are not live,
    (B) Removes function return values that are not live across all callers of
    the function,
    (C) Removes unneccesary operands, results, region arguments, and region
    terminator operands of region branch ops, and,
    (D) Removes simple and region branch ops that have all non-live results and
    don't affect memory in any way,

    iff

    the IR doesn't have any non-function symbol ops, non-call symbol user ops
    and branch ops.

    Here, a "simple op" refers to an op that isn't a symbol op, symbol-user op,
    region branch op, branch op, region branch terminator op, or return-like.

    It is noteworthy that we do not refer to non-live values as "dead" in this
    file to avoid confusing it with dead code analysis's "dead", which refers to
    unreachable code (code that never executes on hardware) while "non-live"
    refers to code that executes on hardware but is unnecessary. Thus, while the
    removal of dead code helps little in reducing runtime, removing non-live
    values should theoretically have significant impact (depending on the amount
    removed).

    It is also important to note that unlike other passes (like `canonicalize`)
    that apply op-specific optimizations through patterns, this pass uses
    different interfaces to handle various types of ops and tries to cover all
    existing ops through these interfaces.

    It is because of its reliance on (a) liveness analysis and (b) interfaces
    that makes it so powerful that it can optimize ops that don't have a
    canonicalizer and even when an op does have a canonicalizer, it can perform
    more aggressive optimizations, as observed in the test files associated with
    this pass.

    Example of optimization (A):-

    ```
    int add_2_to_y(int x, int y) {
      return 2 + y
    }

    print(add_2_to_y(3, 4))
    print(add_2_to_y(5, 6))
    ```

    becomes

    ```
    int add_2_to_y(int y) {
      return 2 + y
    }

    print(add_2_to_y(4))
    print(add_2_to_y(6))
    ```

    Example of optimization (B):-

    ```
    int, int get_incremented_values(int y) {
      store y somewhere in memory
      return y + 1, y + 2
    }

    y1, y2 = get_incremented_values(4)
    y3, y4 = get_incremented_values(6)
    print(y2)
    ```

    becomes

    ```
    int get_incremented_values(int y) {
      store y somewhere in memory
      return y + 2
    }

    y2 = get_incremented_values(4)
    y4 = get_incremented_values(6)
    print(y2)
    ```

    Example of optimization (C):-

    Assume only `%result1` is live here. Then,

    ```
    %result1, %result2, %result3 = scf.while (%arg1 = %operand1, %arg2 = %operand2) {
      %terminator_operand2 = add %arg2, %arg2
      %terminator_operand3 = mul %arg2, %arg2
      %terminator_operand4 = add %arg1, %arg1
      scf.condition(%terminator_operand1) %terminator_operand2, %terminator_operand3, %terminator_operand4
    } do {
    ^bb0(%arg3, %arg4, %arg5):
      %terminator_operand6 = add %arg4, %arg4
      %terminator_operand5 = add %arg5, %arg5
      scf.yield %terminator_operand5, %terminator_operand6
    }
    ```

    becomes

    ```
    %result1, %result2 = scf.while (%arg2 = %operand2) {
      %terminator_operand2 = add %arg2, %arg2
      %terminator_operand3 = mul %arg2, %arg2
      scf.condition(%terminator_operand1) %terminator_operand2, %terminator_operand3
    } do {
    ^bb0(%arg3, %arg4):
      %terminator_operand6 = add %arg4, %arg4
      scf.yield %terminator_operand6
    }
    ```

    It is interesting to see that `%result2` won't be removed even though it is
    not live because `%terminator_operand3` forwards to it and cannot be
    removed. And, that is because it also forwards to `%arg4`, which is live.

    Example of optimization (D):-

    ```
    int square_and_double_of_y(int y) {
      square = y ^ 2
      double = y * 2
      return square, double
    }

    sq, do = square_and_double_of_y(5)
    print(do)
    ```

    becomes

    ```
    int square_and_double_of_y(int y) {
      double = y * 2
      return double
    }

    do = square_and_double_of_y(5)
    print(do)
    ```
    """

def SCCP() -> max._core.Pass:
    """
    This pass implements a general algorithm for sparse conditional constant
    propagation. This algorithm detects values that are known to be constant and
    optimistically propagates this throughout the IR. Any values proven to be
    constant are replaced, and removed if possible.

    This implementation is based on the algorithm described by Wegman and Zadeck
    in [“Constant Propagation with Conditional Branches”](https://dl.acm.org/doi/10.1145/103135.103136) (1991).
    """

def SROA() -> max._core.Pass:
    """
    Scalar Replacement of Aggregates. Replaces allocations of aggregates into
    independant allocations of its elements.

    Allocators must implement `DestructurableAllocationOpInterface` to provide
    the list of memory slots for which destructuring should be attempted.

    This pass will only be applied if all accessors of the aggregate implement
    the `DestructurableAccessorOpInterface`. If the accessors provide a view
    into the struct, users of the view must ensure it is used in a type-safe
    manner and within bounds by implementing `TypeSafeOpInterface`.
    """

def StripDebugInfo() -> max._core.Pass:
    """
    This pass strips the IR of any location information, by replacing all
    operation locations with [`unknown`](Dialects/Builtin.md/#unknownloc).
    """

def SymbolDCE() -> max._core.Pass:
    """
    This pass deletes all symbols that are found to be unreachable. This is done
    by computing the set of operations that are known to be live, propagating
    that liveness to other symbols, and then deleting all symbols that are not
    within this live set. Live symbols are those that have a
    [visibility](SymbolsAndSymbolTables.md/#symbol-visibility) that extends
    beyond the IR, e.g. `public`, or those that are referenced by live symbols
    or other non-Symbol operations.

    For example, consider the following input:

    ```mlir
    func.func private @dead_private_function()
    func.func private @live_private_function()

    // Note: The `public` isn't necessary here, as this is the default.
    func.func public @public_function() {
      "foo.return"() {uses = [@live_private_function]} : () -> ()
    }
    ```

    A known live function, `public_function`, contains a reference to an
    otherwise non-live function `live_private_function`. After running
    `symbol-dce`, only these two symbols should remain, as the final symbol
    `dead_private_function` is not visible outside of the current IR and there
    are no links to known-live operations. After running, we get the expected:

    ```mlir
    func.func private @live_private_function()

    func.func public @public_function() {
      "foo.return"() {uses = [@live_private_function]} : () -> ()
    }
    ```

    See [Symbols and SymbolTables](SymbolsAndSymbolTables.md) for more
    information on `Symbols`.
    """

def SymbolPrivatize(exclude: Sequence[str] = []) -> max._core.Pass:
    """
    This pass marks all top-level symbols of the operation run as `private`
    except if listed in `exclude` pass option.
    """

def TopologicalSort() -> max._core.Pass:
    """
    Recursively sorts all nested regions without SSA dominance in topological
    order. The main purpose is readability, as well as potentially processing of
    certain transformations and analyses. The function sorts the operations in
    all nested regions such that, as much as possible, all users appear after
    their producers.

    This sort is stable. If the block is already topologically sorted, the IR
    is not changed. Operations that form a cycle are moved to the end of the
    regions in a stable order.
    """

def ViewOpGraph(
    max_label_len: int = 20,
    print_attrs: bool = True,
    print_control_flow_edges: bool = False,
    print_data_flow_edges: bool = True,
    print_result_types: bool = True,
) -> max._core.Pass:
    """
    This pass prints a Graphviz graph of a module.

    - Operations are represented as nodes;
    - Uses (data flow) as edges;
    - Control flow as dashed edges;
    - Regions/blocks as subgraphs.

    By default, only data flow edges are printed.

    Note: See https://www.graphviz.org/doc/info/lang.html for more information
    about the Graphviz DOT language.
    """
