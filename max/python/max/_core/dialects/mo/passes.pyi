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
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

from collections.abc import Sequence

import max._core

# C++ overloads on different int types look the same in Python, ignore these
# mypy: disable-error-code="overload-cannot-match"

def MOToMOGG(
    kernel_library_paths: Sequence[str] = [], force_sync: bool = False
) -> max._core.Pass:
    """
    This pass performs a conversion from known `mo` dialect ops to registered
    `mogg` dialect ops, performing kernel selection along the way.

    TODO: Expand this to also perform lowering search
    """

def AddFallbackShapeFunctions(
    kernel_library_paths: Sequence[str] = [],
) -> max._core.Pass:
    """
    This ensures all MO operations will have parameterized outputs by checking
    if they have already been parameterized and if not materializing a fallback
    shape function which will give them those parameters.
    """

def CanonicalizeChainSequences() -> max._core.Pass:
    """
    Will optimize a linear chain of side-effecting operations.

    Consider a case where we have two separate mo.buffers and the following
    sequence of operations:

    ```mlir
    %ch1, %vbuf = mo.mutable.load(%ch0, %buf0)
    ...
    %ch2 = mo.mutable.store(%ch1, %buf0, %res1)
    ...
    %ch3, %vbuf2 = mo.mutable.load(%ch2, %buf1)
    ...
    %ch4 = mo.mutable.store(%ch3, %buf1, %res2)
    ```

    Even though `%buf0` and `%buf1` are entirely different mo.buffer values the
    fact that operations on them are ordered by the same sequence of chains
    means the side-effecting operations on `%buf1` must unnecessarily wait for
    the operations on `%buf1` to complete.

    This pass will try to take such sequences of chains and perform a series of
    local transforms that will ultimately end with something like:

    ```mlir
    %ch1, %vbuf = mo.mutable.load(%ch0, %buf0)
    ...
    %ch2 = mo.mutable.store(%ch1, %buf0, %res1)
    ...
    %ch3, %vbuf2 = mo.mutable.load(%ch0, %buf1) // <- note the difference here
    ...
    %ch4 = mo.mutable.store(%ch3, %buf1, %res2)
    ```
    """

def ConstantFold(iterations: int = 2) -> max._core.Pass:
    """
    This pass identifies constant subgraphs and executes them at compile time
    using existing kernel infrastructure. Does not explicitly handle control flow.

    Example:
    ```
    mo.graph @const_program() -> !mo.tensor<[2], si32> {
      %0 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<0, 1> : tensor<2xsi32>} : !mo.tensor<[2], si32>
      %1 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<6, 3> : tensor<2xsi32>} : !mo.tensor<[2], si32>
      %2 = mo.mul(%0, %1) : !mo.tensor<[2], si32>
      %3 = mo.sub(%2, %1) : !mo.tensor<[2], si32>
      mo.output %3 : !mo.tensor<[2], si32>
    }
    ```
    becomes
    ```
    mo.graph @const_program() -> !mo.tensor<[2], si32> {
      %0 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<0, 2> : tensor<2xsi32>} : !mo.tensor<[2], si32>
      mo.output %0 : !mo.tensor<[2], si32>
    }
    ```

    This pass may be non-determininstic in ordering of output nodes.

    This pass run on ModuleOps instead of GraphOps because it 1. needs to call
    runPipeline(MOToMOGG) which cannot be nested in a GraphOp, and 2. it needs
    to avoid overhead of doing lowering and model setup multiple times.
    """

def ConstantFoldSubgraphTest() -> max._core.Pass:
    """
    This is a testing only pass. It prints out the constant subgraphs found
    within a graph so that they can be checked by filecheck. Printed output
    looks like this:
    ```
    mo.graph @const_program() -> !mo.tensor<[2], si32> {
      %0 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<0, 1> : tensor<2xsi32>} : !mo.tensor<[2], si32>
      %1 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<6, 3> : tensor<2xsi32>} : !mo.tensor<[2], si32>
      %2 = mo.mul(%0, %1) : !mo.tensor<[2], si32>
      %3 = mo.sub(%2, %1) : !mo.tensor<[2], si32>
      mo.output %3 : !mo.tensor<[2], si32>
    }
    subgraph %0 %1 %2 %3
    ```
    which is the full graph followed by zero or more occurrences of word
    'subgraph' with the values that compose a constant subgraph.
    """

def DebugPrintAllTensorsPass() -> max._core.Pass:
    """
    This pass emits a mo.debug.tensor.print for every mo.tensor typed value
    in the graph, including the inputs. The graph is topologically sorted to
    ensure the model won't end up with a cycle due to the chains which must
    serialize the underlying prints. For debugging only.
    """

def DecomposeViewOps() -> max._core.Pass:
    """
    Breaks down large CISC like view ops into a series of simpler
    RISC like view ops. Each one has a simpler lowering in mojo and
    offers potentially cleaner fusion semantics.
    """

def ElideRedundantMutableOps() -> max._core.Pass:
    """
    This pass contains patterns meant to eliminate mo.mutable.store and
    mo.mutable.load ops under appropriate conditions.

    For mo.mutable.load ops:
      Erase the load when it is preceded by a store operating on the same
      buffer and consumes the store's output chain.

      e.g.

      ```
      %ch1 = mo.mutable.store(%ch0, %buffer, %tensor0)
      %ch2, %tensor1 = mo.mutable.load(%ch1, %buffer)

      %res = mo.op(%tensor1, ...)

      %ch3 = mo.mutable.store(%ch2, %buffer, %res)

      ==>

      %ch1 = mo.mutable.store(%ch0, %buffer, %tensor0)

      %res = mo.op(%tensor0, ...)

      %ch3 = mo.mutable.store(%ch1, %buffer, %res)
      ```

    For mo.mutable.store ops:
      Erase a store when it is followed by another store operating on the same
      buffer and the first store's output chain

      e.g.

      ```
      %ch1 = mo.mutable.store(%ch0, %buffer, %tensor)
      %res = mo.op(%tensor, ...)
      %ch3 = mo.mutable.store(%ch1, %buffer, %res)

      ==>

      %res = mo.op(%tensor, ...)
      %ch3 = mo.mutable.store(%ch0, %buffer, %res)
      ```

    This enables the fusion passes in MOGG to work with in-place operations.
    """

def FlattenElementwise() -> max._core.Pass:
    """
    This pass will flatten the tensor shapes for certain elementwise ops if it
    unblocks elementwise fusion opportunities.
    """

def HoistConstantSubgraphsIntoParent() -> max._core.Pass:
    """
    This pass hoists constant and constant-derived ops within subgraphs to the
    parent graph.
    Our implementation of ExecInvariantCodeMotion doesn't support subgraphs, so
    this pass is necessary to ensure that constant operations are either
    correctly constant-folded or hoisted to the init block.
    """

def HoistParameterExpressions() -> max._core.Pass:
    """
    Transforms
    ```
    %1 = mo.op(%arg: !mo.tensor<[add(N,M)], f32>) : !mo.tensor<[add(N,M)], f32>
    ```
    into
    ```
    kgen.param.declare param_expr = <add(N,M)>
    %arg_rebound = mo.rebind(%arg:!mo.tensor<[add(N,M)], f32>) :!mo.tensor<[param_expr], f32>
    %tmp = mo.op(%arg_rebound: !mo.tensor<[param_expr], f32>) : !mo.tensor<[param_expr], f32>
    %1 = %mo.rebind(tmp: !mo.tensor<(N,M), f32>): !mo.tensor<[add(N,M)], f32>
    ```
    `rebind`s are inserted around the block to ensure that the input and outputs of the block remain the same type.

    This pass is necessary for MOGG to jit kernels with parameter expressions in their arguments.
    """

def InferLayouts() -> max._core.Pass:
    """
    This pass tries to infer the layout of tensor values in a `mo.graph`, based
    on constraints of relevant MO ops, e.g., before layout transformations,
    `mo.conv` assumes input has `NHWC` layout. At the end of the inference, the
    pass will annotate the tensor types with inferred layout attributes.
    """

def InvokeShapePackingFunctions(
    use_host_system: bool = True, sanity_check_shape_funcs: bool = False
) -> max._core.Pass:
    """
    Attempts to process the fallback shape functions for packing operations.
    Internally, this will call into a C++ re-implementation of the shape
    calculations.
    """

def MergeDuplicateShapeMaterializations() -> max._core.Pass:
    """
    This pass goes through ops that implement the Staticization interface and
    ensures that only a single op of its type is operating on the same
    operands, thus ensuring that multiple shape parameters are not being
    constructed for the same value.
    Example:
    ```
    mo.graph @example<N1 -> O1>(%arg0: !mo.tensor<[], bf16>, %arg1: !mo.tensor<[N1], bool>) -> !mo.tensor<[O1], bf16> {
      %0 = mo.shape_of(%arg1) : (!mo.tensor<[N1], bool>) -> !mo.tensor<[1], si32>
      %1 = mo.broadcast_to<() -> D1>(%arg0, %0) : (!mo.tensor<[], bf16>, !mo.tensor<[1], si32>) -> !mo.tensor<[D1], bf16>
      %2 = mo.rebind(%1) : !mo.tensor<[D1], bf16> -> !mo.tensor<[N1], bf16>
      %3 = mo.broadcast_to<() -> D2>(%arg0, %0) : (!mo.tensor<[], bf16>, !mo.tensor<[1], si32>) -> !mo.tensor<[D2], bf16>
      %4 = mo.rebind(%3) : !mo.tensor<[D2], bf16> -> !mo.tensor<[N1], bf16>
      %5 = mo.select(%arg1, %2, %4) : (!mo.tensor<[N1], bool>, !mo.tensor<[N1], bf16>, !mo.tensor<[N1], bf16>) -> !mo.tensor<[N1], bf16>
      mo.output<N1> %5 : !mo.tensor<[N1], bf16>
    }
    ```
    becomes
    ```
    mo.graph @example<N1 -> O1>(%arg0: !mo.tensor<[], bf16>, %arg1: !mo.tensor<[N1], bool>) -> !mo.tensor<[O1], bf16> {
      %0 = mo.shape_of(%arg1) : (!mo.tensor<[N1], bool>) -> !mo.tensor<[1], si32>
      %1 = mo.broadcast_to<() -> D1>(%arg0, %0) : (!mo.tensor<[], bf16>, !mo.tensor<[1], si32>) -> !mo.tensor<[D1], bf16>
      %2 = mo.rebind(%1) : !mo.tensor<[D1], bf16> -> !mo.tensor<[N1], bf16>
      %3 = mo.rebind(%1) : !mo.tensor<[D1], bf16> -> !mo.tensor<[N1], bf16>
      %4 = mo.select(%arg1, %2, %3) : (!mo.tensor<[N1], bool>, !mo.tensor<[N1], bf16>, !mo.tensor<[N1], bf16>) -> !mo.tensor<[N1], bf16>
      mo.output<N1> %4 : !mo.tensor<[N1], bf16>
    }
    ```
    Canonicalization and CSE can take this one step further and eliminate a number of additional operations.
    """

def PropagateShapes() -> max._core.Pass:
    """
    This pass evaluates shape parameter logic in the graph, simplifying and
    reducing shape logic as much as it can.
    """

def SplatLargeConstants(num_elements: int = 100000000) -> max._core.Pass:
    """
    This pass will replace a large constant with a splatted version by
    broadcasting a single scalar value. This pass does not preserve correctness
    so it should be used for development purposes only (especially for large
    models).
    """

def SupportConversionsTest() -> max._core.Pass:
    """
    This is a testing only pass. It looks for hard-coded, unregistered ops and
    expands them using helper libraries located in
    GenericML/GraphCompiler/Conversion/MO/Support/*.
    ```
    mo.graph @const_program() -> (!mo.tensor<[2], f32>) {
      ...
      // Pattern rewrite looks for "prelu" and expands op with decomposed
      // operators in MO to execute a "prelu" operation:
      %2 = "prelu"(%0, %1) : (!mo.tensor<[2], f32>, !mo.tensor<[1], f32>) -> !mo.tensor<[2], f32>
      mo.output %2 : !mo.tensor<[5], f32>
    }
    ```
    See SupportConversionsTest.cpp for more registered rewrite patterns.
    """

def SymbolicOptimization(
    symbolic_ffast_math: bool = True, symbolic_no_undefined_math: bool = True
) -> max._core.Pass:
    """
    This pass leverages symbolic information present in the graph (usually as a
    result of shape inference) to optimize computations. It includes
    transformations such as:
    - moving certain operations within the graph
    - rewriting a series of operations into more efficient ones
    - canonicalizing certain operations into simpler form to enable
      downstream optimizations
    """

def TopologicalSort() -> max._core.Pass:
    """
    This pass topologically sort ops within a `mo.graph` based on both SSA and
    parameter dependencies. It also handles nested regions properly (e.g.,
    `mo.if`).
    """

def ValidateDevicesPass(
    default_device_target_triple: str = "", default_device_target_arch: str = ""
) -> max._core.Pass:
    """
    This pass checks the inputs and outputs of the graph, verifying that the
    data type for each tensor can be implemented on the target device.
    """
