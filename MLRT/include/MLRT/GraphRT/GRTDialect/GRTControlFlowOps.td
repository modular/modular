//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// This defines the control flow operations in GRT dialect.
//
//===----------------------------------------------------------------------===//

include "MLRT/GraphRT/MEFDialect/MEFNonStrictInterface.td"
include "MLRT/GraphRT/GRTDialect/GRTOpsBase.td"
include "MLRT/GraphRT/GRTDialect/GRTControlFlowInterface.td"
include "MLRT/GraphRT/GRTDialect/GRTChainInterface.td"
include "MLRT/GraphRT/MEFDialect/MEFTypes.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// ChainCreateOp
//===----------------------------------------------------------------------===//

def ChainCreateOp : GRTOp<"chain.create", [Pure, Commutative,
    DeclareOpInterfaceMethods<GRT_ChainCreateOpInterface>,
    DeclareOpInterfaceMethods<GRT_GatedOnChainOpInterface>
    ]> {
  let summary = "Return a ready !mef.chain when all inputs (if any) are ready";

  let description = [{
    This operation returns a ready !mef.chain when all of its inputs are ready.
    If there are no inputs a ready chain is returned immediately. The operands
    may be of any type, not just !mef.chain. However, non !mef.chain operands
    must be paired with a type annotation.

    Example:

    ```mlir
      %ch0 = grt.chain.create ()
      %idx: index
      %int: i32
      %ch1 = grt.chain.create (%idx: index, %ch0, %int: i32)
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs ChainType:$result);
  let hasCustomAssemblyFormat = 1;
  let hasCanonicalizeMethod = 1;

  let builders = [
    OpBuilder<(ins)>
  ];
  let extraClassDefinition = [{
      ValueRange ChainCreateOp::getInputChains() {
        return getOperands();
      }
      void ChainCreateOp::setInputChains(::mlir::ValueRange inChains) {
        return getOperation()->setOperands(inChains);
      }
  }];
}

//===----------------------------------------------------------------------===//
// InvokeOp
//===----------------------------------------------------------------------===//

def InvokeOp : GRTOp<"invoke", [DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                                NonStrictInterface]> {
  let summary = "Invoke a callable (non-strictly).";

  let arguments = (ins FlatSymbolRefAttr:$target, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    $target `(` $operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

def IfOp : GRTOp<"if",
          [IsolatedFromAbove, SingleBlockImplicitTerminator<"MEF::OutputOp">,
           NonStrictInterface, RegionKindInterface, HasOnlyGraphRegion,
           DeclareOpInterfaceMethods<GRTControlFlowInterface>]> {
  let summary = "if control flow operation";
  let description = [{
    The "grt.if" operation takes an i1 condition, a 'then' block and an 'else'
    block. If the condition is true, the 'then' block is executed and the
    "grt.if" operation returns the results of that block. Otherwise the 'else'
    block is executed and its values are returned.

    Both blocks have access to the outer SSA values listed in the "grt.if"
    lhs signature. Both must return results with the same types as the "grt.if"
    rhs signature, using a "mef.output" operation at the end of each block.

    This operation is 'nonstrict'. When the condition is ready the 'then' or
    'else' blocks will be merged into the overall dataflow graph without
    waiting for any additional arguments to become ready.

    Example:

    ```mlir
      %res = grt.if %cond (%x: i32, %y: f32) -> i32 {
        %v1 = "use"(%x, %y) : (i32, f32) -> i32
        mef.output %v1 : i32
      } else {
        %v2 = "abuse"(%x, %y) : (i32, f32) -> i32
        mef.output %v2 : i32
      }
    ```
  }];

  let arguments = (ins I1:$cond, Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$thenRegion,
                        SizedRegion<1>:$elseRegion);

  let builders = [
    OpBuilder<(ins "Value":$cond, CArg<"ValueRange">:$inputs,
                    CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>">:$thenBuilder,
                    CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>">:$elseBuilder)>
  ];


  let hasCustomAssemblyFormat = 1;

  let extraClassDefinition = [{
    using BodyBuilder = function_ref<void(OpBuilder &, Location, ValueRange)>;

    GRT::ControlFlowFlavor IfOp::getFlavor() { return GRT::kBranchLike; }
    llvm::ArrayRef<mlir::OpOperand> IfOp::getInputOpOperands() {
      return getOperation()->getOpOperands().drop_front();
    }
    mlir::Operation::operand_range IfOp::getStrictExtraOperands() {
      return getOperation()->getOperands().take_front();
    }
    llvm::ArrayRef<mlir::OpOperand> IfOp::getStrictExtraOpOperands() {
      return getOperation()->getOpOperands().take_front();
    }
    llvm::SmallVector<llvm::StringRef> IfOp::getRegionNames() { return {"then", "else"}; }
  }];
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

def WhileOp : GRTOp<"while",
          [IsolatedFromAbove, SingleBlockImplicitTerminator<"MEF::OutputOp">,
           NonStrictInterface, RegionKindInterface, HasOnlyGraphRegion,
           DeclareOpInterfaceMethods<GRTControlFlowInterface>]> {
  let summary = "while control flow operation";
  let description = [{
    The "grt.while" operation takes 'cond' and 'body' blocks. While the
    'cond' block evaluates to the i1 true condition, the 'body' block is
    executed.

    Both the 'cond' and 'body' blocks have access to the outer SSA values listed
    in the "grt.while" signature. The 'cond' block must return results with the
    same types as the "grt.while" signatures, followed by an i1 indicating the
    condition result, using a "mef.output" op. The 'body' block must return
    results with the same types as the "grt.while" signature, again using the
    "mef.output" operation. Each execution of the 'cond' or 'body' block
    effectively 'updates' the arguments given in the "grt.while" signature,
    and the results of the "grt.while" op are the last 'updated' arguments.

    Note we allow the 'cond' block to 'update' arguments so as to support
    side-effecting condition code. In particular, it is possible to thread a
    chain through all 'cond' and 'body' block execution.

    Though this operation is 'nonstrict' on its inputs, execution waits
    for all results of the 'cond' and 'body' blocks to be ready before
    proceeding (either from the 'cond' to the 'body' block, or going around
    the loop again). This ensures we cannot have an unbounded number of
    sub-tasks launched by a while loop, and greatly simplifies asynchronous
    dataflow analysis.

    Example:

    ```mlir
      %res1, %res2 = grt.while (%x: i32, %y: f32) {
        %v1 = "compare"(%x, %y) : (i32, f32) -> i1
        mef.output %x, %y, %v1 : (i32, f32, i1)
      } do {
        %v2, %v3 = "use"(%x, %y) : (i32, f32) -> (i32, f32)
        mef.output %v2, %v3 : (i32, f32)
      }
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$condRegion,
                        SizedRegion<1>:$bodyRegion);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDefinition = [{
    GRT::ControlFlowFlavor WhileOp::getFlavor() { return GRT::kLoopLike; }
    llvm::ArrayRef<mlir::OpOperand> WhileOp::getInputOpOperands() {
      return getOperation()->getOpOperands();
    }
    mlir::Operation::operand_range WhileOp::getStrictExtraOperands() {
      // Ie empty.
      return getOperation()->getOperands().take_front(0);
    }
    llvm::ArrayRef<mlir::OpOperand> WhileOp::getStrictExtraOpOperands() {
      // Ie empty.
      return getOperation()->getOpOperands().take_front(0);
    }
    llvm::SmallVector<llvm::StringRef> WhileOp::getRegionNames() { return {"cond", "body"}; }
  }];

  let extraClassDeclaration = [{
    MEF::OutputOp getCondTerminator();
    MEF::OutputOp getBodyTerminator();
  }];
}

//===----------------------------------------------------------------------===//
// RepeatI32Op
//===----------------------------------------------------------------------===//

def RepeatI32Op : GRTOp<"repeat.i32",
         [IsolatedFromAbove, SingleBlockImplicitTerminator<"MEF::OutputOp">,
          NonStrictInterface, RegionKindInterface, HasOnlyGraphRegion,
          DeclareOpInterfaceMethods<GRTControlFlowInterface>]> {
  let summary = "Repeat a body N times";
  let description = [{
    The "grt.repeat.i32" operation is a repeat loop that executes its 'body'
    block 'tripCount' times, where 'tripCount' is an i32 value specified by the
    first operand.

    The 'body' block has access to the outer SSA values listed in the
    "grt.repeat.i32" signature. The block must return values with the
    same types as the signature, using a "mef.output" operation at the end of
    the block. The result of the "grt.repeat.i32" op is the result of the last
    executed 'body' block, or the "grt.repeat.i32" signature arguments directly
    if 'tripCount' is <= 0.

    Though this operation is 'nonstrict' on its inputs, execution waits
    for all results of the 'body' block to be ready before proceeding with
    the next loop iteration. This ensures we cannot have an unbounded number
    of sub-tasks launched by a repeat loop, and greatly simplifies asynchronous
    dataflow analysis.

    Example:

    ```mlir
      %res1, %res2 = grt.repeat.i32 %i (%loopval1: si32, %loopval2: f32) {
        %v1, v2 = "process"(%loopval1, %loopval2) : (si32, f32) -> (s132, f32)
        mef.output %v1, %v2 : si32, f32
      }
    ```
  }];

  let arguments = (ins SI32:$tripCount, Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDefinition = [{
    GRT::ControlFlowFlavor RepeatI32Op::getFlavor() { return GRT::kLoopLike; }
    llvm::ArrayRef<mlir::OpOperand> RepeatI32Op::getInputOpOperands() {
      return getOperation()->getOpOperands().drop_front();
    }
    mlir::Operation::operand_range RepeatI32Op::getStrictExtraOperands() {
      return getOperation()->getOperands().take_front();
    }
    llvm::ArrayRef<mlir::OpOperand> RepeatI32Op::getStrictExtraOpOperands() {
      return getOperation()->getOpOperands().take_front();
    }
    llvm::SmallVector<llvm::StringRef> RepeatI32Op::getRegionNames() { return {"body"}; }
  }];
}
