//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#ifndef HLCFOPS_TD
#define HLCFOPS_TD

include "Mojo/HLCFDialect/HLCFAttrs.td"
include "Mojo/HLCFDialect/HLCFDialect.td"
include "Mojo/HLCFDialect/HLCFInterfaces.td"
include "Mojo/Interpreter/InterpreterInterface.td"
include "Mojo/KGENDialect/KGENAttrs.td"
include "Mojo/KGENDialect/KGENInterfaces.td"
include "Mojo/KGENDialect/KGENTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Operation Base Classes
//===----------------------------------------------------------------------===//

/// Base class for all HLCF operations.
class HLCFOp<string mnemonic, list<Trait> traits = []> :
    Op<HLCFDialect, mnemonic, traits>;

/// Base class for HLCF control-flow operations.
class HLCFParentOp<string mnemonic, list<Trait> traits = []> :
    HLCFOp<mnemonic, traits # [
      DeclareOpInterfaceMethods<ControlFlowNode>,
      DeclareOpInterfaceMethods<InterpreterOpInterface>]>;

/// Base class for HLCF terminator operations.
class HLCFTerminatorOp<string mnemonic, list<Trait> traits = []> :
    HLCFOp<mnemonic, traits #
           [Terminator, DeclareOpInterfaceMethods<ControlFlowTerminator>]> {
  let assemblyFormat = "attr-dict ($operands `:` type($operands)^)?";
}

//===----------------------------------------------------------------------===//
// ForOp / ForControlVariableType constraint
//===----------------------------------------------------------------------===//

/// Match `!kgen.scalar<si8|si16|si32|si64|index>` or `index` type.
/// Used to type-constrain hlcf.for bounds, step and the induction variable in
/// `hlcf.for.yield`.
def ForControlVariableType : AnyTypeOf<[
    Index,
    ConfinedType<AnyType,
        [CPred<"llvm::isa<::M::KGEN::SIMDType>($_self) && "
               "llvm::cast<::M::KGEN::SIMDType>($_self).isScalar() && "
               "llvm::cast<::M::KGEN::SIMDType>($_self).getResolvedDType().has_value() && "
               "llvm::cast<::M::KGEN::SIMDType>($_self).getResolvedDType()->isSInt()">],
        "!kgen.scalar<signed integer> type">
], "index or !kgen.scalar<signed integer> type">;

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

def HLCF_ForOp : HLCFParentOp<"for"> {
  let summary = "for operation";
  let description = [{
    The `hlcf.for` operation defines a control-flow region taking 3
    operands that represent the lower bound, upper bound and step respectively.
    It has one region capturing the loop body.

    The lower and upper bounds specify a half-open range
    when the `cmpPredicateType` is exclusive (slt, sgt):
    the range includes the lower bound but does not include the upper bound.

    The lower and upper bounds specify a full close range
    when the `cmpPredicateType` is inclusive (sle, sge):
    the range includes the lower and the upper bounds.

    Note that: lower bound is the initial value of the induction variable;
    Upper bound is the iteration end of the induction variable.
    Lower bound's absolute value can be larger than upper bound if
    step value is negative
    (positive) and the iteration update the induction variable by
    adding(subtracting) the step value.

    We also use the `cmpPredicateType` to reconstruct loop from for-loop,
    the predicate constructs the condition for break the loop.
    The induction variable is always on the left side of the comparison.

    `indVarComputeType` (add or sub) specifies how induction variable
    progresses over iterations with `step`. We will use this information
    to reconstruct a general loop lowered from for-loop.

    The body region must contain exactly one block that terminates with
    `scf.for.continue`.

    The operation can define loop-carried values and operation results. The
    operand types of `hlcf.for.yield` operations in the immediate loop scope
    must match the initArgs types of the operation. The operand types of
    `hlcf.break` in the immediate loop scope must match the result types of the
    operation.

    Example:

    ```mlir
     hlcf.for [%lb to %up step %step slt add] (%arg0 = %c1: index) {
       %5 = index.add %arg0, %c2
       hlcf.for.yield[induction_var (%5 : index)] [retvals ()] [iterargs ()]
     } {unrollLevel = #hlcf<unroll_level full>}
    ```

    or

    ```mlir
    %0 = hlcf.for [%lb to %up step %step sge sub]
                  (%arg1 = %k: index, %arg0 = %i: index) -> (index, index) {
      %3 = index.add %arg1, %index2
      %5 = index.add %arg0, %arg1
      hlcf.for.yield [induction_var (%3 : index)] [retvals (%5: index)]
                     [iterargs ()]
    } {unrollLevel = #hlcf<unroll_level full>}
    ```

   The arg operands are ordered in three parts:
   1. Induction variable (one value). (%arg1 = %k: index) in the example above.
   2. Return value args. (%arg0 = %i: index) in the example above.
   3. Other loop carried values. Zero value in the example above.

  }];

  let arguments = (ins
    ForControlVariableType:$lowerBound,
    ForControlVariableType:$upperBound,
    ForControlVariableType:$step,
    Variadic<AnyType>:$iterArgs,
    DefaultValuedAttr<HLCF_UnrollLevelAttr, "UnrollLevel::none()">:$unrollLevel,
    HLCF_ForLoopBoundCmpPredicateAttr:$cmpPredicateType,
    HLCF_ForLoopIndVarComputeAttr:$indVarComputeType
  );

  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    ` ``[` custom<ForBoundsWithOptionalType>($lowerBound, type($lowerBound),
                                            $upperBound, type($upperBound),
                                            $step, type($step))
    enum($cmpPredicateType) enum($indVarComputeType) `]` ``
    custom<Loop>($iterArgs, type($iterArgs), type($results), $body) attr-dict
  }];

  let extraClassDeclaration = [{
    /// Get integer value for lowerBound.
    std::optional<int64_t> getLowerBoundAsInt();
    /// Get integer value for upperBound.
    std::optional<int64_t> getUpperBoundAsInt();
    /// Get integer value for step.
    std::optional<int64_t> getStepAsInt();
    /// Get args that will be return values.
    ValueRange getReturnValueArgs();
    /// Get loop trip count.
    std::optional<int64_t> getTripCount();
    /// Check if the loop is decorated to be fully unrolled.
    bool isFullUnroll();
    /// Get unroll factor if the loop is decorated.
    std::optional<int64_t> getUnrollFactorN();
    /// Insert variants as operation operands.
    void insertVariants(ValueRange newOperands);
    /// Insert argument to region.
    BlockArgument insertArgumentToRegion(Location loc,
      Type argType, size_t argIdx, Region &region);
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// LoopOp
//===----------------------------------------------------------------------===//

def HLCF_LoopOp : HLCFParentOp<"loop"> {
  let summary = "loop operation";
  let description = [{
    The `hlcf.loop` operation defines a single scoped control-flow region.
    Control-flow transfers from the operation to the beginning of the body
    region. The loop operation is the target of the `hlcf.continue` and
    `hlcf.break` terminators.

    As the name suggests, the operation forms a loop construct. Each instance
    of the operation defines a new loop scope, where `hlcf.continue` operations
    in the immediate scope branch to the beginning of the body region and
    `hlcf.break` operations branch to after the operation.

    Example:

    ```mlir
    hlcf.loop {
      cf.cond_br %arg0, ^continue, ^break

    ^continue:
      hlcf.continue

    ^break:
      hlcf.break
    }
    ```

    The operation can define loop-carried values and operation results. The
    operand types of `hlcf.continue` operations in the immediate loop scope
    must match the operand types of the operation. The operand types of
    `hlcf.break` in the immediate loop scope must match the result types of the
    operation.

    Example:

    ```mlir
    %0:2 = hlcf.loop (%lc0 = %arg0 : index) -> (i32, i64) {
      %1 = index.add %lc0, %c1
      %2 = index.cmp le(%1, %c10)
      hlcf.if %2 {
        hlcf.continue %1 : index
      } else {
        hlcf.yield
      }
      %3 = index.casts %lc0 : index to i32
      %4 = index.casts %lc0 : index to i64
      hlcf.break %3, %4 : i32, i64
    }
    ```

    The loop may optionally take a "label", with which break and continue
    operations explicitly target to break or continue from out of inner loops.
  }];

  let arguments = (ins
    Variadic<AnyType>:$operands,
    OptionalAttr<StrAttr>:$label,
    OptionalAttr<AnyAttrOf<[HLCF_UnrollLevelAttr,
                            TypedAttrInterface]>>:$unrollLevel
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    ($label^)? ``
    custom<Loop>($operands, type($operands), type($results), $body) attr-dict
  }];

  let builders = [
    OpBuilder<(ins CArg<"Attribute", "{}">:$level), [{
      build($_builder, $_state, {}, {}, {}, level);
    }]>,
    OpBuilder<(ins "StringAttr":$label), [{
      build($_builder, $_state, {}, {}, label, {});
    }]>,
    OpBuilder<(ins "TypeRange":$results, "ValueRange":$operands,
                   "StringAttr":$label), [{
      build($_builder, $_state, results, operands, label, {});
    }]>,
    OpBuilder<(ins "TypeRange":$results, CArg<"StringAttr", "{}">:$label), [{
      build($_builder, $_state, results, {}, label, {});
    }]>
  ];

  let extraClassDeclaration = [{
    Block &getBodyBlock() { return getBody().front(); }

    /// Check if the loop is decorated to be fully unrolled.
    bool isFullUnroll();

    /// Get unroll factor if the loop is decorated with an constant.
    std::optional<int64_t> getUnrollFactorN();

    /// Get unroll level as a concrete HCLF::UnrollLevel value.
    HLCF::UnrollLevel getUnrollLevelValue();
  }];

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// SwitchOp
//===----------------------------------------------------------------------===//

def HLCF_SwitchOp : HLCFParentOp<"switch",
    [DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "switch operation";
  let description = [{
    The `hlcf.switch` operation accepts an index-valued argument and branches to
    one of its regions based on its value. If there is no matching corresponding
    region, the default branch is taken.

    Example:

    ```mlir
    hlcf.switch %s
    default {
      hlcf.yield
    }
    case 0 {
      hlcf.break
    }
    case 42 {
      hlcf.return
    }
    ```

    The operation can optionally return variadic results.

    Example:

    ```mlir
    hlcf.switch %s -> (i32, i64)
    default {
      hlcf.yield %arg0, %arg1 : i32, i64
    }
    ```
  }];

  let arguments = (ins Index:$arg, DenseI32ArrayAttr:$caseValues);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$defaultRegion,
                        VariadicRegion<SizedRegion<1>>:$caseRegions);

  let assemblyFormat = [{
    $arg (`->` type($results)^)? `\n`
    `` `default` $defaultRegion custom<SwitchCases>($caseValues, $caseRegions)
    attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ContinueOp
//===----------------------------------------------------------------------===//

def HLCF_ContinueOp : HLCFTerminatorOp<"continue",
    [DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "continue loop execution";
  let description = [{
    The `hlcf.continue` operation is a control-flow terminator that transfers
    control-flow to the beginning of the body of the nearest surrounding loop
    operation. The operation expects its operand types to match the types of the
    loop-carried values of that loop, if there are any.

    Example:

    ```mlir
    hlcf.loop (%lc0 = %arg0 : index) {
      hlcf.continue %lc0 : index
    }
    ```

    This operation is never pure because the control-flow edge from this
    terminator will always lead to an earlier point in the IR.

    The operation can take an optional label, in which case it will continue
    from the beginning of the labelled loop. A continue without a label will
    branch to the start of the nearest enclosing loop, regardless of its label.
  }];

  let arguments = (ins Variadic<AnyType>:$operands,
                       OptionalAttr<StrAttr>:$label);

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, ValueRange(), {}); }]>,
    OpBuilder<(ins "ValueRange":$operands), [{
      build($_builder, $_state, operands, {});
    }]>,
  ];

  let assemblyFormat = "attr-dict ($label^)? ($operands `:` type($operands)^)?";
}

//===----------------------------------------------------------------------===//
// BreakOp
//===----------------------------------------------------------------------===//

def HLCF_BreakOp : HLCFTerminatorOp<"break", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<ConditionallySpeculatable>,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "loop exit operation";
  let description = [{
    The `hlcf.break` operation is a control-flow terminator that transfer
    control-flow to after the nearest surrounding loop operation. The operation
    expects its operand types to match the types of the results of that loop, if
    there are any.

    Example:

    ```mlir
    %0 = hlcf.loop () -> index {
      hlcf.break %arg0 : index
    }
    ```

    This operation is conditionally pure. It is always known to be pure if it
    is nested directly beneath a loop operation because the control-flow edge
    from this terminator will always go to after the parent operation.

    The operation can take an optional label, in which case it will branch to
    after the end of the labelled loop. A break without a label will break out
    of the nearest enclosing loop, regardless of its label.
  }];

  let arguments = (ins Variadic<AnyType>:$operands,
                       OptionalAttr<StrAttr>:$label);

  let builders = [
    OpBuilder<(ins CArg<"StringAttr", "{}">:$label), [{
      build($_builder, $_state, ValueRange(), label);
    }]>,
    OpBuilder<(ins "ValueRange":$operands), [{
      build($_builder, $_state, operands, {});
    }]>,
  ];

  let assemblyFormat = "attr-dict ($label^)? ($operands `:` type($operands)^)?";
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def HLCF_YieldOp : HLCFTerminatorOp<"yield", [Pure,
    ParentOneOf<["SwitchOp", "IfOp", "MatchOp"]>,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "conditional yield operation";
  let description = [{
    The `hlcf.yield` operation is a control-flow terminator that transfers
    control-flow to after the parent `hlcf.if`, `hlcf.switch`, or
    `hlcf.match` operation. For match ops, yield is
    only valid in the else region (case regions use `hlcf.match.next` /
    `hlcf.match.complete`). The operation expects its operand types to match
    the types of the results of that operation, if there are any.

    Example:

    ```mlir
    %0 = hlcf.if %arg0 -> index {
      hlcf.yield %arg1 : index
    } else {
      hlcf.yield %arg2 : index
    }
    ```

    This operation can always be marked pure because the control-flow edge from
    this operation leads only to after the parent operation.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<(ins), [{ build($_builder, $_state, {}); }]>];
}

//===----------------------------------------------------------------------===//
// ForYieldOp
//===----------------------------------------------------------------------===//

def HLCF_ForYieldOp : HLCFTerminatorOp<"for.yield", [
    ParentOneOf<["ForOp"]>, AttrSizedOperandSegments]> {
  let summary = "yield operation in a for-loop";
  let description = [{
    The `hlcf.for.yield` operation is a control-flow terminator that transfers
    control-flow either back to the beginning or after the parent `hlcf.for`
    operation, depending on whether the for-loop continues or not.
    The operation expects its operand types to match the types of the
    `initArgs` of `hlcf.for`.

    There are three segments of operands:
    1. Induction variable. There should always be one value.
    2. Return value args. 0 - n values that return as results of the parent
       for-loop when the induction variable exceeds the upperbound.
    3. Other loop carried values. 0 - n values.

    Example:

    ```mlir
    hlcf.for [%lb to %up step %step] (%arg0 = %c1: index) {
      %5 = index.add %arg0, %c2
      hlcf.for.yield[induction_var (%5 : index)] [retvals ()] [iterargs ()]
    } {unrollLevel = #hlcf<unroll_level full>}
    ```
  }];

  let extraClassDeclaration = [{
    /// Insert variants as operation operands.
    void insertVariants(ValueRange newOperands);
  }];

  let arguments = (ins ForControlVariableType:$inductionVar,
                       Variadic<AnyType>:$returnValues,
                       Variadic<AnyType>:$otherIterValues);

  let assemblyFormat =[{
    ` ``[``induction_var`` ``(`$inductionVar `:` type($inductionVar)`)``]`
    ` ``[``retvals`` ``(`($returnValues `:` type($returnValues)^)?`)``]`
    `[``iterargs`` ``(`($otherIterValues `:` type($otherIterValues)^)?`)``]`
    attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// IfElifCondYieldOp
//===----------------------------------------------------------------------===//

def HLCF_IfElifCondYieldOp : HLCFTerminatorOp<"if.elifcond.yield", [
    ParentOneOf<["IfOp"]>,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "yield operation for additional conditions of if operations";
  let description = [{
    The `hlcf.if.elifcond.yield` operation is a control-flow terminator used only in
    the *additional* condition regions of a parent `hlcf.if` (the regions in
    `$elifRegions`). It transfers control either to the paired `then` region
    when the condition is true, or to the next condition region / the `else`
    region when false.

    This op takes at least one `!kgen.scalar<bool>` value which is the
    condition, but can take an arbitrary number of other values. These values
    are passed into the "then" or subsequent/"else" block, enabling mem2reg
    promotion of values stored in the condition block.

    Example:

    ```mlir
    %0 = hlcf.if %cond0 -> index {
      hlcf.yield %arg0 : index
    } else {
      hlcf.if.elifcond.yield %cond1, %x : i32
    } then (%xThen: i32) {
      hlcf.yield %arg0 : index
    } else (%xElse: i32) {
      hlcf.yield %arg0 : index
    }
    ```
  }];
  let arguments = (ins BoolScalar:$cond, Variadic<AnyType>:$values);

  let assemblyFormat = "$cond (`,` $values `:` type($values)^)? attr-dict";
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

def HLCF_IfOp : HLCFParentOp<"if"> {
  let summary = "if operation";
  let description = [{
    The `hlcf.if` operation is a multi-arm conditional. The first condition
    is an SSA operand; its matching `then` is `$thenRegion`. Additional arms
    are written as `else { cond } then { then }` pairs in `$elifRegions`, where
    each condition region is terminated by `hlcf.if.elifcond.yield`. The final
    `$elseRegion` runs when every condition is false.

    All `then` / `else` regions are terminated by `hlcf.yield`, whose operand
    types must match the results of the parent `hlcf.if`.

    Example (single if/else):

    ```mlir
    %0 = hlcf.if %cond -> index {
      hlcf.yield %arg0 : index
    } else {
      hlcf.yield %arg1 : index
    }
    ```

    Example (if / elif / else):

    ```mlir
    %0 = hlcf.if %cond0 -> index {
      hlcf.yield %arg0 : index
    } else {
      hlcf.if.elifcond.yield %cond1
    } then {
      hlcf.yield %arg1 : index
    } else {
      hlcf.yield %arg2 : index
    }
    ```
  }];
  let arguments = (ins BoolScalar:$cond);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$thenRegion,
                        SizedRegion<1>:$elseRegion,
                        VariadicRegion<SizedRegion<1>>:$elifRegions);

  let assemblyFormat = [{
    $cond (`->` type($results)^)?
    custom<If>($thenRegion, $elifRegions, $elseRegion) attr-dict
  }];

  let builders = [
    OpBuilder<(ins "Value":$cond), [{
      build($_builder, $_state, TypeRange(), cond, /*nElifRegions=*/0);
    }]>,
    OpBuilder<(ins "TypeRange":$results, "Value":$cond), [{
      build($_builder, $_state, results, cond, /*nElifRegions=*/0);
    }]>,
  ];

  let extraClassDeclaration = [{
    /// Create a 2-arm if, run `emitThen` / `emitElse` at the start of each
    /// region (they must terminate the blocks), and leave the insertion point
    /// after the op. Returns null if either callback fails.
    static IfOp create(OpBuilder &builder, Location loc,
                       TypeRange resultTypes, Value cond,
                       function_ref<LogicalResult()> emitThen,
                       function_ref<LogicalResult()> emitElse);

    /// Return OpBuilder's pointing the end of the then/else block.
    OpBuilder getThenBodyBuilder();
    OpBuilder getElseBodyBuilder();

    /// Get 'then' block.
    Block& getThenBlock();
    /// Get 'else' block.
    Block& getElseBlock();
    /// Get the terminator op from 'then' block.
    Operation* getThenTerminator();
    /// Get the terminator op from 'else' block.
    Operation* getElseTerminator();
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// MatchNextOp
//===----------------------------------------------------------------------===//

def HLCF_MatchNextOp : HLCFTerminatorOp<"match.next", [
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "advance to the next match case";
  let description = [{
    The `hlcf.match.next` terminator indicates that the current case of the
    nearest enclosing `hlcf.match` did not match (or
    its guard failed). It abandons the current case region and begins executing
    the next case region. If executed from the final case region, control
    transfers to the `else` region of that match.

    This terminator may appear nested under structured control flow such as
    `hlcf.if`, and always targets the nearest enclosing match for which the
    terminator sits in a *case* region. The `else` region of a match is
    outside that match's scope: `hlcf.match.next` inside an `else` targets an
    enclosing match, not the current one.

    Optional operands forward SSA values into the successor region's block
    arguments (used by mem-2-reg when promoting allocations across cases).

    Values and partially initialized bindings local to the abandoned case must
    be cleaned up according to their normal ownership semantics.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, ValueRange()); }]>,
  ];

  let assemblyFormat = "attr-dict ($operands `:` type($operands)^)?";
}

//===----------------------------------------------------------------------===//
// MatchCompleteOp
//===----------------------------------------------------------------------===//

def HLCF_MatchCompleteOp : HLCFTerminatorOp<"match.complete", [
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "finish a match after selecting a case";
  let description = [{
    The `hlcf.match.complete` terminator indicates that the current case of the
    nearest enclosing `hlcf.match` was selected and its
    body completed normally. It exits that match and transfers control to the
    operation following it. Operand types must match the result types of the
    enclosing match, if any.

    Case bodies that leave through another nonlocal control-flow operation
    (such as `hlcf.return`) do not need an additional `hlcf.match.complete`.

    Like `hlcf.match.next`, this targets the nearest enclosing match for which
    the terminator sits in a *case* region; an occurrence in an `else` region
    targets an enclosing match instead.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, ValueRange()); }]>,
  ];

  let assemblyFormat = "attr-dict ($operands `:` type($operands)^)?";
}

//===----------------------------------------------------------------------===//
// MatchOp
//===----------------------------------------------------------------------===//

def HLCF_MatchOp : HLCFParentOp<"match"> {
  let summary = "match operation";
  let description = [{
    The `hlcf.match` operation represents the structured control flow of a
    match statement. It contains an ordered list of single-block case regions
    followed by an else region. Each case region holds the tests, binding
    initialization, optional guard, and body for its corresponding source-level
    case. The operation itself does not interpret patterns; pattern matching is
    lowered into ordinary operations within each region.

    Execution begins in the first case region. A region may terminate with
    `hlcf.match.next` to continue with the next case, or with
    `hlcf.match.complete` to finish the match without evaluating later cases.
    These terminators may occur within nested structured control-flow
    operations (such as `hlcf.if`) and target the nearest enclosing
    `hlcf.match`. If `hlcf.match.next` is executed from the final case region,
    the match enters the else region.

    Case regions are logically in the scope of the match; the else region is
    not. Therefore `hlcf.match.next` / `hlcf.match.complete` inside the else
    region target an enclosing match, not the current one. The else region is
    terminated with `hlcf.yield` (or another nonlocal terminator).

    Example:

    ```mlir
    hlcf.match {
      %matches = ...
      hlcf.if %matches {
        call @handle_zero()
        hlcf.match.complete
      } else {
        hlcf.match.next
      }
      // Case regions are ordinary blocks and need a terminator even when
      // every nested path already exits via match.next / match.complete.
      hlcf.unreachable
    }
    case {
      call @handle_other()
      hlcf.match.complete
    }
    else {
      hlcf.unreachable
    }
    ```
  }];

  let results = (outs Variadic<AnyType>:$results);
  // `else` must come before the variadic case list (ODS constraint); the
  // custom assembly format still prints cases first, then `else`.
  let regions = (region SizedRegion<1>:$elseRegion,
                        VariadicRegion<SizedRegion<1>>:$caseRegions);

  let assemblyFormat = [{
    (`->` type($results)^)?
    custom<Match>($caseRegions, $elseRegion) attr-dict
  }];

  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, TypeRange(), /*nCaseRegions=*/1);
    }]>,
    OpBuilder<(ins "TypeRange":$results), [{
      build($_builder, $_state, results, /*nCaseRegions=*/1);
    }]>,
  ];

  let extraClassDeclaration = [{
    /// Return true if `op` is nested under a case region of this match
    /// (not the else region).
    bool containsInCaseRegion(Operation *op);

    /// Return the case-region index (0-based into `$caseRegions`) containing
    /// `op`, or nullopt if `op` is not nested under a case of this match.
    std::optional<unsigned> getCaseRegionIndexContaining(Operation *op);
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ComptimeForOp
//===----------------------------------------------------------------------===//

def HLCF_ComptimeForOp : HLCFParentOp<"comptime.for", [
    DeclInterface,
    DeclareOpInterfaceMethods<ParamOpInterface, ["isImplicitlyParametric",
      "collectParameterUsesBelow"]>]>, NoParamDecls {
  let summary = "compile-time for loop";
  let description = [{
    The `hlcf.comptime.for` operation instantiates its body with values according
    to its iterator. It takes an initial iterator value, a 'hasNext' function
    that take an iterator and indicates whether more elements exist, and a
    'getNextIter' function that takes an iterator instance and returns the next
    iterator value.

    This operation can have loop-carried values - the "operands" inputs and
    results, which are values promoted within the loop by mem2reg.
  }];

  let arguments = (ins
    TypedAttrInterface:$initial,
    AttrOfType<KGEN_FuncTypeGeneratorType>:$hasNext,
    AttrOfType<KGEN_FuncTypeGeneratorType>:$getNextIter,
    KGEN_ParamDeclAttr:$paramDecl,
    Variadic<AnyType>:$operands,
    UnitAttr:$bodyIsolated,
    UnitAttr:$elseIsolated
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body, SizedRegion<1>:$elseRegion);

  let builders = [
    OpBuilder<(ins "TypedAttr":$initial, "TypedAttr":$hasNext,
                   "TypedAttr":$getNextIter, "ParamDeclAttr":$paramDecl), [{
      build($_builder, $_state, {}, initial, hasNext, getNextIter, paramDecl,
            {}, {}, {});
    }]>
  ];

  let assemblyFormat = [{
    custom<ParamDecl>($paramDecl) `in` custom<ColonTypeParamValue>($initial)`\n`
    `has_next` custom<ColonTypeParamValue>($hasNext) `\n`
    `get_next_iter` custom<ColonTypeParamValue>($getNextIter)
    custom<Loop>($operands, type($operands), type($results), $body)
    `else` custom<RegionWithArgs>($elseRegion) attr-dict
  }];

  let extraClassDeclaration = [{
    bool isIsolatedFromAbove(unsigned regionNum);
    void notifyKnownIsolatedFromAbove(unsigned regionNum);
    ArrayRef<KGEN::ParamDeclAttr> getInputParams();
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ComptimeForBreakOp
//===----------------------------------------------------------------------===//

def HLCF_ComptimeForBreakOp : HLCFTerminatorOp<"comptime.for.break", [
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "param loop exit operation";
  let description = [{
    The `hlcf.comptime.for.break` operation represents an exit from a
    `hlcf.comptime.for`, branching to the end of all generated iterations.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, ValueRange()); }]>,
  ];
}

//===----------------------------------------------------------------------===//
// ComptimeForContinueOp
//===----------------------------------------------------------------------===//

def HLCF_ComptimeForContinueOp : HLCFTerminatorOp<"comptime.for.continue", [
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "param loop continue operation";
  let description = [{
    The `hlcf.comptime.for.continue` operation branches to the next generated
    iteration of the loop.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, ValueRange()); }]>,
  ];
}

//===----------------------------------------------------------------------===//
// ComptimeForGotoElseOp
//===----------------------------------------------------------------------===//

def HLCF_ComptimeForGotoElseOp : HLCFTerminatorOp<"comptime.for.goto.else"> {
  let summary = "param loop 'goto else' operation";
  let description = [{
    The `hlcf.comptime.for.goto.else` operation jumps to the 'else' block of a
    `hlcf.comptime.for`.  It only exists to help the interface between the Mojo
    parser and the LowerSemanticCF pass work more easily.
  }];

  let arguments = (ins);

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// ComptimeIfOp
//===----------------------------------------------------------------------===//

def HLCF_ComptimeIfOp : HLCFParentOp<"comptime.if", [
    DeclInterface,
    DeclareOpInterfaceMethods<ParamOpInterface, ["isImplicitlyParametric",
      "collectParameterUsesBelow"]>]>, NoParamDecls {
  let summary = "compile-time constant if";
  let description = [{
    The `hlcf.comptime.if` op provides a compile-time guarantee that if `cond`
    is false, only the `else` block will be elaborated, and if `cond` is true,
    only the `then` block will be elaborated. Whichever block is elaborated
    will be directly inlined into the scope. This op is fully removed
    during elaboration.

    The result parameter will be resolved by the parameter on the
    `comptime.yield` in the branch that is actually elaborated.

    ```mlir
    %0 = hlcf.comptime.if <condition> -> index {
      %i0 = index.constant 0
      hlcf.comptime.yield %i0
    } else {
      %i1 = index.constant 1
      hlcf.comptime.yield %i1
    }

    hlcf.comptime.if <condition -> result> {
      hlcf.comptime.yield<1>
    } else {
      hlcf.comptime.yield<0>
    }
    %0 = kgen.param.constant = <result> // == <condition ? 1 : 0>
    ```
  }];

  let arguments = (ins
    AttrOfType<BoolScalar>:$cond,
    UnitAttr:$thenIsolated,
    UnitAttr:$elseIsolated
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$thenRegion, SizedRegion<1>:$elseRegion);

  let extraClassDeclaration = [{
    bool isIsolatedFromAbove(unsigned regionNum);
    void notifyKnownIsolatedFromAbove(unsigned regionNum);

    Block::OpListType &getThenOps() {
      return getThenRegion().front().getOperations();
    }
    Block::OpListType &getElseOps() {
      return getElseRegion().front().getOperations();
    }

    /// Parametric ifs have no parameters.
    ArrayRef<KGEN::ParamDeclAttr> getInputParams() { return {}; }
  }];

  let assemblyFormat = [{
    ` ` `<` custom<ScalarBoolParamValue>($cond) `` `>`
    (`->` type($results)^)?  $thenRegion `else` $elseRegion attr-dict
  }];

  let builders = [OpBuilder<(ins "TypedAttr":$cond), [{
    build($_builder, $_state, {}, cond, {});
  }]>];

  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// ComptimeYieldOp
//===----------------------------------------------------------------------===//

def HLCF_ComptimeYieldOp : HLCFTerminatorOp<"comptime.yield", [
    Pure,
    ParentOneOf<["ComptimeIfOp", "ComptimeForOp"]>,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "conditional yield operation";
  let description = [{
    The `hlcf.comptime.yield` operation is used to denote the terminator of a
    block in the `hlcf.comptime.if` op. Conceptually, it branches to the next
    op after a `hlcf.comptime.if`, but in reality it's to make sure the IR
    stays in a reasonable state and is possible to inline directly.

    This op defines the parameter that is used as the result parameter on the
    enclosing `hlcf.comptime.if`.

    Example:

    ```mlir
    %0 = hlcf.comptime.if <condition> -> index {
      hlcf.comptime.yield %arg1 : index
    } else {
      hlcf.comptime.yield %arg2 : index
    }
    ```

    This operation can always be marked pure because the control-flow edge from
    this operation leads only to after the parent operation.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), [{ return build($_builder, $_state, {}); }]>
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}


//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def HLCF_ReturnOp : HLCFTerminatorOp<"return", [ReturnLike,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "return operation for kgen declarations";
  let description = [{
    The `hlcf.return` operation specifies the returned values for a `kgen.func`
    or `kgen.generator`.

    The operation takes variable number of operands and produces no results.
    The operand number and types must match the signature of the
    `kgen.generator` (or enclosing region) that contains the operation. For
    example:

    ```mlir
      kgen.generator @foo() : i32, f8 {
        ...
        hlcf.return %0, %1 : i32, f8
      }
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, {}); }]>
  ];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// UnreachableOp
//===----------------------------------------------------------------------===//

def HLCF_UnreachableOp : HLCFTerminatorOp<"unreachable"> {
  let summary = "Unreachable operation";
  let description = [{
    The `hlcf.unreachable` operation is a block terminator that indicates that
    the previous operations cannot continue control flow.  This is used after
    infinite loops and no-return functions to simplify dataflow analysis.

    Example:

    ```mlir
    hlcf.loop () {
      kgen.call @printHello()
      hlcf.continue
    }
    hlcf.unreachable
    ```
  }];

  let arguments =
   (ins DefaultValuedOptionalAttr<BoolAttr, "false">:$isAfterUnreachableCall);

  let assemblyFormat = "attr-dict";
}

#endif // HLCFOPS_TD
