//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef HLCFOPS_TD
#define HLCFOPS_TD

include "Support/HLCFDialect/HLCFDialect.td"
include "Support/HLCFDialect/HLCFInterfaces.td"
include "Support/Interpreter/InterpreterInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Operation Base Classes
//===----------------------------------------------------------------------===//

/// Base class for all HLCF operations.
class HLCFOp<string mnemonic, list<Trait> traits = []> :
    Op<HLCFDialect, mnemonic, traits>;

/// Base class for HLCF control-flow operations.
class HLCFParentOp<string mnemonic, list<Trait> traits = []> :
    HLCFOp<mnemonic, traits # [
      RecursiveMemoryEffects, RecursivelySpeculatable,
      DeclareOpInterfaceMethods<ControlFlowNode>,
      DeclareOpInterfaceMethods<InterpreterOpInterface>]>;

/// Base class for HLCF terminator operations.
class HLCFTerminatorOp<string mnemonic, list<Trait> traits = []> :
    HLCFOp<mnemonic, traits #
           [Terminator, DeclareOpInterfaceMethods<ControlFlowTerminator>]> {
  let assemblyFormat = "attr-dict ($operands `:` type($operands)^)?";
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

  let arguments = (ins Variadic<AnyType>:$operands,
                       OptionalAttr<StrAttr>:$label);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region MinSizedRegion<1>:$body);

  let assemblyFormat = [{
    ($label^)? ``
    custom<Loop>($operands, type($operands), type($results), $body) attr-dict
  }];

  let builders = [OpBuilder<(ins), [{ build($_builder, $_state, {}, {}); }]>];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

def HLCF_IfOp : HLCFParentOp<"if"> {
  let summary = "if operation";
  let description = [{
    The `hlcf.if` operation defines two scoped control-flow regions.
    Control-flow transfers from the operation to the beginning of one of the
    regions based on the value of the condition operand. The if operation is the
    target of the `hlcf.yield` terminator.

    This operation forms a condition construct. The immediate regions of the
    operation are condition scopes. There are no nested condition scopes; only
    `hlcf.yield` operations nested immediately under the operation can be used
    to branch to after the operation.

    Example:

    ```mlir
    hlcf.if %arg0 {
      call @do_something()
      hlcf.yield
    } else {
      call @do_something_else()
      hlcf.yield
    }
    ```

    This also means the operation can be used to conditionally execute the
    nearest surrounding loop.

    Example:

    ```mlir
    hlcf.loop {
      hlcf.if %arg0 {
        hlcf.break
      } else {
        hlcf.yield
      }
      hlcf.continue
    }
    ```

    The operation can have results. Each `hlcf.yield` operation nested under
    an `hlcf.if` operation must have the same operand types as the latter's
    result types.

    Example:

    ```mlir
    %0 = hlcf.if %arg0 -> index {
      hlcf.yield %arg1 : index
    } else {
      hlcf.yield %arg2 : index
    }
    ```
  }];

  let arguments = (ins I1:$cond);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region MinSizedRegion<1>:$thenRegion,
                        MinSizedRegion<1>:$elseRegion);

  let assemblyFormat = [{
    $cond (`->` type($results)^)? $thenRegion `else` $elseRegion
    attr-dict
  }];

  let builders = [OpBuilder<(ins "Value":$cond), [{
    build($_builder, $_state, {}, cond);
  }]>];

  let extraClassDeclaration = [{
    // Return OpBuilder's pointing the end of the then/else block.
    OpBuilder getThenBodyBuilder();
    OpBuilder getElseBodyBuilder();
  }];
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

  let builders =
    [OpBuilder<(ins), [{ build($_builder, $_state, ValueRange(), {}); }]>];

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
    The `hlcf.break` opeartion is a control-flow terminator that transfer
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

  let builders =
    [OpBuilder<(ins), [{ build($_builder, $_state, ValueRange(), {}); }]>];

  let assemblyFormat = "attr-dict ($label^)? ($operands `:` type($operands)^)?";
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def HLCF_YieldOp : HLCFTerminatorOp<"yield", [Pure, HasParent<"IfOp">,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "conditional yield operation";
  let description = [{
    The `hlcf.yield` operation is a control-flow terminator that transfers
    control-flow to after the parent `hlcf.if` operation. The operation expects
    its operand types to match the types of the results of that operation, if
    there are any.

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
// ReturnOp
//===----------------------------------------------------------------------===//

def HLCF_ReturnOp : HLCFTerminatorOp<"return", [ReturnLike]> {
  let summary = "function return operation";
  let description = [{
    The `hlcf.return` operation is a control-flow operation that transfers
    control-flow to the end of the surrounding function operation. The function
    control-flow tree is a special tree rooted at an operation that implements
    `FunctionOpInterface`. The operand types of the operation must match the
    return types of the enclosing function.

    Example:

    ```mlir
    func.func @pass_argument(%arg0: index, %arg1: i1) -> index {
      hlcf.if %arg1 {
        hlcf.return %arg0 : index
      } else {
        hlcf.yield
      }
      call @do_something() : () -> ()
      return %arg0 : index
    }
    ```

    This operation is modelled to always have unknown side effects because the
    control-flow edge from this operation will always escape the parent
    operation; it can never be nested directly below functions.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<(ins), [{ build($_builder, $_state, {}); }]>];

  let hasVerifier = 1;
}

#endif // HLCFOPS_TD
