//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef DEBUGINFOOPS_TD
#define DEBUGINFOOPS_TD

include "Support/DebugInfoDialect/IR/DebugInfoAttrs.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// DebugInfoOp
//===----------------------------------------------------------------------===//

/// Base class for DebugInfo dialect operations.
class DebugInfoOp<string mnemonic, list<Trait> traits = []>
    : Op<DebugInfoDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// ValueOp
//===----------------------------------------------------------------------===//

def DebugInfo_ValueOp : DebugInfoOp<"value"> {
  let summary = "debug variable value definition";
  let description = [{
    The `debuginfo.value` operation provides information when a debug info
    variable is set to a new value. Note that this operation decribes the value
    of a variable directly, not its address.

    Example:

    ```mlir
    %0 = constant 0 : i32
    debuginfo.value #debuginfo.local_variable<...> = %0 : i32
    ```
  }];

  let arguments = (ins AnyType:$value, DebugInfo_DILocalVariableAttr:$valueInfo);
  let assemblyFormat = [{
    qualified($valueInfo) `=` $value `:` type($value) attr-dict
  }];
  let hasVerifier = 1;
}

#endif // DEBUGINFOOPS_TD
