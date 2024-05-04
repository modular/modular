//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef COOPS_TD
#define COOPS_TD

include "KGEN/CODialect/CODialect.td"
include "KGEN/CODialect/COTypes.td"
include "Support/MDialect/MAttrs.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

class COOp<string mnemonic, list<Trait> traits = []> :
    Op<CODialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// CoroutineHandleOp
//===----------------------------------------------------------------------===//

def CO_CoroutineHandleOp : COOp<"handle"> {
  let summary = "Get the handle to the current coroutine.";
  let description = [{
    The `co.handle` operation returns the handle to the current coroutine.

    Example:

    ```mlir
    %hdl = co.handle : i32, index
    ```
  }];

  let arguments = (ins M_TypeArrayAttr:$types);
  let results = (outs CoroutineType:$coroutine);
  let assemblyFormat = "attr-dict custom<CoroutineTypes>($types)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CoroutineSuspendOp
//===----------------------------------------------------------------------===//

def CO_CoroutineSuspendOp : COOp<"suspend", [SingleBlock,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>]> {
  let summary = "Suspend the current coroutine and await resumption.";
  let description = [{
    The `co.suspend` operation suspends the current coroutine function
    at the site of the operation and executes a region. The code in the region
    must resume the coroutine at some point. This operation can only be used
    inside an async function.

    Example:

    ```mlir
    co.suspend -> %hdl {
      co.resume %hdl
      co.suspend.end
    }
    ```
  }];

  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "`->` custom<SuspendBody>($body) attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CoroutineSuspendEndOp
//===----------------------------------------------------------------------===//

def CO_CoroutineSuspendEndOp : COOp<"suspend.end", [Terminator,
    HasParent<"CoroutineSuspendOp">]> {
  let summary = "terminator for `co.suspend`";
  let description = [{
    The `co.suspend.end` operation is the terminator for the
    `co.suspend` operation. The operation signifies the end of the
    coroutine suspension function denoted by the surrounding
    `co.suspend` operation. In an async function, it is semantically
    a fallthrough.
  }];

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// CoroutinePromiseOp
//===----------------------------------------------------------------------===//

def CO_CoroutinePromiseOp : COOp<"promise"> {
  let summary = "Get the coroutine promise.";
  let description = [{
    The `co.promise` operation returns the pointer to the coroutine
    promise, which contains the result values of the promise. Located at the
    end of the promise is the async context.

    Example:

    ```mlir
    %promise = co.promise %hdl : struct<i32, i64>
    ```
  }];

  let arguments = (ins CoroutineType:$coroutine);
  let results = (outs PointerType:$promise);
  let assemblyFormat = "$coroutine attr-dict `:` type($promise)";
}

//===----------------------------------------------------------------------===//
// CoroutineResumeOp
//===----------------------------------------------------------------------===//

def CO_CoroutineResumeOp : COOp<"resume"> {
  let summary = "Resume execution of a coroutine.";
  let description = [{
    The `co.resume` operation resumes execution of a suspended coroutine.
    Resuming a coroutine that is not suspended or that is uninitialized is
    undefined behaviour.

    Example:

    ```mlir
    co.resume %hdl
    ```
  }];

  let arguments = (ins CoroutineType:$coroutine);
  let assemblyFormat = "$coroutine attr-dict";
}

//===----------------------------------------------------------------------===//
// CoroutineDestroyOp
//===----------------------------------------------------------------------===//

def CO_CoroutineDestroyOp : COOp<"destroy"> {
  let summary = "Destroy the memory associated with a coroutine.";
  let description = [{
    The `co.destroy` operation destroys any memory associated with
    a coroutine handle. A coroutine handle may or may not have any dynamic
    memory associated with it. A call to a coroutine function returns a
    coroutine handle and that handle must be deallocated with this operation.
    Doing anything with a coroutine handle after it is destroyed is undefined
    behaviour.

    Example:

    ```mlir
    co.destroy %hdl
    ```
  }];

  let arguments = (ins CoroutineType:$coroutine);
  let assemblyFormat = "$coroutine attr-dict";
}



#endif // COOPS_TD
