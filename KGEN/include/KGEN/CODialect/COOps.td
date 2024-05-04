//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef COOPS_TD
#define COOPS_TD

include "KGEN/CODialect/CODialect.td"
include "KGEN/CODialect/COTypes.td"
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
    %hdl = co.handle : <i32, index>
    ```
  }];

  let results = (outs CoroutineType:$coroutine);
  let assemblyFormat = "attr-dict `:` type($coroutine)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CoroutineOpaqueHandleOp
//===----------------------------------------------------------------------===//

def CO_CoroutineOpaqueHandleOp : COOp<"opaque_handle"> {
  let summary = "Get an opaque handle to the current coroutine, if it is one.";
  let description = [{
    The `co.opaque_handle` returns an opaque handle to the current
    coroutine, if it is one. If the function in which this operation resides is
    not a coroutine, the result is a null pointer.

    Example:

    ```mlir
    %hdl = co.opaque_handle
    ```
  }];

  let results = (outs OpaqueCoroutineHandle:$hdl);
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// CoroutineSuspendOp
//===----------------------------------------------------------------------===//

def CO_CoroutineSuspendOp : COOp<"suspend", [SingleBlock]> {
  let summary = "Suspend the current coroutine and await resumption.";
  let description = [{
    The `co.suspend` operation suspends the current coroutine function
    at the site of the operation and executes a region. The code in the region
    must resume the coroutine at some point. This operation can only be used
    inside an async function.

    Example:

    ```mlir
    %hdl = co.handle : <i32, index>
    co.suspend {
      co.resume %hdl : <i32>
      co.suspend.end
    }
    ```
  }];

  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$body attr-dict";
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

def CO_CoroutinePromiseOp : COOp<"promise",
    [TypesMatchWith<"result is pointer to coroutine promise", "coroutine",
                    "promise", "getCoroutinePromiseType($_self)">]> {
  let summary = "Get the coroutine promise.";
  let description = [{
    The `co.promise` operation returns the pointer to the coroutine
    promise, which contains the result values of the promise. Located at the
    end of the promise is the async context.

    Example:

    ```mlir
    %promise = co.promise %hdl : <i32, i64>
    ```
  }];

  let arguments = (ins CoroutineType:$coroutine);
  let results = (outs PointerOf<KGEN_StructType>:$promise);
  let assemblyFormat = "$coroutine attr-dict `:` type($coroutine)";
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

  let arguments = (ins AnyTypeOf<[CoroutineType,
                                  OpaqueCoroutineHandle]>:$coroutine);
  let assemblyFormat = "$coroutine attr-dict `:` type($coroutine)";
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
    co.destroy %hdl : <i32>
    ```
  }];

  let arguments = (ins CoroutineType:$coroutine);
  let assemblyFormat = "$coroutine attr-dict `:` type($coroutine)";
}



#endif // COOPS_TD
