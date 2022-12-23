//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef INTERPRETERINTERFACE_TD
#define INTERPRETERINTERFACE_TD

include "mlir/IR/OpBase.td"

def InterpreterOpInterface : OpInterface<"InterpreterOpInterface"> {
  let description = [{
    This interface defines an API for operations to be "interpreted" at compile
    time. Operations may normally plug into the IR interpreter by defining an
    operation folder. However, some operations require more information or state
    to be interpreted. This interface provides access to an interpreter state
    that contains a memory model, a symbol table, and the ability to recursively
    invoke the interpreter.
  }];

  let cppNamespace = "::M";

  let methods = [
    InterfaceMethod<[{
      Given the constant value inputs to the operation, compute the constant
      value results, with access to the interpreter state.
    }], "ErrorOrSuccess", "interpret",
    (ins "ArrayRef<Attribute>":$operands, "InterpreterState &":$state,
         "SmallVectorImpl<OpFoldResult> &":$results)>
  ];
}

#endif // INTERPRETERINTERFACE_TD
