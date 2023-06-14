//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef INTERPRETERINTERFACE_TD
#define INTERPRETERINTERFACE_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// InterpreterOpInterface
//===----------------------------------------------------------------------===//

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
      value results, with access to the interpreter state and the contextual
      value map.
    }], "ErrorTreeOrSuccess", "interpret",
    (ins "ArrayRef<Attribute>":$operands, "InterpreterState &":$state)>
  ];
}

//===----------------------------------------------------------------------===//
// MemoryableTypeInterface
//===----------------------------------------------------------------------===//

def MemoryableTypeInterface : TypeInterface<"MemoryableTypeInterface"> {
  let description = [{
    This interface allows types to provide an API with which to read and write
    the constant values of its type as raw data. This allows types to plug into
    the interpreter's memory model and enables, for example, writing
    `struct<i32, i32>` to a piece of memory, then bitcasting that to read as an
    `i64`.
  }];

  let cppNamespace = "::M";

  let methods = [
    InterfaceMethod<[{
      Write the raw contents of the attribute to the provided chunk of memory.
    }], "ErrorOrSuccess", "writeTo", (ins "TypedAttr":$value,  "int64_t":$addr,
                                          "InterpreterState &":$state)>,
    InterfaceMethod<[{
      Materialize an instance of this type by reading the raw contents of an
      attribute that represents a value of this type from the provided chunk of
      memory.
    }], "ErrorOr<TypedAttr>", "readFrom", (ins "int64_t":$addr,
                                               "InterpreterState &":$state)>
  ];
}

#endif // INTERPRETERINTERFACE_TD
