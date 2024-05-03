//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef CODIALECTBYTECODE_TD
#define CODIALECTBYTECODE_TD

include "Support/Compiler/Bytecode.td"

//===----------------------------------------------------------------------===//
// CODialect Types
//===----------------------------------------------------------------------===//

def CoroutineType : DialectType<(type
  ConcreteType<"SignatureType">:$signature
)>;

def CODialectTypes : DialectTypes<"co"> {
  let elems = [
    CoroutineType
  ];
}

//===----------------------------------------------------------------------===//
// CODialect Attributes
//===----------------------------------------------------------------------===//

def CODialectAttributes : DialectAttributes<"co"> {
  let elems = [];
}

#endif // CODIALECTBYTECODE_TD
