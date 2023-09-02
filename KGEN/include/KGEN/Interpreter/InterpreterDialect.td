//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef INTERPRETERDIALECT_TD
#define INTERPRETERDIALECT_TD

include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// InterpreterDialect
//===----------------------------------------------------------------------===//

def InterpreterDialect : Dialect {
  let name = "interp";

  let summary = "interpreter dialect";
  let description = [{
    The `interp` dialect implements attributes and utilities for
    communicating with the interpreter in MLIR.
  }];

  let cppNamespace = "::M";

  let extraClassDeclaration = [{
    /// Register all dialect attributes.
    void registerAttributes();
  }];

  let useDefaultAttributePrinterParser = 1;
}

#endif // INTERPRETERDIALECT_TD
