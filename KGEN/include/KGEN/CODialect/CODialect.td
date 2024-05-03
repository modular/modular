//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef CODIALECT_TD
#define CODIALECT_TD

include "mlir/IR/DialectBase.td"

def CODialect : Dialect {
  let name = "co";

  let summary = "the coroutine dialect";
  let description = [{
    The `co` dialect defines a parametric opset for working with async functions
    implemented as coroutines with arbitrary suspension points.
  }];

  let cppNamespace = "::M::KGEN::CO";
  let useDefaultTypePrinterParser = 1;

  let dependentDialects = ["KGENDialect"];

  let extraClassDeclaration = [{
    void registerTypes();
    void registerOperations();
  }];
}

#endif // CODIALECT_TD
