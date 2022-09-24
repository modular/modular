//===- ZAPDialect.td ------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef ZAPDIALECT_TD
#define ZAPDIALECT_TD

include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// ZAPDialect
//===----------------------------------------------------------------------===//

def ZAPDialect : Dialect {
  let name = "zap";

  let summary = "The `zap` dialect";
  let description = [{
    The `zap` dialect contains parametric high-level operations that model
    language-level features and/or provide syntax sugar on top of lower level
    operations. It is meant as a substitute for a kernel programming language
    and its standard libraries until they exist.

    This dialect contains operations that are conceptually above the `pop`
    dialect in terms of abstraction level and which lower to it.
  }];

  let cppNamespace = "::M::KGEN::ZAP";
  let useDefaultTypePrinterParser = 1;
  let hasConstantMaterializer = 1;

  let dependentDialects = ["POP::POPDialect"];

  let extraClassDeclaration = [{
    /// Register all dialect operations.
    void registerOperations();
    /// Register all dialect types.
    void registerTypes();
  }];
}

#endif // ZAPDIALECT_TD
