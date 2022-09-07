//===- LitDialect.td ------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/DialectBase.td"

#ifndef LITDIALECT_TD
#define LITDIALECT_TD

//===----------------------------------------------------------------------===//
// LitDialect
//===----------------------------------------------------------------------===//

def LitDialect : Dialect {
  let name = "lit";

  let summary = "The Lightning ⚡️ dialect";
  let description = [{
    The Lightning ⚡️ dialect contains parametric high-level operations that model
    language-level features and/or provide syntax sugar on top of lower level
    operations. It is meant as a substitute for a kernel programming language
    and its standard libraries until they exist.

    This dialect contains operations that are conceptually above the `pop` and
    `meta` dialects in terms of abstraction level and which lower to them.
  }];

  let cppNamespace = "::M::KGEN";

  let extraClassDeclaration = [{
    /// Register all dialect operations.
    void registerOperations();
  }];

  let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
}

#endif // LITDIALECT_TD
