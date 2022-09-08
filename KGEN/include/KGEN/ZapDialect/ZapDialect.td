//===- ZapDialect.td ------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/DialectBase.td"

#ifndef ZAPDIALECT_TD
#define ZAPDIALECT_TD

//===----------------------------------------------------------------------===//
// ZapDialect
//===----------------------------------------------------------------------===//

def ZapDialect : Dialect {
  let name = "zap";

  let summary = "The Zap ⚡️ dialect";
  let description = [{
    The Zap ⚡️ dialect contains parametric high-level operations that model
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

#endif // ZAPDIALECT_TD
