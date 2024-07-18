//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef CUSTOMDIALECT_TD
#define CUSTOMDIALECT_TD

include "mlir/IR/DialectBase.td"

def CustomDialect : Dialect {
  let name = "custom";

  let summary = "the custom dialect";
  let description = [{
    The `custom` dialect contains all user-defined ops.
    No ops are registered by default in the dialect, and ops are registered
    during the compilation pipeline based on the implementation provided by
    structs annotated with the `op_implementation` decorator.
  }];

  let cppNamespace = "::M::KGEN::Custom";

  let extraClassDeclaration = [{
    /// Canonicalization functions of the `custom` unregistered ops.
    DenseMap<StringAttr,
             std::function<LogicalResult(Operation *, PatternRewriter &)>>
        canonicalizationFns;
  }];
}

#endif // CUSTOMDIALECT_TD
