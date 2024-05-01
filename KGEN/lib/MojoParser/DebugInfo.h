//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOJOPARSER_DEBUGINFO_H
#define KGEN_LIB_MOJOPARSER_DEBUGINFO_H

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/MojoParser/SharedState.h"
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"

namespace M::KGEN::LIT {
struct SourceNames : public SharedStateUser {
  using SharedStateUser::SharedStateUser;
  /// Get the source name of a symbol.
  DebugInfo::SourceNameAttr getSourceName(mlir::SymbolOpInterface op);
  /// Get the source name of a type.
  DebugInfo::SourceNameAttr getSourceName(Type type);
  /// Forget that any source name was assigned to this op. Does not modify the
  /// op itself, only the internal source name cache.
  void forgetSourceName(mlir::SymbolOpInterface op) { names.erase(op); }
  /// Get source names of decorators on an op, filling the out vector.
  void processDecorators(Operation *op,
                         SmallVectorImpl<DebugInfo::SourceNameAttr> &out);

  /// Computed source names.
  DenseMap<mlir::SymbolOpInterface, DebugInfo::SourceNameAttr> names;
};
} // namespace M::KGEN::LIT

#endif // KGEN_LIB_MOJOPARSER_DEBUGINFO_H
