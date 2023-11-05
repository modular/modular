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

  /// Computed source names.
  DenseMap<mlir::SymbolOpInterface, DebugInfo::SourceNameAttr> names;
};
} // namespace M::KGEN::LIT

#endif // KGEN_LIB_MOJOPARSER_DEBUGINFO_H
