//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOGGPREELAB_USERLIBRARYCHECKER_H
#define KGEN_LIB_MOGGPREELAB_USERLIBRARYCHECKER_H

#include "KGEN/KGENDialect/KGENOps.h"

namespace M::KGEN::MOGGPreElab {

struct CallGraph;

/// This class performs various checks in the user-provided kernel library to
/// make sure it obeys the contract between the graph compiler and the kernel
/// library.
class UserLibraryChecker {
public:
  explicit UserLibraryChecker(ModuleOp module, const SymbolTable &symtab);

  ~UserLibraryChecker();

  /// Executes various checks against the user kernel library.
  LogicalResult run();

private:
  /// Checks the locations where some constrained MOGG Tensor APIs are invoked.
  LogicalResult checkCallsiteLocation();

  /// The pre-elaborated call graph derived from the user library.
  std::unique_ptr<CallGraph> cg;
  /// The `kgen.param.declare.region` operators in the user library. They don't
  /// exist in the call graph.
  llvm::SmallVector<ParamDeclareRegionOp> paramDeclRegions;
};

} // namespace M::KGEN::MOGGPreElab

#endif // KGEN_LIB_MOGGPREELAB_USERLIBRARYCHECKER_H
