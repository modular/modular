//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Declaration parsing and name binding logic.
//
//===----------------------------------------------------------------------===//

#ifndef LITDECLS_H
#define LITDECLS_H

#include "LLCL/Support/RCRef.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/ADT/DenseMap.h"

namespace M::KGEN::LIT {

class LitSharedState;
class Scope;
//===----------------------------------------------------------------------===//
// DeclResolver
//===----------------------------------------------------------------------===//

class DeclResolver {
public:
  DeclResolver(LitSharedState &state);
  ~DeclResolver();

  /// Resolve all of the declarations that are visible, processing the entire
  /// translation unit.
  void resolveAll();

  /// Add a new declaration that needs to be resolved.
  void addDecl(LLCL::RCRef<Scope> declScope);

private:
  void resolve(Scope &scope);

private:
  /// This is shared state across the whole parser.
  LitSharedState &sharedState;

  /// This is a mapping of every declaration (module, func, struct, etc) that
  /// we have parsed, along with the metadata for it maintained in `Scope`.
  DenseMap<Operation *, LLCL::RCRef<Scope>> parsedDecls;

  /// This array holds all of the parsed declarations in a deterministic order.
  std::vector<Operation *> parsedDeclList;
};

} // namespace M::KGEN::LIT

#endif // LITDECLS_H
