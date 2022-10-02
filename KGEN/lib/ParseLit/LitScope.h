//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Scope handling
//
//===----------------------------------------------------------------------===//

#ifndef LITSCOPE_H
#define LITSCOPE_H

#include "LLCL/Support/RCRef.h"
#include "LLCL/Support/ReferenceCounted.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/StringMap.h"

namespace M::KGEN::LIT {
/// Scopes in Lightning work the same way as in Python: scopes are nested and
/// are defined when a builtin, module, class/struct, or function definition is
/// introduced.  Because Lightning (like Python) allows forward references to
/// values before they are defined, the body of declarations is parsed after the
/// signature of its peer declarations are all parsed.
///
/// This means that we can't just use a ScopedHashTable or similar - we need to
/// maintain our scopes until all bodies that refer to them are resolved.  As
/// such, we heap allocate and reference count these.
class Scope : public LLCL::NonAtomicallyReferenceCounted<Scope> {
public:
  Scope(Operation *decl, LLCL::RCRef<Scope> parentScope)
      : decl(decl), parentScope(std::move(parentScope)) {}

  /// Return the Module, StructDecl, Func/Generator that this scope corresponds
  /// to.
  Operation *getDecl() const { return decl; }
  const LLCL::RCRef<Scope> &getParentScope() const { return parentScope; }

  OpBuilder getBuilder() {
    return OpBuilder::atBlockEnd(&decl->getRegion(0).front());
  }

  /// Add the specified declaration to the current scope, returning non-null if
  /// a previous operation is already in this scope.
  Operation *addToScope(StringRef name, Operation *newDecl) {
    Operation *&entry = decls[name];
    if (entry)
      return entry;
    entry = newDecl;
    return nullptr;
  }

  /// Look up a name in the current scope only.
  Operation *lookupInCurrentScope(StringRef name) {
    auto it = decls.find(name);
    if (it != decls.end())
      return it->second;
    return nullptr;
  }

  /// Perform a lookup in this scope tree, returning the nearest target or null
  /// if nothing is found.
  Operation *lookup(StringRef name) {
    Scope *curScope = this;
    while (curScope) {
      if (Operation *result = curScope->lookupInCurrentScope(name))
        return result;
      curScope = curScope->parentScope.getPointer();
    }
    return nullptr;
  }

private:
  /// This is the Module, StructDecl, Func/Generator that this scope corresponds
  /// to.
  Operation *decl;
  LLCL::RCRef<Scope> parentScope;

  // Note: we could unique the identifiers and use a DenseMap.
  llvm::StringMap<Operation *> decls;
};

} // namespace M::KGEN::LIT

#endif // LITSCOPE_H
