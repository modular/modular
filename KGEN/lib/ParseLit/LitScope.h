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

#include "KGEN/LITDialect/LITOps.h"
#include "LLCL/Support/RCRef.h"
#include "LLCL/Support/ReferenceCounted.h"
#include "LitLexer.h"
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
  Scope(Operation *decl, LLCL::RCRef<Scope> parentScope, LitLexerCursor cursor)
      : decl(decl), parentScope(std::move(parentScope)),
        builder(decl->getContext()), cursor(cursor) {
    if (decl->getNumRegions())
      builder = OpBuilder::atBlockEnd(&decl->getRegion(0).front());
  }

  /// Return the Module, StructDecl, Func/Generator that this scope corresponds
  /// to.
  Operation *getDecl() const { return decl; }
  const LLCL::RCRef<Scope> &getParentScope() const { return parentScope; }

  OpBuilder &getBuilder() { return builder; }

  const LitLexerCursor &getCursor() const { return cursor; }

  /// Return the builder associated to the declaration that introduced the
  /// Scope.
  /// This method must be used instead of getBuilder() when we create
  /// variable declaration ops to make sure we honor the one scope per function
  /// rule of Python.
  OpBuilder getDeclBuilder() {
    return OpBuilder::atBlockBegin(&decl->getRegion(0).front());
  }

  /// This is the value of a parameter bound to a name, attributes in MLIR don't
  /// track locations, so we do so explicitly here.
  struct MetaParameterValue {
    Attribute value;
    Location loc;

    /// The value in a MetaParameterValue is always known to be a TypedAttr.
    TypedAttr getAttr() const { return cast<TypedAttr>(value); }
  };

  /// An entry in the symbol table is either a mutable variable declaration
  /// (VarDeclOp) or an immutable attribute (which is known to be a TypedAttr).
  using ScopeValue = std::variant<MetaParameterValue, VarDeclOp>;

  /// Add the specified declaration to the current scope, emitting an error on
  /// a name collision and setting hadError to true.
  void addToScope(StringRef name, ScopeValue newValue,
                  LitSharedState &sharedState);

  /// Look up a name in the current scope only.
  Optional<ScopeValue> lookupInCurrentScope(StringRef name) {
    auto it = decls.find(name);
    if (it != decls.end())
      return it->second;
    return None;
  }

  /// Perform a lookup in this scope list, returning the nearest target or None
  /// if nothing is found.
  Optional<ScopeValue> lookup(StringRef name) {
    Scope *curScope = this;
    while (curScope) {
      if (Optional<ScopeValue> result = curScope->lookupInCurrentScope(name))
        return result.value();
      curScope = curScope->parentScope.getPointer();
    }
    return None;
  }

  bool getIsResolved() const { return isResolved; }
  void setIsResolved() { isResolved = true; }

private:
  /// This is the Module, LITStructDecl, LITFunc that this scope corresponds
  /// to.
  Operation *decl;
  LLCL::RCRef<Scope> parentScope;
  OpBuilder builder;

  /// This is the cursor that points to the start of the declaration.  This is
  /// useful if we want to reparse the declaration.
  LitLexerCursor cursor;

  // Note: we could unique the identifiers and use a DenseMap.
  llvm::StringMap<Optional<ScopeValue>> decls;

  /// This is true if this declaration is fully type checked.
  /// TODO: Split to enum.
  bool isResolved = false;
};

} // namespace M::KGEN::LIT

#endif // LITSCOPE_H
