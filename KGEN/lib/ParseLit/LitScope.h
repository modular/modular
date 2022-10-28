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
#include "LitLexer.h"
#include "LitSharedState.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"

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
class Scope {
public:
  /// Return the Module, StructDecl, Func/Generator that this scope corresponds
  /// to.
  Operation *getDecl() const { return decl; }
  Scope *getParentScope() const { return parentScope; }

  /// This cursor holds the location the parser should resume for the next phase
  /// of resolution.  For example, after initial scanning of a 'def', this will
  /// be on the def token.  After processing the signature, this will be after
  /// the colon.
  LitLexerCursor &getCursor() { return cursor; }

  /// Return the indentation of the introducer token or -1 if it wasn't on the
  /// start of line.
  ssize_t getIndentation() const { return indentation; }

  /// Return the builder at the end of the region that the decl contains.
  OpBuilder getDeclEndBuilder() {
    if (decl->getNumRegions() == 0)
      return OpBuilder(decl->getContext());
    return OpBuilder::atBlockEnd(&decl->getRegion(0).front());
  }

  /// This is the value of a parameter bound to a name, attributes in MLIR don't
  /// track locations, so we do so explicitly here.
  struct MetaParameterValue {
    Attribute value;
    Location loc;

    /// The value in a MetaParameterValue is always known to be a TypedAttr.
    TypedAttr getAttr() const { return cast<TypedAttr>(value); }
  };

  /// An entry in the symbol table is either the Scope for a declaration (var,
  /// struct, func, etc) or an attribute (known to be a TypedAttr) for a meta
  /// value.
  using NameEntry = std::variant<MetaParameterValue, Scope *>;

  /// Add the specified declaration to the current scope, emitting an error on
  /// a name collision and setting hadError to true.
  void addToScope(StringAttr name, MetaParameterValue newValue,
                  LitSharedState &sharedState);
  void addToScope(Scope *newDeclScope, LitSharedState &sharedState);

  /// Look up a name in the current scope only.
  Optional<NameEntry> lookupInCurrentScope(StringAttr name) {
    auto it = decls.find(name);
    if (it != decls.end())
      return it->second;
    return None;
  }

  /// Perform a lookup in this scope list, returning the nearest target or None
  /// if nothing is found.
  Optional<NameEntry> lookup(StringAttr name) {
    Scope *curScope = this;
    while (curScope) {
      if (Optional<NameEntry> result = curScope->lookupInCurrentScope(name))
        return result.value();
      curScope = curScope->parentScope;
    }
    return None;
  }

  /// Return true if the end of the speculatively scanned decl matches the
  /// specified cursor.
  bool isMatchingEndCursor(const LitLexerCursor &cursor) const {
    return endCursorState == cursor.getState();
  }

  /// This keeps track of what level of type checking this declaration has been
  /// through.  It is maintained by DeclResolver.
  DeclResolvedness resolvedness = DeclResolvedness::unparsed;

  /// This is set to true when an error is detected and reported about this
  /// declaration that could cause references to it to cause spurious downstream
  /// errors.  For example, "var x : SomeUndeclaredType" will cause errors for
  /// every reference to 'x' because the type will be bogus.
  bool hasReferenceError = false;

private:
  // Scope is created by DeclResolver.
  friend class DeclResolver;
  Scope(Operation *decl, Scope *parentScope, LitLexerCursor cursor,
        LitLexerCursor endCursor, ssize_t indentation)
      : decl(decl), parentScope(std::move(parentScope)), cursor(cursor),
        endCursorState(endCursor.getState()), indentation(indentation) {}

private:
  /// This is the declaration that this scope corresponds to.
  Operation *decl;

  /// This the parent scope that should continue name lookup, or null for the
  /// top scope.
  Scope *parentScope;

  /// This is the cursor that points to the next part of declaration to continue
  /// parsing as the declaration is progressively resolved.
  LitLexerCursor cursor;

  /// This is the lexer cursor state for the first token /after/ the
  /// declaration.  This is used to make sure that bits of a declaration are not
  /// skipped in the early parse and not processes in the later parse.
  const char *endCursorState;

  /// This is the indentation level of the introducer keyword, useful for
  /// parsing the body of the declaration.  If the declaration was not at the
  /// start of a line or this is the top level module, then this is set to -1.
  ssize_t indentation;

  /// These are the declarations defined within this scope.
  DenseMap<StringAttr, NameEntry> decls;
};

} // namespace M::KGEN::LIT

#endif // LITSCOPE_H
