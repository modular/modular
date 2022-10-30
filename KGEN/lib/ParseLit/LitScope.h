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

/// This stores the ParamDeclAttr as an Attribute, but this is always known to
/// be a ParamDeclAttr.
using OperationOrParamDecl = PointerUnion<Operation *, Attribute>;

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
  MLIRContext *getContext() const { return loc.getContext(); }

  /// Return the Module, StructDecl, Func, or ParamDecl that this scope
  /// corresponds to.
  OperationOrParamDecl getDecl() const { return decl; }

  /// If this is a ParamDecl, return it otherwise return null.
  ParamDeclAttr getParamDecl() const {
    auto attr = dyn_cast<Attribute>(decl);
    return attr ? cast<ParamDeclAttr>(attr) : ParamDeclAttr();
  }

  Location getLoc() const { return loc; }
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
    if (Operation *op = dyn_cast<Operation *>(decl))
      if (op->getNumRegions() != 0)
        return OpBuilder::atBlockEnd(&op->getRegion(0).front());
    return OpBuilder(getContext());
  }

  /// This is the value of a parameter bound to a name, attributes in MLIR don't
  /// track locations, so we do so explicitly here.
  struct MetaParameterValue {
    Attribute value;
    Location loc;

    /// The value in a MetaParameterValue is always known to be a TypedAttr.
    TypedAttr getAttr() const { return cast<TypedAttr>(value); }
  };

  /// Add the specified declaration to the current scope, emitting an error on
  /// a name collision and setting hadError to true.
  void addToScope(Scope *newDeclScope, LitSharedState &sharedState);

  /// Look up a name in the current scope only, this returns null on failure.
  Scope *lookupInCurrentScope(StringAttr name) {
    auto it = decls.find(name);
    if (it != decls.end())
      return it->second;
    return nullptr;
  }

  /// Perform a lookup in this scope list, returning the nearest target or None
  /// if nothing is found.
  Scope *lookup(StringAttr name) {
    Scope *curScope = this;
    while (curScope) {
      if (Scope *result = curScope->lookupInCurrentScope(name))
        return result;
      curScope = curScope->parentScope;
    }
    return nullptr;
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

  enum class MagicKind {
    // This is not a magic declaration, process it as normal.
    kNormal,
    // This is the __builtin.mlirtype.builtin.index type.
    kIndexType,
    // This is the __builtin.mlirtype.lit.none type.
    kNoneType,
  } magicKind = MagicKind::kNormal;

private:
  // Scope is created by DeclResolver.
  friend class DeclResolver;
  Scope(OperationOrParamDecl decl, Location loc, Scope *parentScope,
        LitLexerCursor cursor, LitLexerCursor endCursor, ssize_t indentation)
      : decl(decl), loc(loc), parentScope(std::move(parentScope)),
        cursor(cursor), endCursorState(endCursor.getState()),
        indentation(indentation) {}

private:
  /// This is the declaration that this scope corresponds to.
  OperationOrParamDecl decl;

  /// This is the source location of the declaration, used for diagnostics and
  /// debug information.
  Location loc;

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
  DenseMap<StringAttr, Scope *> decls;
};

} // namespace M::KGEN::LIT

namespace llvm {
/// Cast from an (const) Scope & to a Decl operation type.
template <typename T>
struct CastInfo<T, M::KGEN::LIT::Scope>
    : public NullableValueCastFailed<T>,
      public DefaultDoCastIfPossible<T, M::KGEN::LIT::Scope &,
                                     CastInfo<T, M::KGEN::LIT::Scope>> {
  // Provide isPossible here because here we have the const-stripping from
  // ConstStrippingCast.
  static bool isPossible(M::KGEN::LIT::Scope &scope) {
    auto *decl = dyn_cast<mlir::Operation *>(scope.getDecl());
    return decl && T::classof(decl);
  }
  static T doCast(M::KGEN::LIT::Scope &scope) {
    return T(cast<mlir::Operation *>(scope.getDecl()));
  }
};
template <typename T>
struct CastInfo<T, const M::KGEN::LIT::Scope>
    : public ConstStrippingForwardingCast<T, const M::KGEN::LIT::Scope,
                                          CastInfo<T, M::KGEN::LIT::Scope>> {};
} // namespace llvm

#endif // LITSCOPE_H
