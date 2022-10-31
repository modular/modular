//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// AST representation for a declaration.
//
//===----------------------------------------------------------------------===//

#ifndef LIT_DECL_AST_H
#define LIT_DECL_AST_H

#include "LitLexer.h"
#include "LitSharedState.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"

namespace M::KGEN {
class ParamDeclAttr;
class RefType;
}

namespace M::KGEN::LIT {

/// This stores the ParamDeclAttr as an Attribute, but this is always known to
/// be a ParamDeclAttr.  When both are null, this is a 'magic' declaration.
using IRDecl = PointerUnion<Operation *, Attribute>;

/// This is the AST representation (as opposed to the MLIR representation) of a
/// declaration in a program.  These maintain type checking and other
/// information that is irrelevant by the time the parser has created a complete
/// and correct IR for a program.  Declarations often have other declarations
/// nested inside of them, forming "scopes" and supporting name lookup.
///
/// Declarations in Lightning work the same way as in Python: scopes are nested
/// and are defined when a builtin, module, class/struct, or function definition
/// is introduced.  Lightning (like Python) allows forward references to values
/// before they are defined, so the parser works in multiple phases where it
/// notices a declaration but does not parse its body until it is demanded.
class ASTDecl {
public:
  MLIRContext *getContext() const { return loc.getContext(); }

  /// Return the Module, StructDecl, Func, or ParamDecl that this scope
  /// corresponds to.
  IRDecl getIRDecl() const { return irDecl; }

  /// If this is a ParamDecl, return it otherwise return null.
  ParamDeclAttr getParamDecl() const;

  /// Return true if this is a "magic" declaration that has no IR
  /// representation.
  bool isMagic() const { return irDecl.isNull(); }

  Location getLoc() const { return loc; }
  ASTDecl *getParentDecl() const { return parentDecl; }

  /// Return the indentation of the introducer token or -1 if it wasn't on the
  /// start of line.
  ssize_t getIndentation() const { return indentation; }

  /// This cursor holds the location the parser should resume for the next phase
  /// of resolution.  For example, after initial scanning of a 'def', this will
  /// be on the def token.  After processing the signature, this will be after
  /// the colon.
  LitLexerCursor &getCursor() { return cursor; }

  /// Return true if the end of the speculatively scanned decl matches the
  /// specified cursor.
  bool isMatchingEndCursor(const LitLexerCursor &cursor) const {
    return endCursorState == cursor.getState();
  }

  /// Return the builder at the end of the region that the decl contains.
  OpBuilder getDeclEndBuilder() {
    if (Operation *op = dyn_cast<Operation *>(irDecl))
      if (op->getNumRegions() != 0)
        return OpBuilder::atBlockEnd(&op->getRegion(0).front());
    return OpBuilder(getContext());
  }

  /// Given a type declaration, return a RefType for a reference to this with
  /// the specified type parameters.  This aborts if the current decl isn't a
  /// type.
  RefType getIRTypeReference(ParamBindArrayAttr params);

  //===--------------------------------------------------------------------===//
  // Name lookup
  //===--------------------------------------------------------------------===//

  /// Look up a name in this declaration's scope only: return null on failure.
  ASTDecl *lookupInCurrentScope(StringAttr name) {
    auto it = declsInScope.find(name);
    if (it != declsInScope.end())
      return it->second;
    return nullptr;
  }

  /// Perform a lookup in this declaration's scope and all parent scopes,
  /// returning the nearest target or null if nothing is found.
  ASTDecl *lookup(StringAttr name) {
    ASTDecl *curScope = this;
    while (curScope) {
      if (ASTDecl *result = curScope->lookupInCurrentScope(name))
        return result;
      curScope = curScope->parentDecl;
    }
    return nullptr;
  }

  //===--------------------------------------------------------------------===//
  // Other State management.
  //===--------------------------------------------------------------------===//

  /// This keeps track of what level of type checking this declaration has been
  /// through.  It is maintained by DeclResolver.
  DeclResolvedness resolvedness = DeclResolvedness::unparsed;

  /// This is set to true when an error is detected and reported about this
  /// declaration that could cause references to it to cause spurious downstream
  /// errors.  For example, "var x : SomeUndeclaredType" will cause errors for
  /// every reference to 'x' because the type will be bogus.
  bool hasReferenceError = false;

  MagicDeclKind magicKind = MagicDeclKind::kNormal;

private:
  // ASTDecl is created by DeclResolver.
  friend class DeclResolver;
  ASTDecl(IRDecl irDecl, Location loc, ASTDecl *parentDecl,
          LitLexerCursor cursor, LitLexerCursor endCursor, ssize_t indentation)
      : irDecl(irDecl), loc(loc), parentDecl(std::move(parentDecl)),
        cursor(cursor), endCursorState(endCursor.getState()),
        indentation(indentation) {}

private:
  /// This is the MLIR declaration that this scope corresponds to.
  IRDecl irDecl;

  /// This is the source location of the declaration, used for diagnostics and
  /// debug information.
  Location loc;

  /// This the parent scope that should continue name lookup, or null for the
  /// top scope.
  ASTDecl *parentDecl;

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
  DenseMap<StringAttr, ASTDecl *> declsInScope;
};

} // namespace M::KGEN::LIT

namespace llvm {
/// Cast from an (const) ASTDecl & to a Decl operation type.
template <typename T>
struct CastInfo<T, M::KGEN::LIT::ASTDecl>
    : public NullableValueCastFailed<T>,
      public DefaultDoCastIfPossible<T, M::KGEN::LIT::ASTDecl &,
                                     CastInfo<T, M::KGEN::LIT::ASTDecl>> {
  // Provide isPossible here because here we have the const-stripping from
  // ConstStrippingCast.
  static bool isPossible(M::KGEN::LIT::ASTDecl &decl) {
    auto *op = dyn_cast<mlir::Operation *>(decl.getIRDecl());
    return op && T::classof(op);
  }
  static T doCast(M::KGEN::LIT::ASTDecl &decl) {
    return T(cast<mlir::Operation *>(decl.getIRDecl()));
  }
};
template <typename T>
struct CastInfo<T, const M::KGEN::LIT::ASTDecl>
    : public ConstStrippingForwardingCast<T, const M::KGEN::LIT::ASTDecl,
                                          CastInfo<T, M::KGEN::LIT::ASTDecl>> {
};
} // namespace llvm

#endif // LIT_DECL_AST_H
