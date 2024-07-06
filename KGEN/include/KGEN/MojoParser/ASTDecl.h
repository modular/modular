//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// AST representation for a declaration.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_ASTDECL_H
#define KGEN_MOJOPARSER_ASTDECL_H

#include "KGEN/MojoParser/DeclResolver.h"
#include "KGEN/MojoParser/Lexer.h"
#include "KGEN/MojoParser/SharedState.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/TinyPtrVector.h"

namespace M::KGEN {
class ParamDeclAttr;
} // namespace M::KGEN

namespace M::KGEN::LIT {
class StructType;
class DocStringAttr;
class DocString;

/// This is the AST representation (as opposed to the MLIR representation) of a
/// declaration in a program.  These maintain type checking and other
/// information that is irrelevant by the time the parser has created a complete
/// and correct IR for a program.  Declarations often have other declarations
/// nested inside of them, forming "scopes" and supporting name lookup.
///
/// Declarations in Mojo work the same way as in Python: scopes are nested
/// and are defined when a builtin, module, class/struct, or function definition
/// is introduced.  Mojo (like Python) allows forward references to values
/// before they are defined, so the parser works in multiple phases where it
/// notices a declaration but does not parse its body until it is demanded.
class ASTDecl {
public:
  MLIRContext *getContext() const;

  /// Return the Module, StructDecl, Func, or ParamDecl that this scope
  /// corresponds to.
  DeclIRValue getIRValue() const { return irValue; }
  void setIRValue(DeclIRValue value) { irValue = std::move(value); }

  /// Dump the underlying IR value.
  void dump() const;

  /// If this declaration is defined by its value (e.g. a parameter value or an
  /// SSA value) then return it.
  RValue getIfRValue() const;
  BValue getIfBValue() const;
  PValue getIfPValue() const { return dyn_cast_or_null<PValue>(irValue); }
  LValue getIfLValue() const;

  /// If the IRValue is an Operation*, return it, otherwise return null.
  Operation *getIfOperation() const {
    return dyn_cast_or_null<Operation *>(irValue);
  }

  /// Get the name of the declaration if it has one.
  std::optional<StringRef> getNameIfOperation() const;

  /// If the IRValue is a function, return it as a PValue.
  PValue getFuncAsPValue() const;

  llvm::SMLoc getLoc() const { return loc; }
  ASTDecl *getParentDecl() const { return parentDecl; }

  /// Get the nearest decl backed by one of the given operations. This can
  /// return itself, a parent decl, or null if no such decl is found.
  template <typename... OpTs>
  ASTDecl *getNearestDeclOfType() {
    ASTDecl *cur = this;
    while (cur && !isa<OpTs...>(*cur))
      cur = cur->getParentDecl();
    return cur;
  }

  /// Return the indentation of the introducer token or -1 if it wasn't on the
  /// start of line.
  ssize_t getIndentation() const { return indentation; }

  /// This cursor holds the location the parser should resume for the next phase
  /// of resolution.  For example, after initial scanning of a 'def', this will
  /// be on the def token.  After processing the signature, this will be after
  /// the colon.
  LexerCursor &getCursor() { return cursor; }

  /// Return true if the end of the speculatively scanned decl matches the
  /// specified cursor.
  bool isMatchingEndCursor(const LexerCursor &cursor) const {
    return endCursorState == cursor.getState();
  }

  /// Return the SymbolRefAttr for a declaration, including all scoping that may
  /// be needed, making it unique for every declaration.  This returns null for
  /// named values that do not have a declaration.
  SymbolRefAttr getSymbolRef() const;

  /// Return the builder at the end of the region that the decl contains.
  OpBuilder getDeclEndBuilder() {
    if (Operation *op = dyn_cast<Operation *>(irValue))
      if (op->getNumRegions() != 0)
        return OpBuilder::atBlockEnd(&op->getRegion(0).front());
    return OpBuilder(getContext());
  }

  /// This return the 'Self' type for a struct or trait, which includes
  /// parameters bound to references to the struct parameter declarations.
  ASTType getTypeDeclSelf() const {
    assert(resolvedness != DeclResolvedness::unparsed &&
           "signature must be resolved to get a resolved type");
    return typeDeclSelf;
  }
  void setTypeDeclSelf(ASTType type) {
    assert(type && "Cannot set null types");
    typeDeclSelf = type;
  }

  /// Given an MLIR op for a struct declaration, return the self type.
  static Type computeSelfTypeForStruct(StructDeclOp structOp);

  /// Given an MLIR op for a trait declaration, return the self type.
  static Type computeSelfTypeForTrait(TraitDeclOp traitOp);

  /// Add an unresolved wild card import into this scope.
  void addUnresolvedWildCardImport(StringAttr importedModule, bool isFullImport,
                                   SMLoc loc) {
    unresolvedWildcardImports.insert({importedModule, {loc, isFullImport}});
  }

  /// Return the doc string for this decl, or nullptr if there isn't one.
  DocStringAttr getDocString() const;

  /// Return the parsed `DocString` for this decl if available.
  std::optional<DocString> getParsedDocString() const;

  /// Given a decl for a struct or trait type, return true if this type conforms
  /// to the specified trait type.  On failure, this *might* set 'diag' to an
  /// inflight diagnostic that explains why this doesn't conform.  It can be
  /// reported or abandoned based on the client's needs.
  bool doesNominalTypeConformsTo(TraitType trait,
                                 std::optional<InflightDiag> &diag,
                                 SharedState &shared);

  //===--------------------------------------------------------------------===//
  // Name lookup
  //===--------------------------------------------------------------------===//

  /// Look up a name in this declaration's scope only: return null on failure.
  ArrayRef<ASTDecl *> lookupInCurrentScope(StringAttr name) const;
  ArrayRef<ASTDecl *> lookupInCurrentScope(StringRef name) const;

  /// Perform a lookup in this declaration's scope and all parent scopes,
  /// returning the nearest target or empty if nothing is found.
  ArrayRef<ASTDecl *> lookup(StringAttr name) const {
    const ASTDecl *curScope = this;
    while (curScope) {
      ArrayRef<ASTDecl *> result = curScope->lookupInCurrentScope(name);
      if (!result.empty())
        return result;
      curScope = curScope->parentDecl;
    }
    return {};
  }

  /// Return the set of declarations in this scope.
  const llvm::MapVector<StringAttr, TinyPtrVector<ASTDecl *>> &
  getDeclsInScope() const {
    return declsInScope;
  }

  //===--------------------------------------------------------------------===//
  // Other State management.
  //===--------------------------------------------------------------------===//

  /// This keeps track of what level of type checking this declaration has been
  /// through.  It is maintained by DeclResolver.
  DeclResolvedness resolvedness = DeclResolvedness::unparsed;

  /// Indicate that the decl has reference errors.
  void setErroneous();
  /// Return true if the decl has reference errors.
  bool isErroneous() const { return hasReferenceError; }

  /// Return any decorators that need to be processed as part of body resolution
  /// phase for a decl.
  ArrayRef<ExprNode *> getBodyDecorators(SharedState &state) const;

  /// During signature resolution, this is called with any decorators that need
  /// to persist until body resolution.
  void setBodyDecorators(ArrayRef<ExprNode *> decorators, SharedState &state);

  /// Check if the given name collides with an existing user declared parameter
  /// name in the scope, and if so, uniquely mangle it by postpending a backtick
  /// ("`"), scope depth, and a unique ID.
  StringAttr mangleUserDefinedParamName(StringAttr name);

  /// Create a unique parameter name by postpending a backtick ("`"), scope
  /// depth, and a unique ID.
  StringAttr mangleParamName(const Twine &name);

  /// Move the children decls of `src` into this decl. This is useful when a
  /// temporary decl needs to be created for parsing subexpressions but whose
  /// children will be inherited later by a decl being resolved.
  void takeDecls(ASTDecl &src);

  /// Anonymous lifetimes, closure impl structs, and potentially other names are
  /// uniqued to avoid collisions. This returns an ID that is unique to this
  /// ASTDecl instance and help generate such names.
  unsigned getNextUniqueID() { return counter++; }

private:
  /// This is set to true if there is an entry for body-decorators in a
  /// backing hashtable.  Clients should use "getBodyDecorators().
  bool hasBodyDecorators = false;

  /// This is set to true when the declaration was loaded from bytecode, not
  /// parsed from a textual source file. These declarations behave differently
  /// than source decls, and e.g., do not resolve in the same way as source
  /// decls.
  bool loadedFromBytecode = false;

  /// The counter to allow the generation of unique IDs for this ASTDecl.
  unsigned counter = 0;

  friend class DeclResolver;
  friend class SharedState;
  ASTDecl(DeclIRValue irValue, llvm::SMLoc loc, ASTDecl *parentDecl,
          LexerCursor cursor, LexerCursor endCursor, ssize_t indentation)
      : irValue(irValue), loc(loc), parentDecl(parentDecl), cursor(cursor),
        endCursorState(endCursor.getState()), indentation(indentation) {}
  ASTDecl(const ASTDecl &) = delete;
  ASTDecl &operator=(const ASTDecl &) = delete;

private:
  /// This is the MLIR declaration that this scope corresponds to.
  DeclIRValue irValue;

  /// This is the source location of the declaration, used for diagnostics and
  /// debug information.
  llvm::SMLoc loc;

  /// For a type declaration like a struct, this is the type of 'self' in a
  /// member.  This is only valid after signature resolution.
  ASTType typeDeclSelf;

  /// This the parent scope that should continue name lookup, or null for the
  /// top scope.
  ASTDecl *parentDecl;

  /// This is the cursor that points to the next part of declaration to continue
  /// parsing as the declaration is progressively resolved.
  LexerCursor cursor;

  /// This is the lexer cursor state for the first token /after/ the
  /// declaration.  This is used to make sure that bits of a declaration are not
  /// skipped in the early parse and not processes in the later parse.
  const char *endCursorState;

  /// This is the indentation level of the introducer keyword, useful for
  /// parsing the body of the declaration.  If the declaration was not at the
  /// start of a line or this is the top level module, then this is set to -1.
  ssize_t indentation;

  /// When a bytecode decl depends on a source decl's children, we have to parse
  /// the signatures of all the children to register them in the symbol table.
  /// Cache this process using a flag on the decl.
  bool referencedFromBytecode = false;

  /// This is set to true when an error is detected and reported about this
  /// declaration that could cause references to it to cause spurious downstream
  /// errors.  For example, "var x : SomeUndeclaredType" will cause errors for
  /// every reference to 'x' because the type will be bogus.
  bool hasReferenceError = false;

  /// These are the declarations defined within this scope.
  llvm::MapVector<StringAttr, TinyPtrVector<ASTDecl *>> declsInScope;

  /// A set of modules with unresolved wildcard imports into this decl, mapped
  /// to the location of the import and whether it's a full import.
  llvm::MapVector<StringAttr, std::pair<SMLoc, bool>> unresolvedWildcardImports;
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
    auto *op = dyn_cast_or_null<mlir::Operation *>(decl.getIRValue());
    return op && T::classof(op);
  }
  static T doCast(M::KGEN::LIT::ASTDecl &decl) {
    return T(cast<mlir::Operation *>(decl.getIRValue()));
  }
};
template <typename T>
struct CastInfo<T, const M::KGEN::LIT::ASTDecl>
    : public ConstStrippingForwardingCast<T, const M::KGEN::LIT::ASTDecl,
                                          CastInfo<T, M::KGEN::LIT::ASTDecl>> {
};

/// Cast from an (const) ASTDecl * to a Decl operation type.
template <typename T>
struct CastInfo<T, M::KGEN::LIT::ASTDecl *>
    : public NullableValueCastFailed<T>,
      public DefaultDoCastIfPossible<T, M::KGEN::LIT::ASTDecl *,
                                     CastInfo<T, M::KGEN::LIT::ASTDecl *>> {
  // Provide isPossible here because here we have the const-stripping from
  // ConstStrippingCast.
  static bool isPossible(M::KGEN::LIT::ASTDecl *decl) {
    if (!decl)
      return false;
    auto *op = dyn_cast_or_null<mlir::Operation *>(decl->getIRValue());
    return op && T::classof(op);
  }
  static T doCast(M::KGEN::LIT::ASTDecl *decl) {
    return T(cast<mlir::Operation *>(decl->getIRValue()));
  }
};
template <typename T>
struct CastInfo<T, const M::KGEN::LIT::ASTDecl *>
    : public ConstStrippingForwardingCast<
          T, const M::KGEN::LIT::ASTDecl *,
          CastInfo<T, M::KGEN::LIT::ASTDecl *>> {};
} // namespace llvm

#endif // KGEN_MOJOPARSER_ASTDECL_H
