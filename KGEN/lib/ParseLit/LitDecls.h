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

#include "LitSharedState.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace M::KGEN {
class LITFuncOp;
class LITStructDeclOp;
class ParamBindArrayAttr;
class ParamDeclAttr;
class VarDeclOp;
} // namespace M::KGEN

namespace M::KGEN::LIT {
class LitLexer;
class LitLexerCursor;
class LitParserBase;
class LitSharedState;
class DeclAST;

//===----------------------------------------------------------------------===//
// DeclResolver
//===----------------------------------------------------------------------===//

class DeclResolver {
public:
  DeclResolver(LitSharedState &state);
  ~DeclResolver();

  MLIRContext *getContext() const { return sharedState.context; }

  /// Resolve all of the declarations that are visible, processing the entire
  /// translation unit.
  void resolveAll(llvm::SMLoc loc);

  /// Add a new declaration that needs to be resolved.
  DeclAST &addDecl(Operation *decl, DeclAST *parentDecl, LitLexerCursor cursor,
                   LitLexerCursor endCursor, ssize_t indentation);

  /// Add a declaration that is already fully resolved.
  DeclAST &addFullyResolvedDecl(Operation *decl, DeclAST *parentDecl);

  /// Add a declaration that is already fully resolved.
  DeclAST &addFullyResolvedDecl(ParamDeclAttr decl, Location loc,
                                DeclAST *parentDecl);

  /// Add a "magic" declaration that has special handling to this scope.  This
  /// is used for builtin machinery internal to the language.
  DeclAST &addMagicDecl(StringRef name, MagicDeclKind kind,
                        DeclAST *parentDecl);

  /// If the specified type is a RefType that resolves to a (possibly
  /// parameterized) type, return the decl for the type and the parameters in
  /// the reference.  This returns null on error.
  std::pair<DeclAST *, ParamBindArrayAttr> getDeclAndParamsFromType(Type type);

  /// Resolve the specified declaration to at least the specified level of
  /// resolution, performing incremental type checking as appropriate.
  LogicalResult resolve(DeclAST &decl, DeclResolvedness howResolved,
                        llvm::SMLoc loc);

private:
  DeclAST &addDecl(PointerUnion<Operation *, Attribute> decl, Location loc,
                   StringAttr name, DeclAST *parentDecl, LitLexerCursor cursor,
                   LitLexerCursor endCursor, ssize_t indentation);

  /// The resolveSignature methods are invoked on an operation to parse and type
  /// check the signature for the operation.  On parse failure, these should
  /// return a failure, which will cause the driver to mark the decl as invalid
  /// for further references.
  LogicalResult resolveSignature(LITFuncOp op, LitLexer &lexer, DeclAST &decl);
  ParseResult resolveBody(LITFuncOp op, LitLexer &lexer, DeclAST &decl);

  LogicalResult resolveSignature(LITStructDeclOp op, LitLexer &lexer,
                                 DeclAST &decl);
  ParseResult resolveBody(LITStructDeclOp op, LitLexer &lexer, DeclAST &decl);

  LogicalResult resolveSignature(VarDeclOp op, LitLexer &lexer, DeclAST &decl);
  ParseResult resolveBody(VarDeclOp op, LitLexer &lexer, DeclAST &decl);

private:
  /// This is shared state across the whole parser.
  LitSharedState &sharedState;

  /// This is a mapping of MLIR symbol to decl for types.
  DenseMap<StringAttr, DeclAST *> typeSymbolDecls;

  /// This array holds all of the parsed declarations in a deterministic order.
  std::vector<DeclAST *> parsedDeclList;

  /// Name binding is an recursive process in the general case.  This keeps
  /// track of the declarations currently being name bound so we can diagnose
  /// cyclic dependencies.
  DenseSet<DeclAST *> declsCurrentlyProcessing;

  DeclResolver(const DeclResolver &) = delete;
  DeclResolver &operator=(const DeclResolver &) = delete;
};

} // namespace M::KGEN::LIT

#endif // LITDECLS_H
