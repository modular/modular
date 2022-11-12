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

#include "IRValues.h"
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
class ASTDecl;

//===----------------------------------------------------------------------===//
// DeclResolver
//===----------------------------------------------------------------------===//

/// This stores declaration references (e.g. vardecls, structdecls, funcdecls)
/// as operations.  It stores RValues for parameters and SSA values as an
/// RValue.
using DeclIRValue = PointerUnion<Operation *, MAValue, DRValue, LValue>;

class DeclResolver {
public:
  DeclResolver(LitSharedState &state);
  ~DeclResolver();

  MLIRContext *getContext() const { return sharedState.context; }

  /// Resolve all of the declarations that are visible, processing the entire
  /// translation unit.
  void resolveAll(llvm::SMLoc loc);

  /// Add a new declaration that needs to be resolved.
  ASTDecl &addDecl(Operation *decl, StringAttr name, ASTDecl *parentDecl,
                   LitLexerCursor cursor, LitLexerCursor endCursor,
                   ssize_t indentation);

  /// Add a declaration that is already fully resolved.
  ASTDecl &addFullyResolvedDecl(Operation *decl, StringAttr name, ASTType type,
                                ASTDecl *parentDecl);

  /// Add a declaration that is already fully resolved.
  ASTDecl &addFullyResolvedDecl(DeclIRValue declVal, StringAttr name,
                                Location loc, ASTType type,
                                ASTDecl *parentDecl);

  /// Add a "magic" declaration that has special handling to this scope.  This
  /// is used for builtin machinery internal to the language.
  ASTDecl &addMagicDecl(StringRef name, MagicDeclKind kind,
                        ASTDecl *parentDecl);

  /// Resolve the specified declaration to at least the specified level of
  /// resolution, performing incremental type checking as appropriate.
  LogicalResult resolve(ASTDecl &decl, DeclResolvedness howResolved,
                        llvm::SMLoc loc);

private:
  ASTDecl &addDecl(DeclIRValue decl, Location loc, StringAttr name,
                   ASTDecl *parentDecl, LitLexerCursor cursor,
                   LitLexerCursor endCursor, ssize_t indentation);

  /// The resolveSignature methods are invoked on an operation to parse and type
  /// check the signature for the operation.  On parse failure, these should
  /// return a failure, which will cause the driver to mark the decl as invalid
  /// for further references.
  LogicalResult resolveSignature(Operation *op, LitLexer &lexer, ASTDecl &decl);
  ParseResult resolveBody(LITFuncOp op, LitLexer &lexer, ASTDecl &decl);

  LogicalResult resolveSignature(LITStructDeclOp op, LitLexer &lexer,
                                 ASTDecl &decl);
  ParseResult resolveBody(LITStructDeclOp op, LitLexer &lexer, ASTDecl &decl);

  LogicalResult resolveSignature(VarDeclOp op, LitLexer &lexer, ASTDecl &decl);
  ParseResult resolveBody(VarDeclOp op, LitLexer &lexer, ASTDecl &decl);

private:
  /// This is shared state across the whole parser.
  LitSharedState &sharedState;

  /// This array holds all of the parsed declarations in a deterministic order.
  std::vector<ASTDecl *> parsedDeclList;

  /// Name binding is an recursive process in the general case.  This keeps
  /// track of the declarations currently being name bound so we can diagnose
  /// cyclic dependencies.
  DenseSet<ASTDecl *> declsCurrentlyProcessing;

  DeclResolver(const DeclResolver &) = delete;
  DeclResolver &operator=(const DeclResolver &) = delete;
};

} // namespace M::KGEN::LIT

#endif // LITDECLS_H
