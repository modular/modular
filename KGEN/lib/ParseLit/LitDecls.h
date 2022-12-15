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
class StructDeclOp;
class StructFieldOp;
class ParamBindArrayAttr;
class ParamDeclAttr;
class ParamDeclRefAttr;
class ParamDeclareOp;
} // namespace M::KGEN

namespace M::KGEN::LIT {
class ASTDecl;
class FuncOp;
class LitLexer;
class LitLexerCursor;
class LitParserBase;
class LitSharedState;
class VarDeclOp;

//===----------------------------------------------------------------------===//
// DeclResolver
//===----------------------------------------------------------------------===//

/// This stores declaration references (e.g. vardecls, structdecls, funcdecls)
/// as operations.  It stores RValues for parameters and SSA values as an
/// RValue.
using DeclIRValue = PointerUnion<Operation *, MValue, DRValue, LValue>;

class DeclResolver {
public:
  DeclResolver(LitSharedState &state);
  ~DeclResolver();

  MLIRContext *getContext() const { return sharedState.context; }

  /// Resolve all of the declarations that are visible, processing the entire
  /// translation unit.
  void resolveAll();

  /// Add a new declaration that needs to be resolved.
  ASTDecl &addDecl(Operation *decl, llvm::SMLoc loc, StringAttr name,
                   ASTDecl *parentDecl, LitLexerCursor decoratorsCursor,
                   LitLexerCursor cursor, LitLexerCursor endCursor,
                   ssize_t indentation);
  ASTDecl &addDecl(DeclIRValue decl, llvm::SMLoc loc, StringAttr name,
                   ASTDecl *parentDecl, LitLexerCursor decoratorsCursor,
                   LitLexerCursor cursor, LitLexerCursor endCursor,
                   ssize_t indentation);

  /// Add a declaration that is already fully resolved.
  ASTDecl &addFullyResolvedDecl(Operation *decl, llvm::SMLoc loc,
                                StringAttr name, ASTDecl *parentDecl);

  /// Add a declaration that is already fully resolved.
  ASTDecl &addFullyResolvedDecl(DeclIRValue declVal, StringAttr name,
                                llvm::SMLoc loc, ASTDecl *parentDecl);
  ASTDecl &addFullyResolvedDecl(DeclIRValue declVal, StringRef name,
                                llvm::SMLoc loc, ASTDecl *parentDecl);

  /// Resolve the specified declaration to at least the specified level of
  /// resolution, performing incremental type checking as appropriate.
  LogicalResult resolve(ASTDecl &decl, DeclResolvedness howResolved,
                        llvm::SMLoc loc);

  /// Given the symbol for a lit type declaration, return the ASTDecl that
  /// corresponds to it.  This doesn't allow null symbols, so it always
  /// succeeds.
  ASTDecl &getDeclForTypeSymbol(SymbolRefAttr symbol) const {
    auto it = declForTypeSymbol.find(symbol);
    assert(it != declForTypeSymbol.end() && "Unknown decl symbol!");
    return *it->second;
  }

private:
  /// The resolveSignature methods are invoked on an operation to parse and type
  /// check the signature for the operation.  On parse failure, these should
  /// return a failure, which will cause the driver to mark the decl as invalid
  /// for further references.
  LogicalResult resolveSignature(ParamDeclRefAttr paramDeclRef, LitLexer &lexer,
                                 ASTDecl &decl);
  ParseResult resolveBody(ParamDeclRefAttr paramDeclRef, LitLexer &lexer,
                          ASTDecl &decl);
  LogicalResult resolveSignature(LIT::FuncOp op, LitLexer &lexer,
                                 ASTDecl &decl);
  ParseResult resolveBody(LIT::FuncOp op, LitLexer &lexer, ASTDecl &decl);

  LogicalResult resolveSignature(StructDeclOp op, LitLexer &lexer,
                                 ASTDecl &decl);
  ParseResult resolveBody(StructDeclOp op, LitLexer &lexer, ASTDecl &decl);

  LogicalResult resolveSignature(StructFieldOp op, LitLexer &lexer,
                                 ASTDecl &decl);
  ParseResult resolveBody(StructFieldOp op, LitLexer &lexer, ASTDecl &decl);

  LogicalResult resolveSignature(VarDeclOp op, LitLexer &lexer, ASTDecl &decl);
  ParseResult resolveBody(VarDeclOp op, LitLexer &lexer, ASTDecl &decl);
  LogicalResult resolveSignature(ParamDeclareOp op, LitLexer &lexer,
                                 ASTDecl &decl);
  ParseResult resolveBody(ParamDeclareOp op, LitLexer &lexer, ASTDecl &decl);

private:
  ParseResult processDecorators(LIT::FuncOp funcOp, ASTDecl &decl);

  /// This is shared state across the whole parser.
  LitSharedState &sharedState;

  /// This map tracks the ASTDecl for every MLIR type declaration with a symbol.
  /// This does not include functions, only things that may be referred to by a
  /// DeclRefType: StructTypes, aliases, etc.
  DenseMap<SymbolRefAttr, ASTDecl *> declForTypeSymbol;

  /// This array holds all of the parsed declarations in a deterministic order.
  std::vector<ASTDecl *> parsedDeclList;

  /// Name binding is an recursive process in the general case.  This keeps
  /// track of the declarations currently being name bound so we can diagnose
  /// cyclic dependencies.
  DenseMap<ASTDecl *, llvm::SMLoc> declsCurrentlyProcessing;

  DeclResolver(const DeclResolver &) = delete;
  DeclResolver &operator=(const DeclResolver &) = delete;
};

} // namespace M::KGEN::LIT

#endif // LITDECLS_H
