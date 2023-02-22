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
class ParamBindArrayAttr;
class ParamDeclAttr;
class ParamDeclareOp;
class ExportOp;
} // namespace M::KGEN

namespace M::KGEN::LIT {
class AliasForwardDeclOp;
class ASTDecl;
class FileModuleOp;
class FuncOp;
class LitLexer;
class LitLexerCursor;
class LitParserBase;
class LitSharedState;
class LetDeclOp;
class UnresolvedImportOp;
class VarDeclOp;
class StructDeclOp;
class StructFieldOp;

//===----------------------------------------------------------------------===//
// DeclResolver
//===----------------------------------------------------------------------===//

/// This stores declaration references (e.g. vardecls, structdecls, funcdecls)
/// as operations.  It stores RValues for parameters and SSA values as an
/// RValue.
using DeclIRValue = PointerUnion<Operation *, MValue, DRValue, LValue>;

class DeclResolver : public LitSharedStateUser {
public:
  DeclResolver(LitSharedState &state);
  ~DeclResolver();

  /// Resolve all of the declarations that are visible, processing the entire
  /// translation unit.
  void resolveAll();

  /// Add a new declaration that needs to be resolved.
  ASTDecl &addDecl(Operation *decl, llvm::SMLoc loc, StringAttr baseName,
                   ASTDecl *parentDecl, LitLexerCursor cursor,
                   LitLexerCursor endCursor, ssize_t indentation);
  ASTDecl &addDecl(DeclIRValue decl, llvm::SMLoc loc, StringAttr baseName,
                   ASTDecl *parentDecl, LitLexerCursor cursor,
                   LitLexerCursor endCursor, ssize_t indentation);

  /// Add a pre-existing set of declarations as children of the specified
  /// context, using the provided alias name (which may differ from that of the
  /// decl).
  void aliasDecls(const TinyPtrVector<ASTDecl *> &decls, StringAttr name,
                  llvm::SMLoc aliasLoc, ASTDecl &context);
  /// Add a pre-existing set of declarations imported from the given module, as
  /// children of the specified context, using the provided alias name (which
  /// may differ from that of the decl).
  LogicalResult aliasImportDecls(const TinyPtrVector<ASTDecl *> &decls,
                                 StringAttr name, StringAttr declName,
                                 StringAttr moduleName, llvm::SMLoc aliasLoc,
                                 ASTDecl &context);

  /// Import the given module into the provided context.
  LogicalResult importModule(ASTDecl &context, StringAttr moduleName,
                             StringAttr importName, SMLoc loc);
  /// Import the provided decl from the given module decl, into the provided
  /// destination context.
  LogicalResult importDeclFromModule(ASTDecl &context, StringAttr moduleName,
                                     StringAttr sourceName, StringAttr destName,
                                     SMLoc loc);
  /// Import decls from the given module into the provided destination context
  /// using a wild-card import (i.e. import all decls that don't start with an
  /// `_`).
  LogicalResult importWildCardDeclsFromModule(ASTDecl &context,
                                              StringAttr moduleName,
                                              llvm::SMLoc loc);

  /// Add a declaration that is already fully resolved.
  ASTDecl &addFullyResolvedDecl(Operation *decl, llvm::SMLoc loc,
                                StringAttr baseName, ASTDecl *parentDecl);

  /// Add a declaration that is already fully resolved.
  ASTDecl &addFullyResolvedDecl(DeclIRValue declVal, StringAttr baseName,
                                llvm::SMLoc loc, ASTDecl *parentDecl);
  ASTDecl &addFullyResolvedDecl(DeclIRValue declVal, StringRef baseName,
                                llvm::SMLoc loc, ASTDecl *parentDecl);

  /// Add a declaration that represents an erroneous declaration. The generated
  /// decl is treated as fully resolved, and in an error state.
  ASTDecl &addErroneousDecl(StringRef baseName, llvm::SMLoc loc,
                            ASTDecl *parentDecl);

  /// Resolve the specified declaration to at least the specified level of
  /// resolution, performing incremental type checking as appropriate.
  LogicalResult resolve(ASTDecl &decl, DeclResolvedness howResolved,
                        llvm::SMLoc loc);
  LogicalResult resolveSignature(ASTDecl &decl, llvm::SMLoc loc) {
    return resolve(decl, DeclResolvedness::signature, loc);
  }
  LogicalResult resolveFully(ASTDecl &decl, llvm::SMLoc loc) {
    return resolve(decl, DeclResolvedness::fully, loc);
  }

  /// Given the symbol for a lit type declaration, return the ASTDecl that
  /// corresponds to it.  This doesn't allow null symbols, so it always
  /// succeeds.
  ASTDecl &getDeclForTypeSymbol(SymbolRefAttr symbol) const {
    auto it = declForTypeSymbol.find(symbol);
    assert(it != declForTypeSymbol.end() && "Unknown decl symbol!");
    return *it->second;
  }

  void registerAndCheckExport(ExportOp exportOp);
  void exportMain(ASTDecl *containingDecl, SymbolRefAttr symbolName);

private:
  /// The resolveSignature methods are invoked on an operation to parse and type
  /// check the signature for the operation.  On parse failure, these should
  /// return a failure, which will cause the driver to mark the decl as invalid
  /// for further references.
  LogicalResult resolveSignature(LIT::FuncOp op, LitLexer &lexer,
                                 ASTDecl &decl);
  ParseResult resolveBody(LIT::FuncOp op, LitLexer &lexer, ASTDecl &decl);

  ParseResult resolveBody(LIT::FileModuleOp op, LitLexer &lexer, ASTDecl &decl);

  ParseResult resolveSignature(LIT::UnresolvedImportOp op, LitLexer &lexer,
                               ASTDecl &decl);

  LogicalResult resolveSignature(StructDeclOp op, LitLexer &lexer,
                                 ASTDecl &decl);
  ParseResult resolveBody(StructDeclOp op, LitLexer &lexer, ASTDecl &decl);
  LogicalResult resolveSignature(StructFieldOp op, LitLexer &lexer,
                                 ASTDecl &decl);
  ParseResult resolveBody(StructFieldOp op, LitLexer &lexer, ASTDecl &decl);
  LogicalResult resolveSignature(LetDeclOp op, LitLexer &lexer, ASTDecl &decl);
  ParseResult resolveBody(LetDeclOp op, LitLexer &lexer, ASTDecl &decl);
  LogicalResult resolveSignature(VarDeclOp op, LitLexer &lexer, ASTDecl &decl);
  ParseResult resolveBody(VarDeclOp op, LitLexer &lexer, ASTDecl &decl);
  LogicalResult resolveSignature(ParamDeclareOp op, LitLexer &lexer,
                                 ASTDecl &decl);
  ParseResult resolveBody(ParamDeclareOp op, LitLexer &lexer, ASTDecl &decl);
  ParseResult resolveBody(AliasForwardDeclOp op, LitLexer &lexer,
                          ASTDecl &decl);

  /// A valid main function must have signature main().
  /// No parameters are allowed and here must be only one main in the final
  /// object file.
  bool isMainFunction(StringAttr &name,
                      SmallVectorImpl<ParamDeclAttr> &inputParamDecls,
                      SmallVectorImpl<ParamDeclAttr> &resultParamDecls,
                      MutableArrayRef<Type> argTypes, ASTType &resultType);

private:
  /// Add a pre-existing set of declarations, which may optionally be imported
  /// from a given module, as children of the specified context, using the
  /// provided alias name (which may differ from that of the decl).
  LogicalResult aliasDeclsImpl(const TinyPtrVector<ASTDecl *> &decls,
                               StringAttr name, llvm::SMLoc aliasLoc,
                               ASTDecl &context,
                               StringAttr moduleName = StringAttr(),
                               StringAttr declNameInModule = StringAttr());

  /// This map tracks the ASTDecl for every MLIR type declaration with a symbol.
  /// This does not include functions, only things that may be referred to by a
  /// DeclRefType: StructTypes, aliases, etc.
  DenseMap<SymbolRefAttr, ASTDecl *> declForTypeSymbol;

  /// This map tracks the ASTDecl for every LIT::FuncOp, allowing clients to map
  /// from MLIR symbol references to their body and AST information.  This is
  /// populated during signature resolution, since the symbol will be mangled.
  DenseMap<SymbolRefAttr, ASTDecl *> declForFuncSymbol;

  /// This map tracks the exported function names and their locations so that
  /// we can check if they are unique.
  /// Note: these StringRef keys cannot dangle because they point to the parsed
  //  source buffer, we don't need to use StringMap here.
  DenseMap<StringRef, Location> exportedSymbolNames;

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
