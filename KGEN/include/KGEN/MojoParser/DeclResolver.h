//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Declaration parsing and name binding logic.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_DECLRESOLVER_H
#define KGEN_MOJOPARSER_DECLRESOLVER_H

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/LITDialect/SpecialFunctions.h"
#include "KGEN/MojoParser/IRValues.h"
#include "KGEN/MojoParser/Lexer.h"
#include "KGEN/MojoParser/SharedState.h"
#include "Support/RCRef.h"

namespace M::KGEN {
class ParamDeclAttr;
} // namespace M::KGEN

namespace M::KGEN::LIT {
class AliasDeclOp;
class ASTDecl;
class FileModuleOp;
class FuncOp;
class PackageOp;
class ParserBase;
class SharedState;
class UnresolvedImportOp;
class GlobalVarDeclOp;
class StructDeclOp;
class StructFieldOp;
class TraitDeclOp;
struct ParsedArgument;
class BaseDLValue;
enum class PassingKind : uint32_t;

//===----------------------------------------------------------------------===//
// DeclResolver
//===----------------------------------------------------------------------===//

/// This stores declaration references (e.g. vardecls, structdecls, funcdecls)
/// as operations.  It stores RValues for parameters and SSA values as an
/// RValue.
using DeclIRValue = SmartVariant<Operation *, CValue>;

class DeclResolver : public SharedStateUser {
public:
  DeclResolver(SharedState &state);
  ~DeclResolver();

  //===--------------------------------------------------------------------===//
  // Decl Constructors
  //===--------------------------------------------------------------------===//

  /// Add a new declaration that needs to be resolved.
  ASTDecl &addDecl(DeclIRValue irValue, llvm::SMLoc loc, StringAttr baseName,
                   ASTDecl *parentDecl, LexerCursor cursor,
                   LexerCursor endCursor, ssize_t indentation);

  /// Add a declaration that is already fully resolved.
  ASTDecl &addFullyResolvedDecl(DeclIRValue declVal, StringAttr baseName,
                                llvm::SMLoc loc, ASTDecl *parentDecl);
  ASTDecl &addFullyResolvedDecl(DeclIRValue declVal, StringRef baseName,
                                llvm::SMLoc loc, ASTDecl *parentDecl);

  /// Add a declaration that represents an erroneous declaration. The generated
  /// decl is treated as fully resolved, and in an error state.
  ASTDecl &addErroneousDecl(StringRef baseName, llvm::SMLoc loc,
                            ASTDecl *parentDecl);

  /// Add a new declaration that needs to be resolved, but don't attach it to
  /// parent's name table.  It needs to be added later.
  ASTDecl &createUnlistedDecl(DeclIRValue irValue, llvm::SMLoc loc,
                              ASTDecl *parentDecl, LexerCursor cursor,
                              LexerCursor endCursor, ssize_t indentation);
  ASTDecl &createUnlistedDecl(Operation *decl, llvm::SMLoc loc,
                              ASTDecl *parentDecl, LexerCursor cursor,
                              LexerCursor endCursor, ssize_t indentation);

  /// Attach a declaration to its parent's name table.  For use with
  /// `makeUnlistedDecl`.
  void attachDeclToParentNameTable(ASTDecl *decl, StringAttr name);

  //===--------------------------------------------------------------------===//
  // Import Resolution
  //===--------------------------------------------------------------------===//

  /// Add a pre-existing set of declarations as children of the specified
  /// context, using the provided alias name (which may differ from that of the
  /// decl).
  void aliasDecls(ArrayRef<ASTDecl *> decls, StringAttr name,
                  llvm::SMLoc aliasLoc, ASTDecl &context);
  /// Try to add a pre-existing set of declarations as children of the specified
  /// context, using the provided alias name (which may differ from that of the
  /// decl). Does not error on failure, but returns a failure result.
  LogicalResult tryAliasDecls(ArrayRef<ASTDecl *> decls, StringAttr name,
                              llvm::SMLoc aliasLoc, ASTDecl &context);
  /// Add a pre-existing set of declarations imported from the given module, as
  /// children of the specified context, using the provided alias name (which
  /// may differ from that of the decl).
  LogicalResult aliasImportDecls(ArrayRef<ASTDecl *> decls, StringAttr name,
                                 StringAttr declName, StringAttr moduleName,
                                 llvm::SMLoc aliasLoc, ASTDecl &context);

private:
  /// Add a pre-existing set of declarations, which may optionally be imported
  /// from a given module, as children of the specified context, using the
  /// provided alias name (which may differ from that of the decl).
  LogicalResult aliasDeclsImpl(ArrayRef<ASTDecl *> decls, StringAttr name,
                               llvm::SMLoc aliasLoc, ASTDecl &context,
                               bool emitDiagnostics = true,
                               StringAttr moduleName = StringAttr(),
                               StringAttr declNameInModule = StringAttr());

public:
  /// Import the given module into the provided destination.
  LogicalResult importModule(ASTDecl &dest, PackageOp currentPackage,
                             StringAttr moduleName, StringAttr importName,
                             SMLoc loc, SMLoc importNameLoc);
  /// Import the provided decl from the given module decl, into the provided
  /// destination.
  LogicalResult importDeclFromModule(ASTDecl &dest, PackageOp currentPackage,
                                     StringAttr moduleName,
                                     StringAttr sourceName, StringAttr destName,
                                     SMLoc loc, SMLoc sourceNameLoc,
                                     SMLoc destNameLoc);
  /// Import decls from the given module into the provided destination context
  /// using a wild-card import. If `isFullImport` is true, all decls are
  /// imported, otherwise only decls that don't start with an `_` are imported.
  LogicalResult importWildCardDeclsFromModule(ASTDecl &context,
                                              StringAttr moduleName,
                                              bool isFullImport,
                                              llvm::SMLoc loc);

  //===--------------------------------------------------------------------===//
  // Decl Resolution
  //===--------------------------------------------------------------------===//

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

  //===--------------------------------------------------------------------===//
  // Top-Level Decl Resolution
  //===--------------------------------------------------------------------===//

  /// Resolve all of the declarations that are defined within or referenced by
  /// the given container `decl`. If `eraseUnparsedDecls` is true, decls that
  /// were not referenced at all during parsing are erased.
  void resolveAllReferencedFrom(ASTDecl &decl, bool eraseUnparsedDecls = true);

  /// Resolve the pending wildcard imports in the decl if it represents a
  /// module.
  LogicalResult resolveAllWildcardImports(ASTDecl &module);

  //===--------------------------------------------------------------------===//
  // Symbol-ASTDecl Mapping
  //===--------------------------------------------------------------------===//

  /// Given the symbol for a declaration, return the ASTDecl that corresponds to
  /// it.  This doesn't allow null symbols, so it always succeeds.
  ASTDecl &getDeclForTypeSymbol(SymbolRefAttr symbol) const;
  ASTDecl *getDeclForFuncSymbol(SymbolRefAttr attr) const;

  /// This registers the finalized function with the DeclResolver after its
  /// signature has been resolved and its mangled name is available.  This
  /// returns an existing function if there is a redefinition problem.
  Operation *finalizeFuncSignature(LIT::FuncOp funcOp, ASTDecl &decl);

  //===--------------------------------------------------------------------===//
  // Export Handling
  //===--------------------------------------------------------------------===//

  void registerAndCheckExport(StringRef aliasName, SMLoc loc);
  void exportMain(ASTDecl &funcDecl);

  //===--------------------------------------------------------------------===//
  // Decl Helpers
  //===--------------------------------------------------------------------===//

  /// Create a name from a signature by appending argument types into the name.
  static StringAttr getMangledName(StringAttr baseName, ASTDecl &container,
                                   LITSignatureType signature);

  /// Given a signature type that may contain references to parameter
  /// declarations in a parent context, isolate it by creating a signatuer with
  /// no external references by inserting an parameter for every captured
  /// parameter declaration. Return the captured parameter references.
  static std::pair<SmallVector<ParamDeclRefAttr>, LITSignatureType>
  createSelfContainedSignature(LITSignatureType original);

private:
  /// The resolveSignature methods are invoked on an operation to parse and type
  /// check the signature for the operation.  On parse failure, these should
  /// return a failure, which will cause the driver to mark the decl as invalid
  /// for further references.
  LogicalResult resolveSignature(LIT::FuncOp op, Lexer &lexer, ASTDecl &decl);
  ParseResult resolveBody(LIT::FuncOp op, Lexer &lexer, ASTDecl &decl);

  ParseResult resolveBody(LIT::FileModuleOp op, Lexer &lexer, ASTDecl &decl);
  ParseResult resolveBody(PackageOp op, ASTDecl &decl);

  ParseResult resolveSignature(LIT::UnresolvedImportOp op, ASTDecl &decl);

  LogicalResult resolveSignature(StructDeclOp op, Lexer &lexer, ASTDecl &decl);
  ParseResult resolveBody(StructDeclOp op, Lexer &lexer, ASTDecl &decl);
  LogicalResult resolveSignature(StructFieldOp op, Lexer &lexer, ASTDecl &decl);
  ParseResult resolveBody(StructFieldOp op, Lexer &lexer, ASTDecl &decl);
  LogicalResult resolveSignature(TraitDeclOp op, Lexer &lexer, ASTDecl &decl);
  ParseResult resolveBody(TraitDeclOp op, Lexer &lexer, ASTDecl &decl);
  LogicalResult resolveSignature(GlobalVarDeclOp op, Lexer &lexer,
                                 ASTDecl &decl);
  ParseResult resolveBody(GlobalVarDeclOp op, Lexer &lexer, ASTDecl &decl);
  LogicalResult resolveSignature(AliasDeclOp op, Lexer &lexer, ASTDecl &decl);
  ParseResult resolveBody(AliasDeclOp op, Lexer &lexer, ASTDecl &decl);

  /// This map tracks the ASTDecl for every MLIR type declaration with a symbol.
  /// This does not include functions, only things that may be referred to by a
  /// StructType: StructTypes, aliases, etc.
  DenseMap<SymbolRefAttr, ASTDecl *> declForTypeSymbol;

  /// This map tracks the ASTDecl for every LIT::FuncOp, allowing clients to map
  /// from MLIR symbol references to their body and AST information.  This is
  /// populated during signature resolution, since the symbol will be mangled.
  DenseMap<SymbolRefAttr, ASTDecl *> declForFuncSymbol;

  /// This map tracks the exported function names and their locations so that
  /// we can check if they are unique.
  /// Note: these StringRef keys cannot dangle because they point to the parsed
  /// source buffer, we don't need to use StringMap here.
  llvm::StringMap<SMLoc> exportedSymbolNames;

  /// This array holds all of the parsed declarations in a deterministic order.
  std::vector<ASTDecl *> parsedDeclList;

  /// Name binding is an recursive process in the general case.  This keeps
  /// track of the declarations currently being name bound so we can diagnose
  /// cyclic dependencies.
  DenseMap<ASTDecl *, llvm::SMLoc> declsCurrentlyProcessing;

  /// Allow access to private fields.
  friend SharedState;

  DeclResolver(const DeclResolver &) = delete;
  DeclResolver &operator=(const DeclResolver &) = delete;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_DECLRESOLVER_H
