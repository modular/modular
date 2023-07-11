//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Declaration parsing and name binding logic.
//
//===----------------------------------------------------------------------===//

#ifndef DECLRESOLVER_H
#define DECLRESOLVER_H

#include "IRValues.h"
#include "SharedState.h"

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/LITDialect/SpecialFunctions.h"

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
class Lexer;
class LexerCursor;
class PackageOp;
class ParserBase;
class SharedState;
class UnresolvedImportOp;
class VarLetDeclOp;
class LetRegDeclOp;
class GlobalVarDeclOp;
class StructDeclOp;
class StructFieldOp;

//===----------------------------------------------------------------------===//
// DeclResolver
//===----------------------------------------------------------------------===//

/// This stores declaration references (e.g. vardecls, structdecls, funcdecls)
/// as operations.  It stores RValues for parameters and SSA values as an
/// RValue.
using DeclIRValue = PointerUnion<Operation *, PValue, SRValue, MRValue, SBValue,
                                 MBValue, SLValue>;

class DeclResolver : public SharedStateUser {
public:
  DeclResolver(SharedState &state);
  ~DeclResolver();

  /// Resolve all of the declarations that are visible, processing the entire
  /// translation unit.
  void resolveAll();

  /// Resolve all of the declarations that are defined within or referenced by
  /// the given container `decl`.
  void resolveAllReferencedFrom(ASTDecl &decl);

  /// Add a new declaration that needs to be resolved.
  ASTDecl &addDecl(Operation *decl, llvm::SMLoc loc, StringAttr baseName,
                   ASTDecl *parentDecl, LexerCursor cursor,
                   LexerCursor endCursor, ssize_t indentation);
  ASTDecl &addDecl(DeclIRValue decl, llvm::SMLoc loc, StringAttr baseName,
                   ASTDecl *parentDecl, LexerCursor cursor,
                   LexerCursor endCursor, ssize_t indentation);

  /// Add a pre-existing set of declarations as children of the specified
  /// context, using the provided alias name (which may differ from that of the
  /// decl).
  void aliasDecls(const TinyPtrVector<ASTDecl *> &decls, StringAttr name,
                  llvm::SMLoc aliasLoc, ASTDecl &context);
  /// Try to add a pre-existing set of declarations as children of the specified
  /// context, using the provided alias name (which may differ from that of the
  /// decl). Does not error on failure, but returns a failure result.
  LogicalResult tryAliasDecls(const TinyPtrVector<ASTDecl *> &decls,
                              StringAttr name, llvm::SMLoc aliasLoc,
                              ASTDecl &context);
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

  /// Fully resolve the declaration and everything in it.
  LogicalResult recursivelyResolveFully(ASTDecl &decl, llvm::SMLoc loc);

  /// Given the symbol for a lit type declaration, return the ASTDecl that
  /// corresponds to it.  This doesn't allow null symbols, so it always
  /// succeeds.
  ASTDecl &getDeclForTypeSymbol(SymbolRefAttr symbol) const {
    auto it = declForTypeSymbol.find(symbol);
#ifndef NDEBUG
    if (it == declForTypeSymbol.end())
      symbol.dump();
    assert(it != declForTypeSymbol.end() && "Unknown decl symbol!");
#endif
    return *it->second;
  }

  /// This registers the finalized function with the DeclResolver after its
  /// signature has been resolved and its mangled name is available.  This
  /// returns an existing function if there is a redefinition problem.
  Operation *finalizeFuncSignature(LIT::FuncOp funcOp, ASTDecl &decl);

  void registerAndCheckExport(StringRef aliasName, Location loc);
  void exportMain(ASTDecl &funcDecl);

  ASTDecl *getDeclForFuncSymbol(SymbolRefAttr attr) const {
    auto it = declForFuncSymbol.find(attr);
    return it != declForFuncSymbol.end() ? it->second : nullptr;
  }

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
  LogicalResult resolveSignature(VarLetDeclOp op, Lexer &lexer, ASTDecl &decl);
  ParseResult resolveBody(VarLetDeclOp op, Lexer &lexer, ASTDecl &decl);
  ParseResult resolveBody(LetRegDeclOp op, Lexer &lexer, ASTDecl &decl);
  LogicalResult resolveSignature(GlobalVarDeclOp op, Lexer &lexer,
                                 ASTDecl &decl);
  ParseResult resolveBody(GlobalVarDeclOp op, Lexer &lexer, ASTDecl &decl);
  LogicalResult resolveSignature(ParamDeclareOp op, Lexer &lexer,
                                 ASTDecl &decl);
  ParseResult resolveBody(ParamDeclareOp op, Lexer &lexer, ASTDecl &decl);
  ParseResult resolveBody(AliasForwardDeclOp op, Lexer &lexer, ASTDecl &decl);

private:
  /// Add a pre-existing set of declarations, which may optionally be imported
  /// from a given module, as children of the specified context, using the
  /// provided alias name (which may differ from that of the decl).
  LogicalResult aliasDeclsImpl(const TinyPtrVector<ASTDecl *> &decls,
                               StringAttr name, llvm::SMLoc aliasLoc,
                               ASTDecl &context, bool emitDiagnostics = true,
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
  /// source buffer, we don't need to use StringMap here.
  llvm::StringMap<Location> exportedSymbolNames;

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

//===----------------------------------------------------------------------===//
// Argument and Parameter List Parsing
//===----------------------------------------------------------------------===//

/// Specify variadic argument kind, e.g. `*x` or `**x`.
enum VarArgKind {
  /// Not a variadic argument, e.g. `x` or `x: Int`.
  None,
  /// A homogeneously typed variadic argument, e.g. `*x` or `*x: Int`.
  VarArg,
  /// A heterogeneously typed variadic argument, e.g. `*x: *Ts`.
  PackVarArg,
  /// A variadic keywords argument, e.g. `**x`.
  KWVarArg
};

/// Parsing support for a function argument and input parameter:
///
/// argument_list      ::= argument ("," argument)*
/// argument           ::= "/" | "*"
/// argument           ::= [argument_ownership] [argument_variadic] identifier
///                        [argument_reference] [argument_type] ["=" expression]
/// argument_ownership ::= "owned" | "borrowed"
/// argument_variadic  ::= "*" | "**"
/// argument_reference ::= "&"
/// argument_type      ::= ":" star_expression
struct ParsedArgument {
  SMLoc loc;
  // Specify argument passing convention, e.g. owned/byref etc.
  enum {
    kConventionUnspec = 0,         // Nothing specified
    kConventionInOut = 1,          // x&
    kConventionOwned = 2,          // owned x
    kConventionBorrowed = 3,       // borrowed x
    kConventionInOutResult = 4,    // No syntax: result slot
    kConventionInitSelfResult = 5, // No syntax: __init__(inout self) argument
  } convention = kConventionUnspec;

  // After type checking, this will hold the KGEN convention to use.
  ValueInputConvention kgenConvention = ValueInputConvention(128);

  VarArgKind vararg = VarArgKind::None;
  StringAttr name;
  const ExprNode *typeExpr = nullptr;
  ExprNode *initExpr = nullptr;

  /// This gets set to true when there is a /diagnosed/ error that should
  /// prevent subsequent references to this argument.
  bool isErroneous = false;

  /// This specifies the handling of keyword arguments in a list.
  enum class KWArgHandling {
    kPositionalOnly,      //< before a standalone '/'
    kPositionalOrKeyword, //< before a standalone '*'
    kKeywordOnly          //< after a standalone '*'
  } kwArgHandling = KWArgHandling::kPositionalOrKeyword;

  enum class KWArgMarkerInfo {
    kNotMarker, //< This is a normal argument.
    kSlash,     //< This argument is a standalone '/' marker.
    kStar,      //< This argument is a standalone '*' marker.
  };

  ParseResult parse(ParserBase &p, KWArgMarkerInfo &markerInfo,
                    bool omitName = false);

  /// This method handles the function argument list for a Python function.
  /// Python has some pretty interesting rules where standalone '*' and '/'
  /// markers (when used in place of an argument) actually change the
  /// interpretation of other argument definitions by specifying how they behave
  /// w.r.t. keyword arguments.  We resolve these here so the client doesn't
  /// have to deal with them.
  ///
  /// This classification logic is described here:
  ///   https://peps.python.org/pep-0570/#how-to-teach-this
  ///
  static ParseResult parseAndResolvePresentArgumentList(
      ParserBase &p, SmallVectorImpl<ParsedArgument> &args,
      bool isParameterList, bool omitNames = false);

  /// Process parsed parameter arguments into input or result parameters by
  /// determining the correct parameter types and conventions.
  static void processParameterArgs(ExprEmitter &emitter, ASTDecl &declScope,
                                   ArrayRef<ParsedArgument> args,
                                   SmallVectorImpl<ParamDeclAttr> &params,
                                   bool isResultParams, bool &paramVararg);

  /// Emit the argument types, default values, and result type and determine
  /// the argument conventions.
  static ASTType emitFunctionArgumentsAndResults(
      function_ref<ParseResult()> reportError, SharedState &shared,
      ExprEmitter &typeEmitter, const ExprNode *resultTypeExpr,
      FnEffects &effects, SmallVectorImpl<ParsedArgument> &args,
      SmallVectorImpl<Type> &argTypes, SmallVectorImpl<TypedAttr> &defaults,
      bool isDef, SMLoc resultLoc, ASTDecl &Scope,
      SpecialFunctionInfo fnInfo = SpecialFunctionInfo(),
      StringRef funcName = "");

  /// Given a fully resolved signature, compute the final types and KGEN input
  /// conventions of the arguments.
  static void computeArgumentConventions(SharedState &shared,
                                         MutableArrayRef<ParsedArgument> args,
                                         MutableArrayRef<Type> argTypes);
};

} // namespace M::KGEN::LIT

#endif // DECLRESOLVER_H
