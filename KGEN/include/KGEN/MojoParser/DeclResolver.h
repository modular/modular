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

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/LITDialect/SpecialFunctions.h"
#include "KGEN/MojoParser/IRValues.h"
#include "KGEN/MojoParser/Lexer.h"
#include "KGEN/MojoParser/SharedState.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"

namespace M::KGEN {
class ParamBindArrayAttr;
class ParamDeclAttr;
} // namespace M::KGEN

namespace M::KGEN::LIT {
class AliasDeclOp;
class AliasForwardDeclOp;
class ASTDecl;
class FileModuleOp;
class FuncOp;
class PackageOp;
class ParserBase;
class SharedState;
class UnresolvedImportOp;
class LetRegDeclOp;
class GlobalVarDeclOp;
class StructDeclOp;
class StructFieldOp;
class TraitDeclOp;
struct ParsedArgument;

//===----------------------------------------------------------------------===//
// DeclResolver
//===----------------------------------------------------------------------===//

/// This stores declaration references (e.g. vardecls, structdecls, funcdecls)
/// as operations.  It stores RValues for parameters and SSA values as an
/// RValue.
using DeclIRValue = PointerUnion<Operation *, PValue, SRValue, MRValue, SBValue,
                                 MBValue, MLValue, XLValue>;

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
  ASTDecl &addDecl(Operation *op, llvm::SMLoc loc, StringAttr baseName,
                   ASTDecl *parentDecl, LexerCursor cursor,
                   LexerCursor endCursor, ssize_t indentation);
  ASTDecl &addDecl(DeclIRValue irValue, llvm::SMLoc loc, StringAttr baseName,
                   ASTDecl *parentDecl, LexerCursor cursor,
                   LexerCursor endCursor, ssize_t indentation);

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

  /// Lookup a declaration from within a given module or package, emitting an
  /// error if it was not found.
  FailureOr<ArrayRef<ASTDecl *>>
  lookupDeclInModule(ASTDecl &module, StringAttr sourceName, SMLoc loc);

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

  /// Resolve the pending wildcard imports in the decl if it represents a
  /// module.
  LogicalResult resolveAllWildcardImports(ASTDecl &module);

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

  void registerAndCheckExport(StringRef aliasName, SMLoc loc);
  void exportMain(ASTDecl &funcDecl);

  ASTDecl *getDeclForFuncSymbol(SymbolRefAttr attr) const {
    auto it = declForFuncSymbol.find(attr);
    return it != declForFuncSymbol.end() ? it->second : nullptr;
  }

  /// Create a name from a signature by appending argument types into the name.
  static StringAttr getMangledName(StringAttr baseName,
                                   SignatureType signature);

  /// Generate a debug subprogram for this function and set it in its location.
  static void
  setLocationDebugScope(SharedState &shared,
                        DebugInfo::DIBuilder::ScopeGuard &diScopeGuard,
                        LIT::FuncOp &funcOp, StringRef baseName);

  /// Given a fully resolved signature, compute the final types and KGEN input
  /// conventions of the arguments.
  void
  computeArgumentConventions(SmallVectorImpl<ParamDeclAttr> &inputParamDecls,
                             MutableArrayRef<ParsedArgument> args,
                             MutableArrayRef<Type> argTypes,
                             MutableArrayRef<TypedAttr> defaults);

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
  ParseResult resolveBody(LetRegDeclOp op, Lexer &lexer, ASTDecl &decl);
  LogicalResult resolveSignature(GlobalVarDeclOp op, Lexer &lexer,
                                 ASTDecl &decl);
  ParseResult resolveBody(GlobalVarDeclOp op, Lexer &lexer, ASTDecl &decl);
  LogicalResult resolveSignature(AliasDeclOp op, Lexer &lexer, ASTDecl &decl);
  ParseResult resolveBody(AliasDeclOp op, Lexer &lexer, ASTDecl &decl);
  ParseResult resolveBody(AliasForwardDeclOp op, Lexer &lexer, ASTDecl &decl);

  /// Add a pre-existing set of declarations, which may optionally be imported
  /// from a given module, as children of the specified context, using the
  /// provided alias name (which may differ from that of the decl).
  LogicalResult aliasDeclsImpl(const TinyPtrVector<ASTDecl *> &decls,
                               StringAttr name, llvm::SMLoc aliasLoc,
                               ASTDecl &context, bool emitDiagnostics = true,
                               StringAttr moduleName = StringAttr(),
                               StringAttr declNameInModule = StringAttr());

  /// Move the children decls of `src` into `dst`. This is useful when a
  /// temporary decl needs to be created for parsing subexpressions but whose
  /// children will be inherited later by a decl being resolved.
  void moveDecls(ASTDecl &dst, ASTDecl &src);

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
/// argument           ::= [argument_convention] [argument_variadic] identifier
///                        [argument_type] ["=" expression]
/// argument_convention ::= "owned" | "borrowed" | "inout"
/// argument_variadic  ::= "*" | "**"
/// argument_type      ::= ":" star_expression
struct ParsedArgument {
  SMLoc loc;
  LexerCursor cursor;
  // Specify argument passing convention, e.g. owned/byref etc.
  enum {
    kConventionUnspec = 0,         // Nothing specified
    kConventionInOut = 1,          // inout x
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

  enum class ArgListKind {
    kParamList,         //< parameter list like `[x: Int, y: Int]`
    kArgList,           //< argument list like `(x: Int, y: Int)`
    kFnTypeArgList,     //< fn type, like `fn (Int, y: Float)`
    kBareLambdaArgList, //< argument list like `lambda x, y: x+y`
  };

  ParseResult parse(ParserBase &p, KWArgMarkerInfo &markerInfo,
                    ArgListKind kind);

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
      ParserBase &p, SmallVectorImpl<ParsedArgument> &args, ArgListKind kind);

  /// Parse an argument list, including the parentheses around them.  The
  /// argument list is allowed to be empty.  If `fnEffects` is non-null, then
  /// this parses 'raises' and other effects.
  static ParseResult parseAndResolveParenthesizedArgumentList(
      ParserBase &p, SmallVectorImpl<ParsedArgument> &args, ArgListKind kind,
      FnEffects *fnEffects);

  /// Process parsed parameter arguments into input parameters by determining
  /// the correct parameter types, conventions, and default parameter values.
  /// The unmangled parameter names are also collected.
  static void processParameterInputArgs(ExprEmitter &emitter,
                                        ASTDecl &declScope,
                                        ArrayRef<ParsedArgument> args,
                                        SmallVectorImpl<ParamDeclAttr> &params,
                                        SmallVectorImpl<StringAttr> &names,
                                        SmallVectorImpl<TypedAttr> &defaults,
                                        bool &paramVarArg);

  /// Process parsed parameter arguments into result parameters by determining
  /// the correct parameter types and conventions.
  static void processParameterResultArgs(ExprEmitter &emitter,
                                         ASTDecl &declScope,
                                         ArrayRef<ParsedArgument> args,
                                         SmallVectorImpl<ParamDeclAttr> &params,
                                         bool &paramVarArg);

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
};

} // namespace M::KGEN::LIT

#endif // DECLRESOLVER_H
