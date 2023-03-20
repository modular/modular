//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Declaration parsing and name binding logic.
//
//===----------------------------------------------------------------------===//

#include "LitDecls.h"
#include "ASTDecl.h"
#include "IRValues.h"
#include "LitDocString.h"
#include "LitExprEmitter.h"
#include "LitExprNodes.h"
#include "LitLexer.h"
#include "LitParameterEvaluator.h"
#include "LitParserBase.h"

#include "KGEN/CompilationOptions.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "LitSharedState.h"
#include "SpecialFunctions.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "Support/STLExtras.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

static constexpr const StringLiteral kMainSymbolName = "main";

/// Parse an expression and immediately resolve it to a type.  This returns
/// failure on parse error.
static ParseResult parseType(LitParserBase &p, ASTType &result,
                             ASTDecl &declScope,
                             std::optional<size_t> stmtIndent) {
  ExprNode *expr = nullptr;
  if (p.parseExpression(expr, stmtIndent))
    return failure();

  ExprEmitter emitter(p.shared, declScope, std::nullopt, nullptr);
  result = emitter.emitExprType(expr, /*isPack=*/false);
  if (!result)
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// ASTDecl
//===----------------------------------------------------------------------===//

MLIRContext *ASTDecl::getContext() const {
  if (auto *op = getIfOperation())
    return op->getContext();
  if (auto mv = dyn_cast<PValue>(getIRValue()))
    return mv.get().getContext();
  if (auto dr = dyn_cast<SRValue>(getIRValue()))
    return dr.getContext();
  if (auto value = dyn_cast_or_null<MRValue>(irValue))
    return value.getContext();
  return cast<SLValue>(getIRValue()).getContext();
}

/// If this is an RValue, return it otherwise return null.
CRValue ASTDecl::getIfRValue() const {
  if (auto attr = dyn_cast_or_null<PValue>(irValue))
    return attr;
  if (auto value = dyn_cast_or_null<SRValue>(irValue))
    return value;
  if (auto value = dyn_cast_or_null<MRValue>(irValue))
    return value;
  return {};
}

/// If this is an BValue, return it otherwise return null.
BValue ASTDecl::getIfBValue() const {
  if (auto attr = dyn_cast_or_null<PValue>(irValue))
    return attr;
  if (auto value = dyn_cast_or_null<SBValue>(irValue))
    return value;
  if (auto value = dyn_cast_or_null<MBValue>(irValue))
    return value;
  return {};
}

/// Return the SymbolRefAttr for a declaration, including all scoping that may
/// be needed, making it unique for every declaration.  This returns null for
/// named values that do not have a declaration.
SymbolRefAttr ASTDecl::getSymbolRef() const {
  auto op = dyn_cast_if_present<mlir::SymbolOpInterface>(getIfOperation());
  if (!op)
    return {};
  assert(
      (!isa<LIT::FuncOp>(op) || resolvedness >= DeclResolvedness::signature) &&
      "Functions don't have a symbol until their signatures are resolved");
  return getFullyResolvedSymbolRef(op);
}

/// Given an MLIR op for a struct declaration, return the self type.
ASTType ASTDecl::computeSelfTypeForStruct(LitSharedState &state) {
  auto structOp = cast<StructDeclOp>(*this);

  SmallVector<ParamBindAttr> parameters;
  for (auto decl : structOp.getInputParams()) {
    // We're using the parameter from the type declaration scope in the
    // parameter binding list.
    TypedAttr ref = ParamDeclRefAttr::get(decl);
    parameters.push_back(ParamBindAttr::get(decl.getName(), ref));
  }

  // Methods on structs (but not classes) take the struct implicitly by
  // pointer so they can use and mutate it.
  return DeclRefType::get(getSymbolRef(), parameters);
}

//===----------------------------------------------------------------------===//
// DeclResolver
//===----------------------------------------------------------------------===//

// Declarations (e.g. module, class, function) are parsed in multiple phases
// to increase laziness of the parse as well as make circular references
// possible.
//
// This ensures that the forward references between peer declarations are
// handled correctly as well as circular references, for example in mutually
// recursive functions and code like this:
//
//   def foo():
//     def bar():
//       print(x)
//     x = 42
//     bar()
//   foo()

DeclResolver::DeclResolver(LitSharedState &state) : LitSharedStateUser(state) {}

DeclResolver::~DeclResolver() {
  // Run the destructors on all the ASTDecl objects to make sure any
  // transitively allocated data is released.
  for (ASTDecl *decl : parsedDeclList)
    decl->~ASTDecl();
}

/// Add a new declaration that needs to be resolved.
ASTDecl &DeclResolver::addDecl(DeclIRValue irValue, SMLoc loc, StringAttr name,
                               ASTDecl *parentDecl, LitLexerCursor cursor,
                               LitLexerCursor endCursor, ssize_t indentation) {
  ASTDecl *decl = shared.allocPersistent<ASTDecl>(
      irValue, loc, parentDecl, cursor, endCursor, indentation);
  parsedDeclList.push_back(decl);

  // If this is a declaration which has a TypeCheckErrorType, then all
  // references to it are invalid.
  if (auto rv = decl->getIfRValue()) {
    if (isa<TypeCheckErrorType>(rv.getType().mlirType))
      decl->hasReferenceError = true;
  } else if (auto lv = decl->getIfLValue()) {
    if (isa<TypeCheckErrorType>(lv.getRValueType().mlirType))
      decl->hasReferenceError = true;
  }

  // If this has a parent and a name, insert it into the parents name table so
  // name lookup will resolve it.  If it does, then we're done.
  if (!name)
    return *decl;

  // Remember the named decl in the symbol table so it can be looked up.
  TinyPtrVector<ASTDecl *> &entries = parentDecl->declsInScope[name];
  if (entries.empty()) {
    entries.push_back(decl);

    // If the decl is a type or alias that has a symbol, remember it.  This
    // allows us to look up decls by symbol when referenced as types.
    if (auto structDecl = dyn_cast<StructDeclOp>(*decl)) {
      // Make sure there are no name conflicts with the MLIR symbol.  If there
      // are, then addDecl will have rejected it with an error.
      shared.setResolvedDeclSymbol(structDecl);

      SymbolRefAttr symbol = decl->getSymbolRef();
      assert(!declForTypeSymbol.count(symbol) &&
             "Symbol redefinition/collision");
      declForTypeSymbol[symbol] = decl;
    }

    return *decl;
  }

  // Function support method overloading on input arguments.  Variables and
  // types cannot be overloaded because they have no inputs.  Well, we could
  // actually allow type overloading on parameters theoretically to support
  // T[4] and T[1,7] as different things, but let's no proactively add
  // complexity.
  if (isa<FuncOp>(*decl)) {
    // Verify that all previous entries are also functions.  Note that we can't
    // check the overload set is compatible with each other because the
    // signatures aren't all resolved.
    for (ASTDecl *previous : entries) {
      if (!isa<FuncOp>(*previous)) {
        auto diag = emitError(decl->getLoc(), "invalid redefinition of ")
                    << name;
        diag.attachNote(previous->getLoc())
            << "cannot overload with this non-function definition";
        decl->hasReferenceError = true;
        previous->hasReferenceError = true;
        return *decl;
      }
    }

    // Otherwise, we're good, charge forwards.
    entries.push_back(decl);
    return *decl;
  }

  // Check if we are adding an identical unresolved import.
  if (auto import = dyn_cast<UnresolvedImportOp>(*decl)) {
    auto prevOp = dyn_cast<UnresolvedImportOp>(*entries.front());
    if (prevOp && import.getModuleNameAttr() == prevOp.getModuleNameAttr() &&
        import.getDeclNameAttr() == prevOp.getDeclNameAttr()) {
      entries.push_back(decl);
      return *decl;
    }
  }

  ASTDecl *existing = entries.back();
  auto diag = emitError(decl->getLoc(), "invalid redefinition of ") << name;
  diag.attachNote(existing->getLoc()) << "previous definition here";

  // Mark the existing decl and this one as erroneous so uses of either
  // don't create confusing errors.
  decl->hasReferenceError = true;
  for (ASTDecl *previous : entries)
    previous->hasReferenceError = true;
  return *decl;
}

void DeclResolver::aliasDecls(const TinyPtrVector<ASTDecl *> &decls,
                              StringAttr name, llvm::SMLoc aliasLoc,
                              ASTDecl &context) {
  (void)aliasDeclsImpl(decls, name, aliasLoc, context);
}

LogicalResult DeclResolver::aliasImportDecls(
    const TinyPtrVector<ASTDecl *> &decls, StringAttr name, StringAttr declName,
    StringAttr moduleName, llvm::SMLoc aliasLoc, ASTDecl &context) {
  return aliasDeclsImpl(decls, name, aliasLoc, context, moduleName, declName);
}

LogicalResult
DeclResolver::aliasDeclsImpl(const TinyPtrVector<ASTDecl *> &decls,
                             StringAttr name, llvm::SMLoc aliasLoc,
                             ASTDecl &context, StringAttr moduleName,
                             StringAttr declNameInModule) {
  auto [it, inserted] = context.declsInScope.try_emplace(name, decls);
  if (inserted)
    return success();

  // We hit an overlap, check to see if this is just resolving a module import.
  // If so, replace the unresolved import with the real decls.
  if (moduleName) {
    auto importOp = dyn_cast<UnresolvedImportOp>(*it->second.back());
    if (importOp && importOp.getModuleNameAttr() == moduleName &&
        importOp.getDeclNameAttr() == declNameInModule) {
      // Mark the placeholder imports as being resolved.
      for (ASTDecl *decl : it->second)
        decl->resolvedness = DeclResolvedness::fully;
      it->second = decls;
      return success();
    }
  }

  // Rejecting overlap is conservative and not what python does, but we can
  // relax this in the future when we know what the right policy should be.
  ASTDecl *existing = it->second.back();
  auto diag = emitError(aliasLoc, "invalid redefinition of ") << name;
  diag.attachNote(existing->getLoc()) << "previous definition here";

  for (ASTDecl *previous : it->second)
    previous->hasReferenceError = true;
  return failure();
}

LogicalResult DeclResolver::importModule(ASTDecl &context,
                                         StringAttr moduleName,
                                         StringAttr importName, SMLoc loc) {
  ASTDecl &module = shared.importModule(moduleName, loc);
  return aliasImportDecls(TinyPtrVector<ASTDecl *>(&module), importName,
                          /*declName=*/StringAttr(), moduleName, loc, context);
}

LogicalResult DeclResolver::importDeclFromModule(ASTDecl &context,
                                                 StringAttr moduleName,
                                                 StringAttr sourceName,
                                                 StringAttr destName,
                                                 SMLoc loc) {
  // Make sure the module has been resolved.
  ASTDecl &module = shared.importModule(moduleName, loc);
  if (failed(resolveFully(module, loc)))
    return failure();

  // Check to see if the module has the construct we are importing.
  auto result = shared.lookupAndResolveDecl(sourceName, loc, module,
                                            /*searchParentScopes=*/false);
  if (result.isErroneous())
    return failure();
  if (result.isFailure()) {
    // Emit an error with the module name without the leading `$` mangle.
    StringRef moduleName =
        cast<FileModuleOp>(module.getIfOperation()).getName();
    assert(moduleName.startswith("$") && "unexpected module name mangling");
    return emitError(loc, "module '" + moduleName.drop_front() +
                              "' does not contain '" + sourceName.getValue() +
                              "'");
  }
  return aliasImportDecls(TinyPtrVector<ASTDecl *>(result.getIfSuccess()),
                          destName, sourceName, moduleName, loc, context);
}

LogicalResult DeclResolver::importWildCardDeclsFromModule(ASTDecl &context,
                                                          StringAttr moduleName,
                                                          llvm::SMLoc loc) {
  // Make sure the module has been resolved.
  ASTDecl &module = shared.importModule(moduleName, loc);
  if (failed(resolveFully(module, loc)))
    return failure();

  // Wildcard imports don't import decls with a leading '_'.
  LogicalResult result = success();
  for (const auto &[name, decls] : module.declsInScope) {
    if (name.getValue()[0] == '_')
      continue;
    if (failed(aliasImportDecls(decls, name, name, moduleName, loc, context)))
      result = failure();
  }
  return result;
}

/// Add a new declaration that needs to be resolved.
ASTDecl &DeclResolver::addDecl(Operation *op, SMLoc loc, StringAttr name,
                               ASTDecl *parentDecl, LitLexerCursor cursor,
                               LitLexerCursor endCursor, ssize_t indentation) {
  return addDecl(DeclIRValue(op), loc, name, parentDecl, cursor, endCursor,
                 indentation);
}

/// Add a declaration that is already fully resolved.
ASTDecl &DeclResolver::addFullyResolvedDecl(DeclIRValue declVal,
                                            StringAttr name, SMLoc loc,
                                            ASTDecl *parentDecl) {
  auto &decl = addDecl(declVal, loc, name, parentDecl, LitLexerCursor(),
                       LitLexerCursor(), 0);
  decl.resolvedness = DeclResolvedness::fully;
  return decl;
}

ASTDecl &DeclResolver::addFullyResolvedDecl(DeclIRValue declVal, StringRef name,
                                            llvm::SMLoc loc,
                                            ASTDecl *parentDecl) {
  return addFullyResolvedDecl(declVal, StringAttr::get(getContext(), name), loc,
                              parentDecl);
}

ASTDecl &DeclResolver::addErroneousDecl(StringRef baseName, llvm::SMLoc loc,
                                        ASTDecl *parentDecl) {
  // Use a dummy attribute representation for the error.
  BoolAttr dummyAttr = BoolAttr::get(parentDecl->getContext(), true);
  ASTDecl &errDecl =
      addFullyResolvedDecl(PValue(dummyAttr), baseName, loc, parentDecl);
  errDecl.hasReferenceError = true;
  return errDecl;
}

/// Resolve all of the declarations that are visible.
void DeclResolver::resolveAll() {
  // We can do this in any order, but choose to use the order they are
  // discovered so diagnostics are mostly top-down.  Resolving declarations
  // may cause more entries to be added to this list.
  for (size_t i = 0; i != parsedDeclList.size(); ++i) {
    (void)resolveFully(*parsedDeclList[i], parsedDeclList[i]->getLoc());
  }
}

void DeclResolver::registerAndCheckExport(ExportOp exportOp) {
  StringAttr aliasName = exportOp.getAliasAttr();
  auto it = exportedSymbolNames.find(aliasName);
  if (it != exportedSymbolNames.end()) {
    auto diag = emitError(exportOp.getLoc(), "invalid re-export of ")
                << aliasName.getValue();
    diag.attachNote(it->getSecond()) << "previous export here";
    return;
  }
  exportedSymbolNames.insert({aliasName, exportOp.getLoc()});
}

void DeclResolver::exportMain(ASTDecl *containingDecl,
                              SymbolRefAttr symbolName) {
  StringAttr mainAttr = StringAttr::get(getContext(), kMainSymbolName);
  // If main has an explicit @export decorator we are done.
  if (exportedSymbolNames.count(mainAttr))
    return;
  // main was not exported explicitly, export it.
  OpBuilder builder = containingDecl->getDeclEndBuilder();
  auto exportOp = builder.create<ExportOp>(
      builder.getUnknownLoc(), symbolName,
      StringAttr::get(getContext(), kMainSymbolName), /*isCExport=*/true);
  exportedSymbolNames.insert({mainAttr, exportOp.getLoc()});
}

/// Resolve the specified declaration to at least the specified level of
/// resolution, performing incremental type checking as appropriate.
LogicalResult DeclResolver::resolve(ASTDecl &decl, DeclResolvedness howResolved,
                                    SMLoc loc) {
  // If decl is already resolved enough, we're done.
  if (decl.resolvedness >= howResolved) {
    // If decl is busted, then return failure.
    return success(!decl.hasReferenceError);
  }

  auto emitError = [&](SMLoc loc, const Twine &message) -> LitDiagnostic {
    return this->emitError(loc, message);
  };

  // If we are currently name binding this operation, we found a cycle, reject
  // it with an error.
  if (!declsCurrentlyProcessing.insert({&decl, loc}).second) {
    emitError(loc, "recursive reference to declaration")
            .attachNote(declsCurrentlyProcessing[&decl])
        << "previously used here";
    decl.hasReferenceError = true;
    return failure();
  }

  // If the signature hasn't been parsed, do so.
  if (decl.resolvedness < DeclResolvedness::signature) {
    // Handle each operation that can be name bound.  We handle this by
    // restoring the lexer to the position where parsing can continue, calling
    // the `resolveSignature` method for the op, and re-saving the new cursor
    // for the next stage of resolution.
    TypeSwitch<ASTDecl &>(decl)
        .Case<LIT::FuncOp, StructDeclOp, StructFieldOp, VarLetDeclOp,
              ParamDeclareOp, UnresolvedImportOp>([&](auto op) {
          LitLexer lexer(shared, decl.getCursor());

          // Resolve the signature: on a parse error, we note that the decl
          // is malformed and should not be referenced to silence downstream
          // errors.
          if (failed(resolveSignature(op, lexer, decl)))
            decl.hasReferenceError = true;
          decl.getCursor() = lexer.getCursor();
        })
        .Case<LIT::FileModuleOp, ModuleOp>([&](auto op) { /*Nothing*/ })
        .Default([&](auto &attr) {
          emitError(decl.getLoc(),
                    "do not know how to resolve the signature of this decl!");
          decl.hasReferenceError = true;
        });
    decl.resolvedness = DeclResolvedness::signature;
  }

  // If the declaration hasn't been fully parsed and we need to, do so.
  if (decl.resolvedness < DeclResolvedness::fully &&
      howResolved == DeclResolvedness::fully) {
    auto checkEndOfBodyCursor = [&](LitLexer &lexer) {
      // If the final parse of the declaration didn't match the initial
      // parse, report an error about unrecognized tokens at end of
      // declaration.
      if (!decl.isMatchingEndCursor(lexer.getCursor()) &&
          !decl.hasReferenceError) {
        if (lexer.getToken().isAny(LitToken::kw_def, LitToken::kw_struct,
                                   LitToken::kw_class, LitToken::kw_var))
          lexer.emitTokenError(
              "definition isn't on its own line at the correct "
              "indentation");
        else
          lexer.emitTokenError("unknown tokens at the end of a declaration");
      }
    };

    // Handle each operation that can be name bound.
    TypeSwitch<ASTDecl &>(decl)
        .Case<FileModuleOp, LIT::FuncOp, StructDeclOp, StructFieldOp,
              VarLetDeclOp, LetRegDeclOp, ParamDeclareOp, AliasForwardDeclOp>(
            [&](auto op) {
              // Parse the body of the declaration from the correct point.
              LitLexer lexer(shared, decl.getCursor());
              if (resolveBody(op, lexer, decl))
                return;

              checkEndOfBodyCursor(lexer);
            })
        .Case<ModuleOp, UnresolvedImportOp>([&](auto op) { /*Nothing*/ })
        .Default([&](auto &attr) {
          emitError(decl.getLoc(),
                    "do not know how to resolve the body of this decl!");
        });
    decl.resolvedness = DeclResolvedness::fully;

    // With the decl fully processed, validate the doc string.
    if (shared.shouldValidateDocStrings())
      validateLitDocString(shared, decl);
  }

  declsCurrentlyProcessing.erase(&decl);
  // If decl is busted, then return failure.
  return success(!decl.hasReferenceError);
}

//===----------------------------------------------------------------------===//
// LitParameterEvaluator implementation
//===----------------------------------------------------------------------===//

LitParameterEvaluator::LitParameterEvaluator(
    DeclResolver &resolver, ArrayRef<ParamBindAttr> paramValues)
    : ParameterEvaluator(paramValues), InterpreterState(/*target=*/nullptr),
      resolver(resolver) {}

FailureOr<TypedAttr>
LitParameterEvaluator::evaluateFunctionCall(SymbolRefAttr symbol,
                                            ArrayRef<Attribute> arguments) {
  ErrorOr<Region *> body = lookupFunctionBody(symbol);
  if (body.isError()) {
    // Swallow the error.
    return failure();
  }

  ErrorTreeOr<SmallVector<Attribute>> result =
      startInterpreterAt(*body.takeValue(), arguments);
  if (result.isError()) {
    // Swallow the error.
    DEBUG_WITH_TYPE("lit-parameter-evaluator",
                    result.getError().emit(
                        (InFlightDiagnostic(*)(Location))mlir::emitError));
    return failure();
  }

  return cast<TypedAttr>(result->front());
}

FailureOr<TypedAttr>
LitParameterEvaluator::evaluateExpression(ParamOperatorAttr op) {
  if (op.getOpcode() != POC::Apply)
    return failure();

  // We can only fold direct calls.
  auto ref = dyn_cast<SymbolConstantAttr>(op.getOperands().front());
  if (!ref)
    return failure();

  // All inputs must be simple constants.
  ArrayRef<TypedAttr> inputs = op.getOperands().drop_front();
  if (!llvm::all_of(inputs, ParameterAttr::isSimpleConstant))
    return failure();

  SmallVector<Attribute> arguments;
  for (TypedAttr input : inputs)
    arguments.push_back(input);

  return evaluateFunctionCall(ref.getSymbol(), arguments);
}

ErrorOr<Region *>
LitParameterEvaluator::lookupFunctionBody(SymbolRefAttr symbol) {
  ASTDecl *decl = resolver.getDeclForFuncSymbol(symbol);
  if (!decl)
    return Error("function not found: " + mlir::debugString(symbol));

  // Fail if the function is parameterized.
  if (failed(resolver.resolveSignature(*decl, decl->getLoc())))
    return Error("failed to resolve function signature");

  auto func = cast<LIT::FuncOp>(*decl);
  if (func.getAlwaysInlineLevel() == AlwaysInlineLevel::Disabled)
    return Error("function is not always_inline");
  SignatureType fullSig = func.getFullSignature();
  if (!fullSig.getInputParams().empty() || !fullSig.getResultParams().empty())
    return Error("function is parametric");

  // Make sure to fully resolve the body and everything within it.
  if (failed(resolver.resolveFully(*decl, decl->getLoc())))
    return Error("failed to fully resolve function");
  for (auto [name, childDecls] : decl->getDeclsInScope()) {
    for (ASTDecl *childDecl : childDecls) {
      if (failed(resolver.resolveFully(*childDecl, childDecl->getLoc())))
        return Error("failed to fully resolve function");
    }
  }
  return &func.getBodyRegion();
}

Type LitParameterEvaluator::refineType(Type type) {
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](ParamOperatorAttr op) -> TypedAttr {
    FailureOr<TypedAttr> result = evaluateExpression(op);
    if (failed(result))
      return op;
    return *result;
  });
  return replacer.replace(type);
}

//===----------------------------------------------------------------------===//
// Argument and Parameter List Parsing
//===----------------------------------------------------------------------===//

namespace {
enum VarArgKind { None, VarArg, KWVarArg };

/// Parsing support for a function argument and input parameter:
///
/// argument_list      ::= argument ("," argument)*
/// argument           ::= "/" | "*"
/// argument           ::= argument_variadic identifier_opt_type
///                           argument_ownership ["=" expression]
/// argument           ::= identifier "*" ":" type
/// argument_variadic  ::= "*" | "**"
/// argument_ownership ::= "&"
struct ParsedArgument {
  SMLoc loc;
  // Specify argument passing convention, e.g. owned/byref etc.
  enum {
    kConventionUnspec = 0,      // Nothing specified
    kConventionByRef = 1,       // x&
    kConventionOwned = 2,       // owned x
    kConventionBorrowed = 3,    // borrowed x
    kConventionByRefResult = 4, // No syntax: created by type checker.
  } convention = kConventionUnspec;

  // After type checking, this will hold the KGEN convention to use.
  ValueInputConvention kgenConvention = ValueInputConvention(128);

  bool isPack = false;
  VarArgKind vararg = VarArgKind::None;
  StringAttr name;
  ExprNode *typeExpr = nullptr;
  ExprNode *initExpr = nullptr;

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

  ParseResult parse(LitParserBase &p, KWArgMarkerInfo &markerInfo) {
    loc = p.getToken().getLoc();

    // The owned/borrowed keyword sets convention.
    // NOTE: We might consider a postfix ^ syntax after the language bakes out
    // more, that is probably going to be tightly coupled to ownership transfer,
    // but this is more explicit for now.
    if (p.consumeIf(LitToken::kw_owned))
      convention = kConventionOwned;

    SMLoc borrowLoc;
    if (p.consumeIf(LitToken::kw_borrowed, &borrowLoc)) {
      if (convention != kConventionUnspec)
        p.emitError(borrowLoc, "argument already has a convention specified");
      convention = kConventionBorrowed;
    }

    markerInfo = KWArgMarkerInfo::kNotMarker;

    // The first token of an argument may be a standalone '*' or '/' marker, and
    // the '*' may also be part of a varargs specification.  Check for these
    // first.
    if (p.consumeIf(LitToken::slash)) {
      markerInfo = KWArgMarkerInfo::kSlash;
      return success();
    }
    if (p.consumeIf(LitToken::star)) {
      if (p.getToken().isAny(LitToken::comma, LitToken::r_paren,
                             LitToken::r_square)) {
        markerInfo = KWArgMarkerInfo::kStar;
        return success();
      }
      vararg = VarArgKind::VarArg;
    } else if (p.consumeIf(LitToken::star_star)) {
      vararg = VarArgKind::KWVarArg;
      kwArgHandling = KWArgHandling::kKeywordOnly;
    }

    if (p.consumeIf(LitToken::star)) // '*' => variadic
      vararg = VarArgKind::VarArg;

    if (p.parseIdentifier(name, "expected parameter name"))
      // TODO: Scan ahead for better recovery.
      return failure();

    SMLoc packStarLoc;
    if (p.consumeIf(LitToken::star, &packStarLoc)) { // '*' => pack
      isPack = true;
      if (vararg != VarArgKind::None)
        p.emitError(packStarLoc,
                    "variadic arguments may not also be variadic packs");
    }

    // Process any convention markers.
    SMLoc ampLoc;
    if (p.consumeIf(LitToken::amp, &ampLoc)) { // '&' => by-ref
      if (isPack)
        p.emitError(ampLoc, "variadic packs may not have input conventions");
      else if (convention != kConventionUnspec)
        p.emitError(ampLoc, "argument already has a convention specified");
      else
        convention = kConventionByRef;
    }

    if (p.consumeIf(LitToken::colon)) {
      if (p.parseExpression(typeExpr, std::nullopt))
        return failure();
    }

    SMLoc equalLoc;
    if (p.consumeIf(LitToken::equal, &equalLoc)) {
      if (p.parseExpression(initExpr, std::nullopt))
        return failure();

      // Default args and varargs/packs don't mix.
      if (isPack || vararg != VarArgKind::None) {
        p.emitError(equalLoc, isPack ? "variadic packs" : "variadic arguments")
            << " may not have defaults" << initExpr->getRange();
        initExpr = nullptr;
      }
    }
    return success();
  };

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
  static ParseResult
  parseAndResolvePresentArgumentList(LitParserBase &p,
                                     SmallVectorImpl<ParsedArgument> &args,
                                     bool isParameterList) {
    // Figure out where to stop scanning.
    SmallVector<LitToken::Kind, 2> stopTokens;
    if (isParameterList)
      stopTokens.append({LitToken::r_square, LitToken::minus_greater});
    else
      stopTokens.push_back(LitToken::r_paren);

    // As we parse all of the arguments and the keyword arguments and markers,
    // we resolve the markers and check the invariants.  Python's parameter
    // grammar embeds checking for `/` and `*` into it, but we do this ad-hoc
    // for simplicity, according to the following rules:
    //
    //   1) Only one '/' and '*' marker may exist in the parameter list.
    //   2) They are specified in that order.
    //   3) `/` cannot be first, and '*' cannot be last in the list.
    //
    // See this for more information:
    // https://peps.python.org/pep-0570/#how-to-teach-this
    bool hasSlashMarker = false, hasStarMarker = false;
    auto defaultKWArgHandling = KWArgHandling::kPositionalOrKeyword;

    // This is invoked when we see a '/' marker.
    auto handleSlashMarker = [&](SMLoc loc) {
      if (hasSlashMarker) {
        p.emitError(loc,
                    "cannot have two '/' markers in the same argument list");
        return;
      }
      if (hasStarMarker) {
        p.emitError(loc, "cannot specify '/' marker after '*' marker");
        return;
      }

      if (args.empty())
        p.emitError(
            loc, "'/' marker cannot be used at the start of the argument list");

      // Ok, process it by changing all arguments we've seen to be positional
      // only.  The remaining ones will stay kPositionalOrKeyword though.
      for (auto &arg : args)
        arg.kwArgHandling = KWArgHandling::kPositionalOnly;
      hasSlashMarker = true;
    };

    // This is invoked when we see a '*' marker or '*arg' argument.
    auto handleStarMarker = [&](SMLoc loc, bool isMarker) {
      if (hasStarMarker)
        p.emitError(loc,
                    "cannot have two '*' markers in the same argument list");

      // Diagnose '*' marker at end of argument list for completeness.
      if (p.getToken().isAny(stopTokens) && isMarker)
        p.emitError(loc, "'*' marker is not allowed at end of argument list");

      // From now on, any parsed arguments are keyword only.
      defaultKWArgHandling = KWArgHandling::kKeywordOnly;
      hasStarMarker = true;
    };

    // This parses either an argument or a keyword argument specifier.
    auto parseArgument = [&]() -> ParseResult {
      KWArgMarkerInfo marker = KWArgMarkerInfo::kNotMarker;
      ParsedArgument arg;
      arg.kwArgHandling = defaultKWArgHandling;
      if (arg.parse(p, marker))
        return failure();

      // If this argument is just a marker, process it.
      if (marker == KWArgMarkerInfo::kSlash)
        return handleSlashMarker(arg.loc), success();
      if (marker == KWArgMarkerInfo::kStar)
        return handleStarMarker(arg.loc, /*isMarker=*/true), success();

      // Otherwise, if this is a varargs marker (*arg) or variadic pack (arg*),
      // handle it as a marker and an argument.
      if (arg.isPack || arg.vararg == VarArgKind::VarArg)
        handleStarMarker(arg.loc, /*isMarker=*/false);

      // If we have a **arg then it must be the last argument.
      if (arg.vararg == VarArgKind::KWVarArg &&
          p.getToken().isNot(stopTokens)) {
        p.emitError(arg.loc, "'**' marker must be at end of argument list");
        arg.vararg = VarArgKind::None;
      }

      // Otherwise just remember the argument.
      args.push_back(arg);
      return success();
    };

    // Parse a list of arguments and keyword argument specifiers.  Each argument
    // will leave its `kwargHandling` default initialized.
    if (p.parseCommaSeparatedList(parseArgument, stopTokens))
      return failure();

    // TODO(Keyword Args): now that we parsed a fully generic parameter list,
    // reject keyword arguments.
    if (!args.empty() &&
        args.back().kwArgHandling == KWArgHandling::kKeywordOnly)
      p.emitError(args.back().loc, "TODO: keyword arguments not supported yet");
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Parameter signature implementation
//===----------------------------------------------------------------------===//

/// meta_signature    ::= "[" meta_param_list ("->" meta_result_types)? "]"
/// meta_param_list   ::= argument_list | "(" ")"
/// meta_result_types ::= expression ("," expression)*
static ParseResult
parseOptionalParameterSignature(LitParserBase &p, ASTDecl &declScope,
                                SmallVector<ParamDeclAttr> &inputParams,
                                SmallVector<ParamDeclAttr> &resultParams,
                                bool &paramVararg) {
  if (!p.consumeIf(LitToken::l_square) || p.consumeIf(LitToken::r_square))
    return success();

  SmallVector<ParsedArgument> args;

  // Parse the meta parameters.  We either have () or a parameter list.
  if (p.consumeIf(LitToken::l_paren)) {
    if (p.parseToken(LitToken::r_paren,
                     "expected ')' in empty parameter list; try dropping the "
                     "'(' if you have parameters"))
      return failure();
  } else {
    // Parse an actual parameter list.
    if (ParsedArgument::parseAndResolvePresentArgumentList(
            p, args, /*isParameterList=*/true))
      return failure();
  }

  // Resolve each of the parameter declarations.
  auto &declResolver = p.getDeclResolver();

  // Mark the decl container as 'fully resolved' temporarily to facilitate
  // this, so it doesn't attempt to get resolved again.
  // FIXME(5975): This is a hack and shouldn't be needed.  The problem is
  // that parameters should be accessible before the body is, and we have
  // no way to express this currently.
  assert(declScope.resolvedness == DeclResolvedness::unparsed);
  llvm::SaveAndRestore X(declScope.resolvedness, DeclResolvedness::fully);
  ExprEmitter emitter(p.shared, declScope, std::nullopt, nullptr);

  auto processParameterArgs = [&](ArrayRef<ParsedArgument> args,
                                  SmallVectorImpl<ParamDeclAttr> &params,
                                  bool isResultParams) {
    for (auto &arg : args) {
      // Check for things supported in arguments that are not supported in
      // parameters.
      if (arg.initExpr)
        p.emitError(arg.loc,
                    "TODO: default values in parameters not supported");
      if (arg.isPack)
        p.emitError(arg.loc, "parameters may not be variadic packs");

      ASTType type;
      if (arg.typeExpr)
        type = emitter.emitExprType(arg.typeExpr, /*isPack=*/false);
      else
        p.emitError(arg.loc, "parameters must always have a type");

      if (!type)
        type = TypeCheckErrorType::get(p.getContext());

      // Parameters must be register passable for now.
      if (!type.isRegisterPassable(arg.loc, p.shared)) {
        p.emitError(arg.loc, "cannot use type ")
            << type
            << " in a parameter: only @register_passable types are supported "
               "right now";
        type = TypeCheckErrorType::get(p.getContext());
      }

      VarArgKind vararg = arg.vararg;
      if (vararg != VarArgKind::None) {
        if (isResultParams) {
          p.emitError(arg.loc, "result parameters may not be variadic");
        } else if (!isa<TypeCheckErrorType>(type.mlirType)) {
          type = KGEN::VariadicType::get(type);
          paramVararg = true;
        }
      }

      // TODO: Parameter decls should support conventions at some point.
      if (arg.convention != ParsedArgument::kConventionUnspec)
        p.emitError(arg.loc, "parameters must always be passed by-value");

      // Bind the parsed type expression so references from other parameters
      // can be resolved.
      if (arg.isPack)
        // The type of pack parameters such as `Ts*: type` is `variadic<type>`.
        type = KGEN::VariadicType::get(type);
      auto tmpDecl = ParamDeclRefAttr::get(arg.name, type);
      declResolver.addFullyResolvedDecl(PValue(tmpDecl), arg.name, arg.loc,
                                        &declScope);
      params.push_back(ParamDeclAttr::get(arg.name, type));
    }
  };
  processParameterArgs(args, inputParams, /*isResultParams=*/false);

  // Parse the meta results if present.
  if (p.consumeIf(LitToken::minus_greater)) {
    args.clear();
    // Parse a result parameter list.
    if (ParsedArgument::parseAndResolvePresentArgumentList(
            p, args, /*isParameterList=*/true))
      return failure();
    processParameterArgs(args, resultParams, /*isResultParams=*/true);
  }
  return p.parseToken(LitToken::r_square, "expected ']' for parameter list");
}

//===----------------------------------------------------------------------===//
// Doc String support logic
//===----------------------------------------------------------------------===//

void LitParserBase::parseDocString(ASTDecl &decl) {
  // The doc string is simply a followon string literal.
  if (getToken().isNot(LitToken::string))
    return;
  decl.setDocString(consumeToken());
}

//===----------------------------------------------------------------------===//
// Decorator support logic
//===----------------------------------------------------------------------===//

SmallVector<ExprNode *> LitParserBase::parseDecorators(ASTDecl &decl) {
  return parseDecorators(decl.getParentDecl()->getIndentation());
}

SmallVector<ExprNode *> LitParserBase::parseDecorators(ssize_t indentation) {
  SmallVector<ExprNode *> result;
  if (getToken().getIndentation())
    indentation = getToken().getIndentation().value();
  while (consumeIf(LitToken::at)) {
    ExprNode *decoratorExpr;
    if (parseExpression(decoratorExpr, indentation))
      break;
    result.push_back(decoratorExpr);
  }
  return result;
}

static void rejectDecorators(ArrayRef<ExprNode *> decoratorExprs, ASTDecl &decl,
                             LitSharedState &shared) {
  if (!decoratorExprs.empty())
    shared.emitError(decoratorExprs[0]->getLoc(),
                     "decorators not supported on this statement")
        << LitSourceRange(decoratorExprs.front()->getRangeStart(),
                          decoratorExprs.back()->getRangeEnd());
}

//===----------------------------------------------------------------------===//
// Function Decl implementation
//===----------------------------------------------------------------------===//

/// If this is a special function like __init__ return the enum that
/// identifies it, otherwise return kNormal.
SpecialFunctionKind SpecialFunctionInfo::getKind(StringRef name) {
  if (name.size() < 5 || !name.startswith("__") || !name.endswith("__"))
    return SpecialFunctionKind::kNormal;

#define SF(ENUM, NAME, NUMOPERANDS, EXPRNODE, FLAGS)                           \
  if (name == NAME)                                                            \
    return SpecialFunctionKind::ENUM;
#include "SpecialFunctions.def"

  // Otherwise, this declaration isn't known.
  return SpecialFunctionKind::kNormal;
}

/// If this is a special function like __init__ return the enum that
/// identifies it, otherwise return kNormal.
const SpecialFunctionInfo &SpecialFunctionInfo::get(SpecialFunctionKind kind) {
  static const SpecialFunctionInfo infos[] = {
      {nullptr, SpecialFunctionKind::kNormal, /*numOperands=*/-1, /*flags=*/0},
#define SF(ENUM, NAME, NUMOPERANDS, EXPRNODE, FLAGS)                           \
  {NAME, SpecialFunctionKind::ENUM, (NUMOPERANDS), (FLAGS)},
#include "SpecialFunctions.def"
  };

  assert(unsigned(kind) < sizeof(infos) / sizeof(infos[0]));
  return infos[unsigned(kind)];
}

/// Now that all the structural properties are determined, perform any
/// name-binding specific checks over the declaration.  This happens after
/// decorator processing because that is how defs work in Python.  This also
/// fills in any implicitly declared types, performs name mangling, and sets up
/// the signature correctly.
///
/// This allows magic behavior (like __new__ being static, checking of method
/// self requirements and enforcement of other invariants.
///
/// This returns failure (after emitting an error) when a type checking problem
/// is detected.
static void verifyFunctionNameBinding(ASTDecl &decl, LIT::FuncOp funcOp,
                                      StringAttr &name,
                                      SmallVector<ParsedArgument> &args,
                                      MutableArrayRef<Type> argTypes,
                                      ASTType &resultType,
                                      LitSharedState &shared) {
  SpecialFunctionInfo fnInfo = SpecialFunctionInfo::get(name);

  // On any semantic error we mark the declaration erroneous - so references to
  // it don't type check, and we clear our special function information.  This
  // reduces cascade errors.
  auto emitErrorLoc = [&](SMLoc loc, const Twine &message) {
    fnInfo = SpecialFunctionInfo();
    decl.hasReferenceError = true;
    return shared.emitError(loc, message);
  };
  auto emitError = [&](const Twine &message) {
    fnInfo = SpecialFunctionInfo();
    decl.hasReferenceError = true;
    return shared.emitError(funcOp.getLoc(), message);
  };

  // Fill in any missing arguments or diagnose missing ones in fn's.
  bool seenInitExpr = false;
  for (auto [arg, type] : llvm::zip(args, argTypes)) {
    if (!type) {
      if (funcOp.getIsDef()) {
        // If we are in a 'def', we infer object type for Python compatibility.
        type = shared.lookupObjectType(arg.loc, *decl.getParentDecl());
        if (!type)
          type = shared.getTypeCheckErrorType();
      } else {
        // In an 'fn' we report an error.
        emitErrorLoc(arg.loc, "'fn' parameter type must be specified")
            << LitSourceRange(arg.loc, arg.loc);
        type = shared.getTypeCheckErrorType();
      }
    }
    if (arg.initExpr) {
      seenInitExpr = true;
    } else if (seenInitExpr) {
      shared.emitError(arg.loc, "non-default argument follows default argument")
          << arg.typeExpr->getRange();
    }

    // If no convention was explicitly specified, provide a default.
    if (arg.convention == ParsedArgument::kConventionUnspec) {
      // TODO: Default to borrowed.
      arg.convention = ParsedArgument::kConventionOwned;

      // The first/self argument to __init__ is weird because it gets
      // ByRefResult argument convention, even though it is a declared argument.
      if (fnInfo.kind == SpecialFunctionKind::kInit && &arg == &args[0] &&
          resultType.isEqualCanon(shared.getNoneType())) {
        if (!ASTType(type).isRegisterPassable(arg.loc, shared)) {
          arg.convention = ParsedArgument::kConventionByRefResult;
          decl.isInitFnWithByRefResultSelf = true;
        } else
          emitErrorLoc(
              arg.loc,
              "'__init__' is not supported on register_passable types yet");
      }
    }
  }

  // This is true if the declared result type is modeled as the first argument
  // because it is returned in memory.
  bool hasMemoryResult =
      !args.empty() &&
      args[0].convention == ParsedArgument::kConventionByRefResult;
  ASTType declaredResultType =
      hasMemoryResult ? ASTType(argTypes[0]) : resultType;

  // If this definition is a struct/class member, compute the self type.
  ASTType selfType;
  size_t selfArgNumber = 0;
  if (auto *parentDecl = decl.getParentDecl())
    if (isa<StructDeclOp>(*parentDecl)) {
      //  The parent decl must be fully resolved in order to resolve any members
      //  of it.
      assert(parentDecl->resolvedness == DeclResolvedness::fully);
      selfType = parentDecl->getSelfType();
      // If there is an in-memory result, self is passed as arg #1 unless this
      // is init, where the return slot is self.
      if (hasMemoryResult && fnInfo.kind != SpecialFunctionKind::kInit)
        selfArgNumber = 1;
    }

  // Check any special function information.

  // __new__ and similar methods are implicitly static.
  if (fnInfo.flags & SpecialFunctionInfo::kImplicitlyStaticMethod)
    funcOp.setIsStatic(true);

  // Check that the 'self' argument of a method was specified correctly.
  if (selfType && !funcOp.getIsStatic()) {
    if (selfArgNumber >= argTypes.size()) {
      // TODO: We can/should relax this for 'def' declarations in the future,
      // they should be able to implicit ignore arguments like Python does.
      emitError("self argument must be present in instance method");
    } else if (!ASTType(argTypes[selfArgNumber]).isEqualCanon(selfType)) {
      auto diag = emitErrorLoc(args[selfArgNumber].loc,
                               "'self' argument must have type ")
                  << selfType << " but actually has type "
                  << ASTType(argTypes[selfArgNumber]);
      if (args[selfArgNumber].typeExpr)
        diag << args[selfArgNumber].typeExpr->getRange();
    }
  }

  if (funcOp.getIsStatic() && !selfType) {
    emitError("only methods on structs may be declared static");
    funcOp.setIsStatic(false);
  }

  // Verify the operand count lines up.
  if (fnInfo.numOperands != -1 &&
      size_t(fnInfo.numOperands + selfArgNumber) != args.size()) {
    size_t numOperands = fnInfo.numOperands;
    emitError("special function must have ")
        << numOperands << " operand" << plural(numOperands);
  }

  // Check other invariants based on method flags.
  if (fnInfo.isInstMethod()) {
    if (!selfType)
      emitError("special function must be a method");
    else if (funcOp.getIsStatic())
      emitError("special method may not be a static method");
    else if (!fnInfo.allowsByRefSelfInstMethod() &&
             args[selfArgNumber].convention != ParsedArgument::kConventionOwned)
      emitErrorLoc(args[selfArgNumber].loc,
                   "self argument cannot be passed by reference");
  }

  // Some functions like __new__ require a Self result type.
  if (fnInfo.flags & SpecialFunctionInfo::kSelfResult) {
    // Note: We could allow omitting result type and default it, at the cost of
    // extra language magic.
    if (!declaredResultType.isEqualCanon(selfType))
      emitError("") << name << " result type must be " << selfType;
  }

  // If the function is required to return None, verify that.
  if (fnInfo.hasNoneResult() &&
      !declaredResultType.isEqualCanon(shared.getNoneType())) {
    emitError("") << name << " result type must be elided (or None)";
    resultType = shared.getNoneType();
  }

  // Diagnose a common errors and handle other special cases.
  switch (fnInfo.kind) {
  default:
    break;
  case SpecialFunctionKind::kNew:
    emitError("'__new__' is not supported on structs; use '__init__' instead");
    break;
  case SpecialFunctionKind::kLitBool:
    if (!resultType.mlirType.isSignlessInteger(1))
      emitError("") << name << " result type must be __mlir_type.i1";
    break;
  case SpecialFunctionKind::kClone:
    if (fnInfo.isInstMethod() && selfType &&
        args[selfArgNumber].convention != ParsedArgument::kConventionByRef)
      emitErrorLoc(args[selfArgNumber].loc,
                   "self argument must be passed by reference");
    break;
  }

  // Mangle 'name', ensuring that overloaded methods get unique symbol names.
  SmallString<64> mangledName(name.getValue().begin(), name.getValue().end());
  mangledName += '(';

  // Finally, after all semantic checks are done, update the types to reflect
  // ABI information form the calling convention.

  // Now that all the types and signature information have been resolved,
  // compute the final MLIR types, KGEN conventions and mangled name.
  for (auto [arg, argType] : llvm::zip(args, argTypes)) {
    // Update the mangled name for this argument.
    if (&arg != &args[0])
      mangledName += ",";

    mangledName += ASTType(argType).getAsString();
    switch (arg.convention) {
    case ParsedArgument::kConventionUnspec:
      llvm_unreachable("should be resolved above");
    case ParsedArgument::kConventionOwned:
      // Memory-only owned argument are passed with a layer of indirection and
      // use a specific convention to model this.
      if (ASTType(argType).isRegisterPassable(arg.loc, shared))
        arg.kgenConvention = ValueInputConvention::OwnedInReg;
      else
        arg.kgenConvention = ValueInputConvention::OwnedInMem;
      break;
    case ParsedArgument::kConventionBorrowed:
      // Memory-only owned argument are passed with a layer of indirection and
      // use a specific convention to model this.
      if (ASTType(argType).isRegisterPassable(arg.loc, shared))
        arg.kgenConvention = ValueInputConvention::BorrowedInReg;
      else
        arg.kgenConvention = ValueInputConvention::BorrowedInMem;
      break;
    case ParsedArgument::kConventionByRef:
      arg.kgenConvention = ValueInputConvention::ByRef;
      mangledName += '&';
      break;
    case ParsedArgument::kConventionByRefResult:
      arg.kgenConvention = ValueInputConvention::ByRefResult;
      mangledName += "=&";
      break;
    }

    // Adjust the MLIR type if needed.
    if (arg.kgenConvention != ValueInputConvention::OwnedInReg &&
        arg.kgenConvention != ValueInputConvention::BorrowedInReg)
      argType = POP::PointerType::get(argType);
    if (arg.vararg == VarArgKind::VarArg)
      argType = KGEN::VariadicType::get(argType);

    if (arg.vararg == VarArgKind::VarArg)
      mangledName += '*';
  }
  mangledName += ')';

  name = StringAttr::get(funcOp.getContext(), mangledName);
}

namespace {
struct FnDecorators : public LitSharedStateUser {
  FnDecorators(ASTDecl &decl, LitSharedState &shared)
      : LitSharedStateUser(shared), decl(decl), funcOp(cast<LIT::FuncOp>(decl)),
        isMethod(isa<StructDeclOp>(*decl.getParentDecl())) {}

  void apply(SmallVector<ExprNode *> &decoratorExprs);
  void applyLate(SymbolRefAttr symbolName, StringRef unmangledName,
                 SmallVector<ExprNode *> &decoratorExprs);

private:
  void applyAdaptive(const DeclRefNode &node);
  void applyRaises(const DeclRefNode &node);
  void applyLateExport(Location loc, SymbolRefAttr symbolName,
                       StringRef aliasName);
  void applyLateExport(Location loc, SymbolRefAttr symbolName,
                       const CallNode &callNode);

  ASTDecl &decl;
  LIT::FuncOp funcOp;
  const bool isMethod;
};
} // namespace

void FnDecorators::applyAdaptive(const DeclRefNode &node) {
  if (funcOp.getIsAdaptive())
    emitError(node.getLoc(), "only one '@adaptive' decorator is allowed")
        << node.getRange();

  funcOp.setIsAdaptive(true);
}

void FnDecorators::applyRaises(const DeclRefNode &node) {
  if (funcOp.getIsDef()) {
    emitError(node.getLoc(), "methods defined with 'def' always raise")
        << node.getRange();
    return;
  }

  funcOp.setSignature(funcOp.getSignature().setFnEffect(FnEffects::Throws));
}

// Apply all signature decorators.
void FnDecorators::apply(SmallVector<ExprNode *> &decoratorExprs) {
  SmallVector<ExprNode *> unprocessed;
  for (ExprNode *decorator : decoratorExprs) {
    bool processedIt = false;

    // Process all the decorators we know about.
    if (auto declRef = dyn_cast<DeclRefNode>(decorator)) {
      processedIt = true;
      if (declRef->spelling == "staticmethod")
        funcOp.setIsStatic(true);
      else if (declRef->spelling == "raises")
        applyRaises(*declRef);
      else if (declRef->spelling == "always_inline")
        funcOp.setAlwaysInlineLevel(AlwaysInlineLevel::Enabled);
      else if (declRef->spelling == "adaptive")
        applyAdaptive(*declRef);
      else
        processedIt = false;
    }

    // `x()` forms.
    if (auto callNode = dyn_cast<CallNode>(decorator)) {
      if (auto declRef = dyn_cast<DeclRefNode>(callNode->callee)) {
        processedIt = true;
        // @always_inline("nodebug")
        if (declRef->spelling == "always_inline" &&
            callNode->args.size() == 1 &&
            isa<StringLiteralNode>(callNode->args[0]) &&
            cast<StringLiteralNode>(callNode->args[0])->getValue() == "nodebug")
          funcOp.setAlwaysInlineLevel(AlwaysInlineLevel::EnabledNoDebug);
        else
          processedIt = false;
      }
    }

    if (!processedIt)
      unprocessed.push_back(decorator);
  }
  decoratorExprs = unprocessed;
}

void FnDecorators::applyLateExport(Location loc, SymbolRefAttr symbolName,
                                   StringRef aliasName) {
  if (isMethod) {
    emitError(funcOp.getLoc(), "methods cannot be exported");
    return;
  }

  ASTDecl *containingDecl = decl.getParentDecl();
  auto builder = containingDecl->getDeclEndBuilder();
  auto exportOp = builder.create<ExportOp>(
      loc, symbolName, StringAttr::get(getContext(), aliasName),
      /*isCExport=*/true);
  getDeclResolver().registerAndCheckExport(exportOp);
}

void FnDecorators::applyLateExport(Location loc, SymbolRefAttr symbolName,
                                   const CallNode &node) {
  if (node.args.size() != 1 || !isa<StringLiteralNode>(node.args.front())) {
    emitError(
        node.getLoc(),
        "@export requires a string specifying the name of the exported symbol")
        << node.getParenRange();
    return;
  }
  std::string aliasName =
      cast<StringLiteralNode>(node.args.front())->getValue();
  if (!isCIdentifier(aliasName)) {
    emitError(loc, aliasName) << " is not a valid C identifier";
    return;
  }
  applyLateExport(loc, symbolName, aliasName);
}

void FnDecorators::applyLate(SymbolRefAttr symbolName, StringRef unmangledName,
                             SmallVector<ExprNode *> &decoratorExprs) {
  // Scan through and process decorator expressions that are in the late pass.
  for (ExprNode *decorator : decoratorExprs) {
    Location loc = translateLocation(decorator->getLoc());
    // Process all the decorators we know about.
    if (auto declRef = dyn_cast<DeclRefNode>(decorator)) {
      if (declRef->spelling == "export") {
        applyLateExport(loc, symbolName, unmangledName);
        continue;
      }

      emitError(decorator->getLoc(), "unsupported decorator: @")
          << declRef->spelling << declRef->getRange();
      continue;
    } else if (auto callNode = dyn_cast<CallNode>(decorator)) {
      if (auto declRef = dyn_cast<DeclRefNode>(callNode->callee))
        if (declRef->spelling == "export") {
          applyLateExport(loc, symbolName, *callNode);
          continue;
        }
    }
    emitError(decorator->getLoc(), "unsupported decorator")
        << decorator->getRange();
  }
}

/// funcdef ::=  [decorators] "def" identifier [meta_signature]
///              "(" [argument_list] ")" ["->" expression] ":" suite
///
LogicalResult DeclResolver::resolveSignature(LIT::FuncOp funcOp,
                                             LitLexer &lexer, ASTDecl &decl) {
  LitParserBase p(lexer);
  SmallVector<ExprNode *> decoratorExprs = p.parseDecorators(decl);
  assert(p.getToken().isAny(LitToken::kw_async, LitToken::kw_def,
                            LitToken::kw_fn) &&
         "not a function definition?");
  p.consumeIf(LitToken::kw_async);
  p.consumeToken();

  StringAttr baseName;
  if (p.parseIdentifier(baseName, "expected function name"))
    return failure();

  // Add meta parameters from an enclosing declaration to the symbol table.
  // These are /in/ our current scope because we do not want name conflicts with
  // them and they are instance (not type-level) values.
  // TODO: Generalize this to support nested structs and functions.
  bool inAStruct = isa<StructDeclOp>(*decl.getParentDecl());
  if (inAStruct) {
    auto structDecl = cast<StructDeclOp>(*decl.getParentDecl());
    auto parentLoc = decl.getParentDecl()->getLoc();
    for (auto param : structDecl.getInputParams()) {
      auto paramRef = ParamDeclRefAttr::get(param);
      addFullyResolvedDecl(PValue(paramRef), param.getName(), parentLoc, &decl);
    }
  }

  // Parse declared meta parameters and add them to the current scope.
  SmallVector<ParamDeclAttr> inputParamDecls, resultParamDecls;
  SmallVector<ParsedArgument> args;

  // Add the meta parameters to the symbol table, and resolve their types.  We
  // add all of these after generic signature parsing so types used in the
  // signature list resolve to enclosing scopes, and we add them before the
  // value signature list so the types and parameters can resolve to the bound
  // values.
  bool paramVararg = false;
  if (parseOptionalParameterSignature(p, decl, inputParamDecls,
                                      resultParamDecls, paramVararg) ||
      p.parseToken(LitToken::l_paren, "expected '(' for parameter list"))
    return failure();

  // Parse the argument list next if present.
  if (!p.consumeIf(LitToken::r_paren)) {
    if (ParsedArgument::parseAndResolvePresentArgumentList(
            p, args, /*isParameterList=*/false) ||
        p.parseToken(LitToken::r_paren, "expected ')' in argument list"))
      return failure();
  }

  // Parse the result type if present.
  ExprNode *resultTypeExpr = nullptr;
  if (p.consumeIf(LitToken::minus_greater)) {
    if (p.parseExpression(resultTypeExpr, std::nullopt))
      return failure();
  }
  if (p.parseToken(LitToken::colon, "expected ':' in function definition"))
    return failure();

  // Resolve the result parameter types now that the arguments are in scope.
  ExprEmitter typeEmitter(shared, decl, std::nullopt, nullptr);

  // Resolve the result type and any argument types that are present, leaving
  // any unspecified types null.
  SmallVector<Type> argTypes;
  SmallVector<TypedAttr> defaults;

  ASTType resultType;
  if (!resultTypeExpr) {
    // TODO: We shouldn't default this to none for 'def's.  This should default
    // to object type.  Our return checker is currently a lame duck.
    resultType = shared.getNoneType();
  } else {
    resultType = typeEmitter.emitExprType(resultTypeExpr, /*isPack=*/false);
    // On error, a diagnostic will be emitted, but we don't want to kill the
    // entire function definition.  We won't be able to correctly type check any
    // calls to this function though.
    if (!resultType) {
      resultType = shared.getTypeCheckErrorType();
      decl.hasReferenceError = true;
    }

    // Memory-only types get passed as the first argument to the function
    // by-reference.
    if (!resultType.isRegisterPassable(resultTypeExpr->getLoc(), shared)) {
      // Synthesize a result argument for this, and use None as the actual
      // function result.
      ParsedArgument resultArg;
      resultArg.loc = resultTypeExpr->getLoc();
      resultArg.name = StringAttr::get(shared.getContext(), "__result__");
      resultArg.convention = ParsedArgument::kConventionByRefResult;
      resultArg.typeExpr = resultTypeExpr;
      args.insert(args.begin(), resultArg);
      resultType = shared.getNoneType();
    }
  }

  for (auto [idx, arg] : llvm::enumerate(args)) {
    ASTType type;
    if (arg.typeExpr) {
      type = typeEmitter.emitExprType(arg.typeExpr, arg.isPack);

      // If the type couldn't be emitted, mark this function erroneous and put
      // in a placeholder type so we can continue type checking.
      if (!type) {
        decl.hasReferenceError = true;
        type = shared.getTypeCheckErrorType();
      }

    } else if (arg.name == "self" && inAStruct) {
      // If this is a 'self' argument in a fn that is a method, default to a
      // self type.  TODO: Should we do this, or default to object in a 'def'?
      assert(decl.getParentDecl()->resolvedness == DeclResolvedness::fully);
      type = decl.getParentDecl()->getSelfType();
    }
    argTypes.push_back(type);

    // Emit default argument values.
    if (const ExprNode *initExpr = arg.initExpr) {
      ExprEmitter emitter(shared, decl, /*builder*/ {},
                          /*varDeclCursor*/ nullptr);
      PValue value = emitter.emitExprPValue(initExpr, EC_DefaultArgument, type);
      if (!value)
        return failure();
      defaults.push_back(value);
    }
  }

  // Now that we have figured out the lexical structure, allow decorators to
  // take a crack at the signature.
  // Okay, apply them now.
  FnDecorators(decl, shared).apply(decoratorExprs);

  // Now that all the structural properties are determined, perform any
  // name-binding specific checks over the declaration.  This happens after
  // decorator processing because that is how defs work in Python.  This also
  // fills in any implicitly declared types.
  StringAttr name = baseName;
  verifyFunctionNameBinding(decl, funcOp, name, args, argTypes, resultType,
                            shared);

  // Finally now that the full signature has been resolved, build our IR.

  // Set the symbol to the mangled name and check for redefinition.
  funcOp.setSymNameAttr(name);

  // Remove the temporary "sym_namex" attribute set up in FuncOp::build, see
  // that method for an explanation.
  funcOp->removeAttr("sym_namex");

  if (Operation *existing = shared.setResolvedDeclSymbol(funcOp)) {
    // If the thing is adaptive, then we actually don't want to error.
    if (!existing->hasAttr(funcOp.getIsAdaptiveAttrName())) {
      // On redefinition this is an overload of the same name and same
      // signature.
      auto diag = p.emitError(funcOp.getLoc(), "redefinition of function ")
                  << name << " with identical signature";
      diag.attachNote(existing->getLoc()) << "previous definition here";
      decl.hasReferenceError = true;
    }
  }

  // Remember the mapping from its fully mangled symbol so we can find its AST
  // representation and body from IR references.
  SymbolRefAttr symbolName = getFullyResolvedSymbolRef(funcOp);
  declForFuncSymbol[symbolName] = &decl;

  // TODO: Handle the export attribute somehow else.  It should be a 'body
  // decorator' that is handled after the decl is fully resolved.
  FnDecorators(decl, shared).applyLate(symbolName, baseName, decoratorExprs);

  // If have a main function, fn main(), export it automatically.
  if (!inAStruct && isMainFunction(baseName, inputParamDecls, resultParamDecls,
                                   argTypes, resultType))
    getDeclResolver().exportMain(decl.getParentDecl(), symbolName);

  // Generate a debug subprogram for this function.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (auto &diBuilder = shared.diBuilder) {
    FileLineColLoc fileLineCol =
        funcOp.getLoc()->findInstanceOf<FileLineColLoc>();

    // Compute the subprogram flags.
    /// If we have any optimizations, mark the subprogram as optimized.
    DebugInfo::SubprogramFlags spFlags =
        shared.options.optimizationLevel ? DebugInfo::SubprogramFlags::Optimized
                                         : DebugInfo::SubprogramFlags::None;
    /// If the function has a body, treat it as a definition.
    if (!funcOp.isExternal())
      spFlags = spFlags | DebugInfo::SubprogramFlags::Definition;

    // Use unresolved types now for simplicity, these will get resolved during
    // compilation.
    auto mapUnresolvedType = [](Type type) -> DebugInfo::DIType {
      return DebugInfo::DIUnresolvedMLIRType::get(type);
    };
    auto type = DebugInfo::DISubroutineType::get(
        getContext(), map_to_vector(argTypes, mapUnresolvedType),
        mapUnresolvedType(resultType.mlirType));
    diScopeGuard = diBuilder->pushSubprogram(
        baseName, name, diBuilder->createFile(fileLineCol),
        fileLineCol.getLine(), fileLineCol.getLine(), spFlags, type);
    funcOp->setLoc(diBuilder->createScopedLoc(fileLineCol));
  }

  // Handle function effects.
  SmallVector<Location> argLocs;
  SmallVector<StringAttr> argNames;
  SmallVector<ValueInputConvention> inputConventions;
  FnEffects effects = funcOp.getMetadata().getFnEffects();
  if (paramVararg)
    effects = effects | FnEffects::ParamVararg;
  for (const ParsedArgument &arg : args) {
    argLocs.push_back(p.translateLocation(arg.loc));
    argNames.push_back(arg.name);
    inputConventions.push_back(arg.kgenConvention);
    if (arg.vararg == VarArgKind::VarArg)
      effects = effects | FnEffects::Vararg;
    else if (arg.vararg == VarArgKind::KWVarArg)
      effects = effects | FnEffects::KWVararg;
  }

  OpBuilder builder = decl.getDeclEndBuilder();
  auto signature = SignatureType::getChecked(
      [&] { return mlir::emitError(funcOp.getLoc()); },
      builder.getAttr<ParamDeclArrayAttr>(inputParamDecls),
      builder.getAttr<ParamDeclArrayAttr>(resultParamDecls),
      builder.getFunctionType(argTypes, {resultType.mlirType}),
      builder.getAttr<MetadataAttr>(inputConventions, defaults, effects));
  if (!signature)
    return failure();

  funcOp.setValueParamNamesAttr(builder.getAttr<StringArrayAttr>(argNames));
  funcOp.setSignature(signature);
  // If this is a nested function, set its parameter declaration. It will be
  // referenced via parameter references instead of symbol references.
  if (funcOp->getParentOfType<LIT::FuncOp>())
    funcOp.setParamDeclAttr(
        ParamDeclAttr::get(funcOp.getSymNameAttr(), signature));

  funcOp.getBody()->addArguments(argTypes, argLocs);

  // Functor used to build the debug info for an argument.
  auto buildArgDIInfo = [&](Value argVal, StringRef name, unsigned argIdx) {
    auto &diBuilder = shared.diBuilder;
    if (!diBuilder ||
        shared.options.debugLevel != CompilationOptions::kFullDebugInfo)
      return;
    auto bbArgLoc = argVal.getLoc()->findInstanceOf<FileLineColLoc>();

    auto varAttr = diBuilder->createLocalVariable(
        name, diBuilder->createFile(bbArgLoc), bbArgLoc.getLine(), argIdx + 1,
        /*alignInBits=*/0,
        DebugInfo::DIUnresolvedMLIRType::get(argVal.getType()));
    builder.create<DebugInfo::ValueOp>(argVal.getLoc(), argVal, varAttr);
  };

  // Set up the body of the fn/def, creating declarations for the value
  // parameters and adding them to the symbol table.
  for (auto [bbArg, parsedArg] :
       llvm::zip(funcOp.getBody()->getArguments(), args)) {
    auto convention = parsedArg.kgenConvention;
    // Don't bind byref-result, it is handled specially by 'return'.
    if (convention == ValueInputConvention::ByRefResult &&
        // The self argument to __init__ is special, it is explicitly
        // visible in the function but is byref-result.
        !decl.isInitFnWithByRefResultSelf)
      continue;

    buildArgDIInfo(bbArg, parsedArg.name, bbArg.getArgNumber());

    // VarArg arguments are always treated as their pop.variadic type
    // by-value right now.  TODO(literals): Project to a list like thing.
    if (parsedArg.vararg == VarArgKind::VarArg) {
      addFullyResolvedDecl(SRValue(bbArg), parsedArg.name, parsedArg.loc,
                           &decl);
      continue;
    }

    auto addDecl = [&, name = parsedArg.name,
                    loc = parsedArg.loc](DeclIRValue declVal) {
      addFullyResolvedDecl(declVal, name, loc, &decl);
    };

    switch (convention) {
    // Arguments passed by-reference can be directly used.
    case ValueInputConvention::ByRef:
    case ValueInputConvention::ByRefResult:
      addDecl(SLValue(bbArg));
      break;

    case ValueInputConvention::OwnedInMem:
      // by-value arguments are mutable in a def, immutable in an fn.
      // OwnedInMem passes ownership of the argument into the callee so we
      // can directly mutate it if we want to.
      if (funcOp.getIsDef())
        addDecl(SLValue(bbArg));
      else
        // FIXME: This should be an SLValue also even in an 'fn', we want to be
        // able to consume the argument since we own it, and might as well allow
        // it to be mutated.  Handle this in future patch.
        addDecl(MRValue(bbArg));
      break;

    case ValueInputConvention::OwnedInReg:
    case ValueInputConvention::BorrowedInReg:
    case ValueInputConvention::BorrowedInMem:
      // If this was passed by-value, then it becomes an rvalue in a `fn`.
      if (!funcOp.getIsDef()) {
        if (convention == ValueInputConvention::BorrowedInMem)
          addDecl(MBValue(bbArg));
        else if (convention == ValueInputConvention::OwnedInReg)
          // FIXME: This is incorrect, this makes every use think it owns the
          // argument as an RValue.  This should be dropped into a memory slot
          // so it is an LValue that can be consumed, or do we need a new
          // "SSA lvalue" sort of thing.
          addDecl(SRValue(bbArg));
        else
          addDecl(SBValue(bbArg));
        break;
      }

      // In a `def`, we create a mutable var.decl lvalue to allow
      // reassignment.  Figure out how to model the input value.
      CValue srcVal;
      if (convention == ValueInputConvention::BorrowedInMem)
        srcVal = MBValue(bbArg);
      else if (convention == ValueInputConvention::OwnedInReg)
        srcVal = SRValue(bbArg);
      else
        srcVal = SBValue(bbArg);

      Type varType = POP::PointerType::get(srcVal.getRValueType());
      auto varDecl = builder.create<VarLetDeclOp>(bbArg.getLoc(), varType,
                                                  parsedArg.name, /*isVar*/ 1);

      // Emit the initializer expression into the slot.
      ExprEmitter emitter(shared, decl, builder, /*varDeclCursor*/ nullptr);

      // Expr to provide location information.
      DeclRefNode srcExpr(
          StringRef(parsedArg.loc.getPointer(), parsedArg.name.size()));
      ValueDest dest(SLValue(varDecl), EC_DefArgumentShadow);
      if (!emitter.emitBValue({srcVal, &srcExpr}, dest))
        dest.resetForError();

      addDecl(SLValue(varDecl));
      break;
    }
  }
  return success();
}

ParseResult DeclResolver::resolveBody(LIT::FuncOp funcOp, LitLexer &lexer,
                                      ASTDecl &decl) {
  // Push the debug scope for this function if necessary so that nested
  // operations have proper debug info.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (auto spAttr = DebugInfo::extractScope(funcOp))
    diScopeGuard = shared.diBuilder->pushScopeGuard(spAttr);

  // Resolve the body of the decl.
  if (LitParserBase::parseSuite(decl, lexer))
    return failure();

  // Check to see if we have a kgen.return at the end of function.  If not,
  // complain or add one implicitly if we have no results.
  Block *bodyBlock = funcOp.getBody();

  // Insert the default end terminator.
  OpBuilder::atBlockEnd(bodyBlock).create<LIT::EndFuncOp>(funcOp.getLoc());

  // Check that any alias forward declarations have been completed.
  if (!shared.diags.isErrorEmitted()) {
    bodyBlock->walk([&](AliasForwardDeclOp aliasFwdDeclOp) {
      // If the location for the resultParam was never set then this forward
      // declaration was never defined.
      if (!aliasFwdDeclOp.getResultParamLoc().has_value()) {
        emitError(aliasFwdDeclOp.getLoc(), "alias ")
            << aliasFwdDeclOp.getNameAttr()
            << " was never defined by a result parameter";
      }
    });
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Module Decl implementation
//===----------------------------------------------------------------------===//

ParseResult DeclResolver::resolveBody(LIT::FileModuleOp op, LitLexer &lexer,
                                      ASTDecl &decl) {
  // Push a scope for the file of this module.
  DebugInfo::DIBuilder::ScopeGuard fileGuard;
  if (shared.diBuilder) {
    auto &sourceMgr = lexer.getSourceMgr();
    int fileId = sourceMgr.FindBufferContainingLoc(lexer.getToken().getLoc());
    if (fileId) {
      StringRef filename =
          sourceMgr.getMemoryBuffer(fileId)->getBufferIdentifier();
      fileGuard = shared.diBuilder->pushFile(filename, "/");
    }
  }

  return LitParserBase::parseSuite(decl, lexer);
}

//===----------------------------------------------------------------------===//
// VarLetDecl implementation
//===----------------------------------------------------------------------===//

/// var_decl_stmt ::= var_or_let identifier ":" expression ["=" expression]
///                 | var_or_let identifier "=" expression
/// var_or_let    ::= "var" | "let"
LogicalResult DeclResolver::resolveSignature(VarLetDeclOp varOp,
                                             LitLexer &lexer, ASTDecl &decl) {
  LitParserBase p(lexer);
  SmallVector<ExprNode *> decorators = p.parseDecorators(decl);

  p.consumeToken(); // eat the let/var.
  if (p.parseToken(LitToken::identifier,
                   "internal error: checked by stmt parser"))
    return failure();

  //  Parse the type if present.
  ASTType parsedType;
  if (p.consumeIf(LitToken::colon)) {
    if (parseType(p, parsedType, *decl.getParentDecl(), decl.getIndentation()))
      return failure();
  }

  // Parse the initializer if present.
  ExprNode *initExpr = nullptr;
  if (p.consumeIf(LitToken::equal)) {
    if (p.parseExpression(initExpr, decl.getIndentation()))
      return failure();
  }

  // Now that parsing succeeded, we do IR emission and semantic processing.

  // Handle the initializer if present.
  if (initExpr) {
    // We insert after var decl.
    OpBuilder builder(varOp->getBlock(), ++Block::iterator(varOp));
    ExprEmitter emitter(shared, *decl.getParentDecl(), builder,
                        /*varDeclCursor*/ nullptr);

    // If we have a type, then emit directly into the LValue.  Otherwise emit
    // into
    ValueDest dest;
    ExprContext exprContext = varOp.getIsVar() ? EC_VarInit : EC_LetInit;
    if (parsedType) {
      varOp.getResult().setType(POP::PointerType::get(parsedType));
      dest = ValueDest(SLValue(varOp), exprContext);
    } else {
      // If we don't, we emit into the varOp itself, because this will infer the
      // type of the varOp from the initializer expression.
      dest = ValueDest(varOp, exprContext);
    }

    if (!initExpr->emitIR(dest, emitter)) {
      dest.resetForError();
      return failure();
    }

    assert(!isa_and_nonnull<UnresolvedType>(
               varOp.getType().getResolvedElementType()) &&
           "RValue emission should have inferred var type");

  } else if (parsedType) {
    varOp.getResult().setType(POP::PointerType::get(parsedType));
  } else {
    // If there was neither a type or initializer, reject the var.
    emitError(varOp.getLoc(),
              "declaration must have either a type or an initializer");
    return failure();
  }

  rejectDecorators(decorators, decl, shared);

  // Now that this has been fully checked, we can promote to a LetRegDeclOp if
  // this was a non-parameteric register-passable `let` declaration with an
  // initializer.  We don't care about the address being available and this
  // produces smaller IR.
  ASTType inferredRValueType = varOp.getType().getResolvedElementType();
  if (initExpr && !varOp.getIsVar() &&
      // NOTE: This is assuming type parameters are valid register types.  We
      // will need to build out better support when we have traits, but this is
      // important for kernels in practice today.
      (!inferredRValueType ||
       inferredRValueType.isRegisterPassable(initExpr->getLoc(), shared))) {
    // There should be exactly one store to the original op, sanity check this.
    assert(varOp->hasOneUse() && "Should have one store use");
    auto theStore = cast<POP::StoreOp>(*varOp->user_begin());

    // Create new LetRegDeclOp and put it into the ASTDecl.
    OpBuilder builder(theStore);
    auto newLetOp = builder.create<LetRegDeclOp>(
        varOp.getLoc(), varOp.getNameAttr(), theStore.getArg());
    decl.setIRValue(newLetOp.getOperation());

    // Remove the store and the original LetVarDeclOp.
    theStore->erase();
    varOp->erase();
  }

  return success();
}

ParseResult DeclResolver::resolveBody(VarLetDeclOp op, LitLexer &lexer,
                                      ASTDecl &decl) {
  return success();
}

ParseResult DeclResolver::resolveBody(LetRegDeclOp op, LitLexer &lexer,
                                      ASTDecl &decl) {
  return success();
}

//===----------------------------------------------------------------------===//
// Alias Decl implementation
//===----------------------------------------------------------------------===//

/// alias_decl_stmt ::= "alias" identifier ":" expression ["=" expression]
///                   | "alias" identifier "=" expression
///
LogicalResult DeclResolver::resolveSignature(ParamDeclareOp paramDeclOp,
                                             LitLexer &lexer, ASTDecl &decl) {
  LitParserBase p(lexer);
  SmallVector<ExprNode *> decoratorExprs = p.parseDecorators(decl);

  // Parse the type if present.
  if (p.parseToken(LitToken::kw_alias,
                   "internal error: checked by stmt parser") ||
      p.parseToken(LitToken::identifier,
                   "internal error: checked by stmt parser"))
    return failure();

  ASTType type;
  if (p.consumeIf(LitToken::colon)) {
    if (parseType(p, type, *decl.getParentDecl(), decl.getIndentation()))
      return failure();
  }

  // Handle the case where there is no initializer.
  if (!p.consumeIf(LitToken::equal)) {
    // If there was neither a type or initializer, reject the var.
    if (!type) {
      p.emitError(paramDeclOp.getLoc(),
                  "declaration must have either a type or an initializer");
      return failure();
    }

    // `alias x: Int` is a forward declaration of a return parameter from a
    // function call, so it must occur in a function.
    if (!isa<LIT::FuncOp>(paramDeclOp->getParentOp())) {
      p.emitError(paramDeclOp.getLoc(),
                  "parameter results may only be declared in a function");
      return failure();
    }

    // Ok, things seem set up right, replace the ParamDeclOp with the right
    // operation that will allow us to track things.
    OpBuilder builder(paramDeclOp);
    Operation *forwardDecl = builder.create<AliasForwardDeclOp>(
        paramDeclOp.getLoc(), paramDeclOp.getName(), TypeAttr::get(type),
        mlir::LocationAttr());
    decl.setIRValue(forwardDecl);

    // Remove the paramDeclOp from the IR, since we ended up changing our mind
    // about how to represent this.
    paramDeclOp->erase();

    // The check that the alias was specified is handled when the function body
    // has been fully resolved.
    rejectDecorators(decoratorExprs, decl, shared);
    return success();
  }

  // Otherwise this is a normal `alias` declaration with an initializer.
  ExprNode *initExpr = nullptr;
  if (p.parseExpression(initExpr, decl.getIndentation()))
    return failure();

  ASTDecl &parentDecl = *decl.getParentDecl();
  ExprEmitter emitter(shared, parentDecl, /*builder*/ {},
                      /*varDeclCursor*/ nullptr);

  // Emit the value and convert to the expected type if we know it.
  auto rhsValue = emitter.emitExprPValue(initExpr, EC_AliasValue, type);
  if (!rhsValue)
    return failure();

  // If we had no declared type (`alias x = 42`), infer the type from the
  // initializer.
  if (!type)
    type = rhsValue.getType();

  // Remember the value, and update the type from UnresolvedType.
  NamedAttrList attrs = paramDeclOp->getAttrDictionary();
  attrs.set(paramDeclOp.getValueAttrName(), rhsValue.get());
  attrs.set(paramDeclOp.getParamDeclAttrName(),
            ParamDeclAttr::get(paramDeclOp.getName(), type));
  paramDeclOp->setAttrs(attrs.getDictionary(decl.getContext()));
  rejectDecorators(decoratorExprs, decl, shared);
  return success();
}

ParseResult DeclResolver::resolveBody(ParamDeclareOp op, LitLexer &lexer,
                                      ASTDecl &decl) {
  return success();
}

ParseResult DeclResolver::resolveBody(AliasForwardDeclOp aliasFwdDeclOp,
                                      LitLexer &lexer, ASTDecl &decl) {
  return success();
}

//===----------------------------------------------------------------------===//
// Struct Decl implementation
//===----------------------------------------------------------------------===//

/// structdef ::=
///   [decorators] "struct" identifier [meta_signature] ":" suite
///
LogicalResult DeclResolver::resolveSignature(StructDeclOp structOp,
                                             LitLexer &lexer, ASTDecl &decl) {
  LitParserBase p(lexer);
  SmallVector<ExprNode *> decoratorExprs = p.parseDecorators(decl);

  SmallVector<ParamDeclAttr> inputParamDecls;
  SmallVector<ParamDeclAttr> resultParamDecls;
  bool paramVarargs = false;
  if (p.parseToken(LitToken::kw_struct,
                   "internal error: checked by stmt parser") ||
      p.parseToken(LitToken::identifier,
                   "internal error: checked by stmt parser") ||
      parseOptionalParameterSignature(p, decl, inputParamDecls,
                                      resultParamDecls, paramVarargs) ||
      p.parseToken(LitToken::colon, "expected ':' in struct definition"))
    return failure();

  structOp.setInputParams(inputParamDecls);
  structOp.setParamVarargs(paramVarargs);

  // Reject result parameters.
  if (!resultParamDecls.empty())
    emitError(decl.getLoc(),
              "struct declarations do not support result parameters");

  // This is a struct, so we can use 'computeSelfTypeForStruct' to figure out
  // the self type.
  decl.setSelfType(decl.computeSelfTypeForStruct(shared));

  // Structs are memory-only unless they opt-in to being passed in registers.
  structOp.setIsRegisterPassable(false);

  // Now that we have the basic struct set up, process any known decorators.
  for (ExprNode *decorator : decoratorExprs) {
    if (auto declRef = dyn_cast<DeclRefNode>(decorator)) {
      if (declRef->spelling == "register_passable") {
        structOp.setIsRegisterPassable(true);
        continue;
      }

      emitError(decorator->getLoc(), "unsupported decorator: @")
          << declRef->spelling << declRef->getRange();
      continue;
    }

    emitError(decorator->getLoc(), "unsupported decorator")
        << decorator->getRange();
  }

  return success();
}

ParseResult DeclResolver::resolveBody(StructDeclOp structOp, LitLexer &lexer,
                                      ASTDecl &decl) {
  if (LitParserBase::parseSuite(decl, lexer))
    return failure();

  // Mark the declaration as fully resolved so we can lookup into it.
  decl.resolvedness = DeclResolvedness::fully;

  // Register-passable structs may only contain register-passable stored values.
  // TODO(traits): We need to type constrain mlirtype parameters to being
  // register-only types to support things like this correctly:
  //  struct P[T: mlirtype]:
  //    var storage : T
  if (structOp.getIsRegisterPassable()) {
    for (StructFieldOp field : structOp.getFieldDecls()) {
      // Make sure the field is fully resolved.
      auto elt = decl.lookupInCurrentScope(field.getNameAttr());
      assert(elt && elt->size() == 1 && "field decls cannot be overloaded");
      ASTDecl &fieldASTDecl = *elt->front();
      if (failed(resolveSignature(fieldASTDecl, fieldASTDecl.getLoc())))
        continue;

      // If the field is register-passable, then we're happy.
      if (ASTType(field.getType())
              .isRegisterPassable(fieldASTDecl.getLoc(), shared))
        continue;

      auto diag = emitError(structOp.getLoc(),
                            "all members of `@register_passable` struct must "
                            "themselves be register passable");
      diag.attachNote(fieldASTDecl.getLoc())
          << field.getNameAttr() << " declared with memory-only type "
          << ASTType(field.getType());

      // We cannot support IRGen'ing references to this type, since it will
      // break invariant about being register passable without being composed of
      // such types.
      decl.hasReferenceError = true;
      return failure();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// StructFieldDecl implementation
//===----------------------------------------------------------------------===//

/// struct_field_decl_stmt ::= "var" identifier ":" expression
/// TODO: Support default values?
///
LogicalResult DeclResolver::resolveSignature(StructFieldOp fieldOp,
                                             LitLexer &lexer, ASTDecl &decl) {
  LitParserBase p(lexer);
  SmallVector<ExprNode *> decoratorExprs = p.parseDecorators(decl);

  ASTType type;
  // Parse the type if present.
  if (p.parseToken(LitToken::kw_var,
                   "internal error: checked by stmt parser") ||
      p.parseToken(LitToken::identifier,
                   "internal error: checked by stmt parser") ||
      p.parseToken(LitToken::colon,
                   "struct field declaration must have a type") ||
      parseType(p, type, *decl.getParentDecl(), decl.getIndentation()))
    return failure();

  fieldOp.setType(type);
  rejectDecorators(decoratorExprs, decl, shared);
  return success();
}

ParseResult DeclResolver::resolveBody(StructFieldOp op, LitLexer &lexer,
                                      ASTDecl &decl) {
  return success();
}

//===----------------------------------------------------------------------===//
// UnresolvedImport Decl implementation
//===----------------------------------------------------------------------===//

ParseResult DeclResolver::resolveSignature(LIT::UnresolvedImportOp op,
                                           LitLexer &lexer, ASTDecl &decl) {
  // Check if we are importing a specific decl within the module, or the
  // module itself.
  if (auto declName = op.getDeclNameAttr()) {
    return getDeclResolver().importDeclFromModule(
        *decl.getParentDecl(), op.getModuleNameAttr(), declName,
        op.getImportNameAttr(), decl.getLoc());
  }
  return getDeclResolver().importModule(*decl.getParentDecl(),
                                        op.getModuleNameAttr(),
                                        op.getImportNameAttr(), decl.getLoc());
}

bool DeclResolver::isMainFunction(
    StringAttr &name, SmallVectorImpl<ParamDeclAttr> &inputParamDecls,
    SmallVectorImpl<ParamDeclAttr> &resultParamDecls,
    MutableArrayRef<Type> argTypes, ASTType &resultType) {
  return name == kMainSymbolName && inputParamDecls.empty() &&
         resultParamDecls.empty() && argTypes.empty() &&
         resultType.isEqualCanon(shared.getNoneType());
}
