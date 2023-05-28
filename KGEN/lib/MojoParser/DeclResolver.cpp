//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Declaration parsing and name binding logic.
//
//===----------------------------------------------------------------------===//

#include "DeclResolver.h"
#include "ASTDecl.h"
#include "CallEmission.h"
#include "DocString.h"
#include "ExprEmitter.h"
#include "ExprNodes.h"
#include "IRValues.h"
#include "Lexer.h"
#include "ParserBase.h"
#include "ParserParamEvaluator.h"

#include "KGEN/CompilationOptions.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "SharedState.h"
#include "SpecialFunctions.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

static constexpr const StringLiteral kMainSymbolName = "main";

/// Parse an expression and immediately resolve it to a type.  This returns
/// failure on parse error.
static ParseResult parseType(ParserBase &p, ASTType &result, ASTDecl &declScope,
                             std::optional<size_t> stmtIndent) {
  ExprNode *expr = nullptr;
  if (p.parseExpression(expr, stmtIndent))
    return failure();

  ExprEmitter emitter(p.shared, declScope, EC_Type, nullptr);
  result = emitter.emitExprType(expr);
  if (!result)
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// ASTDecl
//===----------------------------------------------------------------------===//

ArrayRef<ASTDecl *> ASTDecl::lookupInCurrentScope(StringRef name) const {
  return lookupInCurrentScope(StringAttr::get(getContext(), name));
}

/// Look up a name in this declaration's scope only: return null on failure.
ArrayRef<ASTDecl *> ASTDecl::lookupInCurrentScope(StringAttr name) const {
  assert((resolvedness == DeclResolvedness::fully ||
          // FIXME(Issue#5975): FuncOp shouldn't be special cased.
          isa<FuncOp>(*this)) &&
         "cannot perform lookup in a decl that isn't fully resolved");
  auto it = declsInScope.find(name);
  if (it != declsInScope.end() && !it->second.empty())
    return it->second;
  return {};
}

void ASTDecl::dump() const {
  // The value is either an operation or a type of MLIR `Value`.
  TypeSwitch<DeclIRValue>(getIRValue())
      .Case<Operation *>([](Operation *op) { op->dump(); })
      .Case<PValue, SRValue, MRValue, SBValue, MBValue, SLValue>(
          [](auto v) { v.dump(); })
      .Default([](DeclIRValue v) { llvm::errs() << "<null decl>\n"; });
}

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

PValue ASTDecl::getFuncAsPValue() const {
  return SymbolConstantAttr::get(getSymbolRef(),
                                 cast<LIT::FuncOp>(*this).getSignature());
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
ASTType ASTDecl::computeSelfTypeForStruct(SharedState &state) {
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

DeclResolver::DeclResolver(SharedState &state) : SharedStateUser(state) {}

DeclResolver::~DeclResolver() {
  // Run the destructors on all the ASTDecl objects to make sure any
  // transitively allocated data is released.
  for (ASTDecl *decl : parsedDeclList)
    decl->~ASTDecl();
}

/// This registers the finalized function with the DeclResolver after its
/// signature has been resolved and its mangled name is available.  This
/// returns an existing function if there is a redefinition problem.
Operation *DeclResolver::finalizeFuncSignature(LIT::FuncOp funcOp,
                                               ASTDecl &decl) {
  // Remember the mapping from its fully mangled symbol so we can find its AST
  // representation and body from IR references.
  declForFuncSymbol[getFullyResolvedSymbolRef(funcOp)] = &decl;

  // Install it in the symbol table and check for redefinition while doing so.
  return shared.setResolvedDeclSymbol(funcOp);
}

/// Add a new declaration that needs to be resolved.
ASTDecl &DeclResolver::addDecl(DeclIRValue irValue, SMLoc loc, StringAttr name,
                               ASTDecl *parentDecl, LexerCursor cursor,
                               LexerCursor endCursor, ssize_t indentation) {
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
  } else if (auto bv = decl->getIfBValue()) {
    if (isa<TypeCheckErrorType>(bv.getRValueType().mlirType))
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

LogicalResult DeclResolver::tryAliasDecls(const TinyPtrVector<ASTDecl *> &decls,
                                          StringAttr name, llvm::SMLoc aliasLoc,
                                          ASTDecl &context) {
  return aliasDeclsImpl(decls, name, aliasLoc, context,
                        /*emitDiagnostics=*/false);
}

LogicalResult DeclResolver::aliasImportDecls(
    const TinyPtrVector<ASTDecl *> &decls, StringAttr name, StringAttr declName,
    StringAttr moduleName, llvm::SMLoc aliasLoc, ASTDecl &context) {
  return aliasDeclsImpl(decls, name, aliasLoc, context,
                        /*emitDiagnostics=*/true, moduleName, declName);
}

LogicalResult DeclResolver::aliasDeclsImpl(
    const TinyPtrVector<ASTDecl *> &decls, StringAttr name,
    llvm::SMLoc aliasLoc, ASTDecl &context, bool emitDiagnostics,
    StringAttr moduleName, StringAttr declNameInModule) {
  // Check to see if the decl is an import. We create new decls within the
  // context for thse instead of aliasing, because import decls lazily replace
  // themselves with new decls (depending on what gets imported). That
  // replacement is only known when the import decl is referenced (and thus
  // resolved), so we can't alias the import directly.
  ASTDecl *frontDecl = decls.front();
  if (isa<UnresolvedImportOp>(*frontDecl)) {
    ASTDecl &importDecl = addDecl(
        frontDecl->getIfOperation(), frontDecl->getLoc(), name, &context,
        frontDecl->getCursor(), frontDecl->getCursor(), /*indentation=*/-1);
    return success(!importDecl.hasReferenceError);
  }

  auto [it, inserted] = context.declsInScope.try_emplace(name, decls);
  if (inserted)
    return success();
  TinyPtrVector<ASTDecl *> &entries = it->second;

  // We hit an overlap, check to see if this is just resolving a module import.
  // If so, replace the unresolved import with the real decls.
  if (moduleName) {
    auto importOp = dyn_cast<UnresolvedImportOp>(*it->second.back());
    if (importOp && importOp.getModuleNameAttr() == moduleName &&
        importOp.getDeclNameAttr() == declNameInModule) {
      // Mark the placeholder imports as being resolved.
      for (ASTDecl *decl : entries)
        decl->resolvedness = DeclResolvedness::fully;
      entries = decls;
    }
    return success();
  }
  ASTDecl *existing = it->second.back();

  // If the decls are functions, try to merge them into the existing set.
  if (isa<LIT::FuncOp>(*frontDecl) && isa<LIT::FuncOp>(*existing)) {
    // Check that none of the decls are already in the set.
    auto canMergeDecl = [&](ASTDecl *decl) {
      LIT::FuncOp declOp = cast<LIT::FuncOp>(decl->getIfOperation());
      bool isAdaptive = declOp.getIsAdaptive();
      return llvm::all_of(entries, [&](ASTDecl *existing) {
        if (failed(resolve(*existing, DeclResolvedness::signature, aliasLoc)))
          return false;
        LIT::FuncOp existingOp = cast<LIT::FuncOp>(existing->getIfOperation());

        // If the decl is adaptive, we can merge it with another adaptive decl.
        if (isAdaptive != existingOp.getIsAdaptive())
          return false;
        if (isAdaptive)
          return true;

        SignatureType declSignature = declOp.getFullSignature();
        SignatureType existingSignature = existingOp.getFullSignature();
        // If the value input types match exactly *and* the input parameter
        // types match exactly, then we don't want to merge this decl into the
        // set.
        if (declSignature.getValueInputs() ==
                existingSignature.getValueInputs() &&
            declSignature.getInputParamTypes() ==
                existingSignature.getInputParamTypes())
          return false;

        // We can merge the decl into the set.
        return true;
      });
    };
    if (llvm::all_of(decls, canMergeDecl)) {
      for (ASTDecl *decl : decls)
        entries.push_back(decl);
      return success();
    }
  }

  // Rejecting overlap is conservative and not what python does, but we can
  // relax this in the future when we know what the right policy should be.
  if (emitDiagnostics) {
    auto diag = emitError(aliasLoc, "invalid redefinition of ") << name;
    diag.attachNote(existing->getLoc()) << "previous definition here";

    for (ASTDecl *previous : it->second)
      previous->hasReferenceError = true;
  }
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
                               ASTDecl *parentDecl, LexerCursor cursor,
                               LexerCursor endCursor, ssize_t indentation) {
  return addDecl(DeclIRValue(op), loc, name, parentDecl, cursor, endCursor,
                 indentation);
}

/// Add a declaration that is already fully resolved.
ASTDecl &DeclResolver::addFullyResolvedDecl(DeclIRValue declVal,
                                            StringAttr name, SMLoc loc,
                                            ASTDecl *parentDecl) {
  auto &decl =
      addDecl(declVal, loc, name, parentDecl, LexerCursor(), LexerCursor(), 0);
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
  for (size_t i = 0; i != parsedDeclList.size(); ++i)
    (void)resolveFully(*parsedDeclList[i], parsedDeclList[i]->getLoc());
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

void DeclResolver::exportMain(ASTDecl &funcDecl) {
  ASTDecl *containingDecl = funcDecl.getParentDecl();
  auto symbolName = getFullyResolvedSymbolRef(cast<LIT::FuncOp>(funcDecl));

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

  auto emitError = [&](SMLoc loc, const Twine &message) -> InflightDiag {
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
          Lexer lexer(shared, decl.getCursor());

          // Resolve the signature: on a parse error, we note that the decl
          // is malformed and should not be referenced to silence downstream
          // errors.
          if (failed(resolveSignature(op, lexer, decl)))
            decl.hasReferenceError = true;
          decl.getCursor() = lexer.getCursor();
        })
        .Case<LIT::FileModuleOp, ModuleOp>([&](auto op) { /*Nothing*/ })
        .Default([&](auto &attr) {
          // Invalid function arguments will not be resolved to a value and will
          // have a null IR representation.
          if (!decl.hasReferenceError) {
            emitError(decl.getLoc(),
                      "do not know how to resolve the signature of this decl!");
            decl.hasReferenceError = true;
          }
        });
    // Never regress resolvedness. In the case of non inlined nested functions,
    // the body is fully resolved when the signature is resolved in order
    // to identify the value of 'capturing'
    if (decl.resolvedness != DeclResolvedness::fully)
      decl.resolvedness = DeclResolvedness::signature;
  }

  // If the declaration hasn't been fully parsed and we need to, do so.
  if (decl.resolvedness < DeclResolvedness::fully &&
      howResolved == DeclResolvedness::fully) {
    auto checkEndOfBodyCursor = [&](Lexer &lexer) {
      // If the final parse of the declaration didn't match the initial
      // parse, report an error about unrecognized tokens at end of
      // declaration.
      if (!decl.isMatchingEndCursor(lexer.getCursor()) &&
          !decl.hasReferenceError) {
        if (lexer.getToken().isAny(Token::kw_def, Token::kw_struct,
                                   Token::kw_class, Token::kw_var))
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
              Lexer lexer(shared, decl.getCursor());
              if (resolveBody(op, lexer, decl))
                return;

              checkEndOfBodyCursor(lexer);
            })
        .Case<ModuleOp, UnresolvedImportOp>([&](auto op) { /*Nothing*/ })
        .Default([&](auto &attr) {
          if (!decl.hasReferenceError)
            emitError(decl.getLoc(),
                      "do not know how to resolve the body of this decl!");
        });
    decl.resolvedness = DeclResolvedness::fully;

    // With the decl fully processed, validate the doc string.
    if (shared.shouldValidateDocStrings())
      validateDocString(shared, decl);
  }

  declsCurrentlyProcessing.erase(&decl);
  // If decl is busted, then return failure.
  return success(!decl.hasReferenceError);
}

//===----------------------------------------------------------------------===//
// ParserParamEvaluator implementation
//===----------------------------------------------------------------------===//

ParserParamEvaluator::ParserParamEvaluator(DeclResolver &resolver,
                                           ArrayRef<ParamBindAttr> paramValues)
    : ParameterEvaluator(paramValues), InterpreterState(/*target=*/nullptr),
      resolver(resolver) {}

FailureOr<TypedAttr>
ParserParamEvaluator::evaluateFunctionCall(SymbolRefAttr symbol,
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
ParserParamEvaluator::evaluateExpression(ParamOperatorAttr op) {
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
ParserParamEvaluator::lookupFunctionBody(SymbolRefAttr symbol) {
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
  if (!fullSig.getInputParamTypes().empty() ||
      !fullSig.getResultParamTypes().empty())
    return Error("function is parametric");

  // Make sure to fully resolve the body and everything within it.
  if (failed(resolver.resolveFully(*decl, decl->getLoc())) ||
      failed(resolver.recursivelyResolveFully(*decl, decl->getLoc())))
    return Error("failed to fully resolve function");
  return &func.getBodyRegion();
}

Type ParserParamEvaluator::refineType(Type type) {
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

ParseResult ParsedArgument::parse(ParserBase &p, KWArgMarkerInfo &markerInfo,
                                  bool omitName) {
  loc = p.getToken().getLoc();

  // Any owned/borrowed/inout keyword sets convention.
  if (p.consumeIf(Token::kw_owned))
    convention = kConventionOwned;
  else if (p.consumeIf(Token::kw_borrowed))
    convention = kConventionBorrowed;
  else if (p.consumeIf(Token::kw_inout))
    convention = kConventionInOut;
  while (p.getToken().isAny(Token::kw_owned, Token::kw_borrowed,
                            Token::kw_inout)) {
    p.emitTokenError("argument already has a convention specified");
    p.consumeToken();
  }

  markerInfo = KWArgMarkerInfo::kNotMarker;

  // The first token of an argument may be a standalone '*' or '/' marker, and
  // the '*' may also be part of a varargs specification.  Check for these
  // first.
  if (p.consumeIf(Token::slash)) {
    markerInfo = KWArgMarkerInfo::kSlash;
    return success();
  }
  if (p.consumeIf(Token::star)) {
    if (p.getToken().isAny(Token::comma, Token::r_paren, Token::r_square)) {
      markerInfo = KWArgMarkerInfo::kStar;
      return success();
    }
    vararg = VarArgKind::VarArg;
  } else if (p.consumeIf(Token::star_star)) {
    vararg = VarArgKind::KWVarArg;
    kwArgHandling = KWArgHandling::kKeywordOnly;
  }

  // When parsing a function type, the name is optional.
  SMLoc identifierLoc;
  if (!omitName) {
    if (p.parseIdentifier(name, "expected parameter name", &identifierLoc)) {
      // TODO: Scan ahead for better recovery.
      return failure();
    }
  }

  // Parse an optional type annotation: `":" ["*"] expression`. Omit the colon
  // if a name was not specified.
  if (!name || p.consumeIf(Token::colon)) {
    SMLoc starLoc = p.getToken().getLoc();
    if (p.getToken().getKind() == Token::star) {
      if (vararg != VarArgKind::VarArg) {
        InflightDiag diag = p.emitError(
            starLoc, "only variadic arguments' types can be unpacked");
        if (name) {
          diag.attachNote(identifierLoc)
              << "'" << name.getValue() << "' is not a variadic argument";
        }
      }
      vararg = VarArgKind::PackVarArg;
    }
    ExprNode *typeExprNode;
    if (p.parseStarredItem(typeExprNode))
      return failure();
    typeExpr = typeExprNode;
  }

  // Parse an optional default argument value: `"=" expression`.
  SMLoc equalLoc;
  if (p.consumeIf(Token::equal, &equalLoc)) {
    if (p.parseExpression(initExpr, std::nullopt))
      return failure();

    // Default args and varargs don't mix.
    if (vararg != VarArgKind::None) {
      p.emitError(equalLoc, "variadic arguments may not have defaults")
          << initExpr->getRange();
      initExpr = nullptr;
    }
  }
  return success();
}

ParseResult ParsedArgument::parseAndResolvePresentArgumentList(
    ParserBase &p, SmallVectorImpl<ParsedArgument> &args, bool isParameterList,
    bool omitNames) {
  // Figure out where to stop scanning.
  SmallVector<Token::Kind, 2> stopTokens;
  if (isParameterList)
    stopTokens.append({Token::r_square, Token::minus_greater});
  else
    stopTokens.push_back(Token::r_paren);

  // As we parse all of the arguments and the keyword arguments and markers, we
  // resolve the markers and check the invariants.  Python's parameter grammar
  // embeds checking for `/` and `*` into it, but we do this ad-hoc for
  // simplicity, according to the following rules:
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
      p.emitError(loc, "cannot have two '/' markers in the same argument list");
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
    for (ParsedArgument &arg : args)
      arg.kwArgHandling = KWArgHandling::kPositionalOnly;
    hasSlashMarker = true;
  };

  // This is invoked when we see a '*' marker or '*arg' argument.
  auto handleStarMarker = [&](SMLoc loc, bool isMarker) {
    if (hasStarMarker)
      p.emitError(loc, "cannot have two '*' markers in the same argument list");

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
    if (arg.parse(p, marker, omitNames))
      return failure();

    // If this argument is just a marker, process it.
    if (marker == KWArgMarkerInfo::kSlash)
      return handleSlashMarker(arg.loc), success();
    if (marker == KWArgMarkerInfo::kStar)
      return handleStarMarker(arg.loc, /*isMarker=*/true), success();

    // Otherwise, if this is a varargs marker, handle it as a marker and an
    // argument.
    if (arg.vararg == VarArgKind::VarArg ||
        arg.vararg == VarArgKind::PackVarArg)
      handleStarMarker(arg.loc, /*isMarker=*/false);

    // If we have a **arg then it must be the last argument.
    if (arg.vararg == VarArgKind::KWVarArg && p.getToken().isNot(stopTokens)) {
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
  if (!args.empty() && args.back().kwArgHandling == KWArgHandling::kKeywordOnly)
    p.emitError(args.back().loc, "TODO: keyword arguments not supported yet");
  return success();
}

//===----------------------------------------------------------------------===//
// Parameter signature implementation
//===----------------------------------------------------------------------===//

void ParsedArgument::processParameterArgs(
    ExprEmitter &emitter, ASTDecl &declScope, ArrayRef<ParsedArgument> args,
    SmallVectorImpl<ParamDeclAttr> &params, bool isResultParams,
    bool &paramVararg) {
  for (const ParsedArgument &arg : args) {
    // Check for things supported in arguments that are not supported in
    // parameters.
    if (arg.initExpr)
      emitter.emitError(arg.loc,
                        "TODO: default values in parameters not supported");

    ASTType type;
    if (arg.typeExpr)
      type = emitter.emitExprType(arg.typeExpr);
    else
      emitter.emitError(arg.loc, "parameters must always have a type");
    if (!type)
      type = TypeCheckErrorType::get(emitter.getContext());

    // Parameters must be register passable for now.
    if (!type.isRegisterPassable(arg.loc, emitter.shared)) {
      emitter.emitError(arg.loc, "cannot use type ")
          << type
          << " in a parameter: only @register_passable types are supported "
             "right now";
      type = TypeCheckErrorType::get(emitter.getContext());
    }

    VarArgKind vararg = arg.vararg;
    if (vararg != VarArgKind::None && isResultParams)
      emitter.emitError(arg.loc, "result parameters may not be variadic");
    if (vararg == VarArgKind::PackVarArg)
      emitter.emitError(arg.loc, "parameters may not be variadic packs");

    if (vararg == VarArgKind::VarArg &&
        !isa<TypeCheckErrorType>(type.mlirType)) {
      type = VariadicType::get(type);
      paramVararg = true;
    }

    // TODO: Parameter decls should support conventions at some point.
    if (arg.convention != ParsedArgument::kConventionUnspec)
      emitter.emitError(arg.loc, "parameters must always be passed by-value");

    // Bind the parsed type expression so references from other parameters
    // can be resolved.
    auto tmpDecl = ParamDeclRefAttr::get(arg.name, type);
    emitter.getDeclResolver().addFullyResolvedDecl(PValue(tmpDecl), arg.name,
                                                   arg.loc, &declScope);
    params.push_back(ParamDeclAttr::get(arg.name, type));
  }
}

/// meta_signature    ::= "[" meta_param_list ("->" meta_result_types)? "]"
/// meta_param_list   ::= argument_list | "(" ")"
/// meta_result_types ::= expression ("," expression)*
static ParseResult parseOptionalParameterSignature(
    ParserBase &p, ASTDecl &declScope, SmallVector<ParamDeclAttr> &inputParams,
    SmallVector<ParamDeclAttr> &resultParams, bool &paramVararg) {
  if (!p.consumeIf(Token::l_square) || p.consumeIf(Token::r_square))
    return success();

  SmallVector<ParsedArgument> args;

  // Parse the meta parameters.  We either have () or a parameter list.
  if (p.consumeIf(Token::l_paren)) {
    if (p.parseToken(Token::r_paren,
                     "expected ')' in empty parameter list; try dropping the "
                     "'(' if you have parameters"))
      return failure();
  } else {
    // Parse an actual parameter list.
    if (ParsedArgument::parseAndResolvePresentArgumentList(
            p, args, /*isParameterList=*/true))
      return failure();
  }

  // Mark the decl container as 'fully resolved' temporarily to facilitate
  // this, so it doesn't attempt to get resolved again.
  // FIXME(5975): This is a hack and shouldn't be needed.  The problem is
  // that parameters should be accessible before the body is, and we have
  // no way to express this currently.
  assert(declScope.resolvedness == DeclResolvedness::unparsed);
  llvm::SaveAndRestore X(declScope.resolvedness, DeclResolvedness::fully);
  ExprEmitter emitter(p.shared, declScope, EC_Type, nullptr);

  // Resolve each of the parameter declarations.
  ParsedArgument::processParameterArgs(emitter, declScope, args, inputParams,
                                       /*isResultParams=*/false, paramVararg);

  // Parse the meta results if present.
  if (p.consumeIf(Token::minus_greater)) {
    args.clear();
    // Parse a result parameter list.
    if (ParsedArgument::parseAndResolvePresentArgumentList(
            p, args, /*isParameterList=*/true))
      return failure();
    ParsedArgument::processParameterArgs(emitter, declScope, args, resultParams,
                                         /*isResultParams=*/true, paramVararg);
  }
  return p.parseToken(Token::r_square, "expected ']' for parameter list");
}

ASTType ParsedArgument::emitFunctionArgumentsAndResults(
    function_ref<ParseResult()> reportError, SharedState &shared,
    ExprEmitter &typeEmitter, const ExprNode *resultTypeExpr,
    FnEffects &effects, SmallVectorImpl<ParsedArgument> &args,
    SmallVectorImpl<Type> &argTypes, SmallVectorImpl<TypedAttr> &defaults,
    bool isDef, SMLoc resultLoc, ASTDecl &scope, SpecialFunctionInfo fnInfo,
    StringRef funcName) {
  // Resolve the result type and any argument types that are present, leaving
  // any unspecified types null.
  ASTType resultType;
  size_t skipIndex = 0;
  if (!resultTypeExpr) {
    resultType = shared.getNoneType();
    // Don't insert the return value for certain special functions.
    if (isDef && !fnInfo.hasNoneResult() && !fnInfo.isInitializer()) {
      // Insert an object memory-only result type.
      ParsedArgument resultArg;
      resultArg.loc = resultLoc;
      resultArg.name = StringAttr::get(shared.getContext(), "__result__");
      resultArg.convention = ParsedArgument::kConventionInOutResult;
      args.insert(args.begin(), resultArg);
      skipIndex = 1;
      argTypes.push_back(shared.lookupObjectType(resultLoc, scope));
      if (!argTypes.back()) {
        if (reportError())
          return {};
        argTypes.back() = shared.getTypeCheckErrorType();
      }
    }
  } else if (resultTypeExpr->kind == ExprNode::kNoneLiteral) {
    // If the result type is a `None` literal, then convert it to NoneType.
    resultType = shared.getNoneType();
  } else {
    resultType = typeEmitter.emitExprType(resultTypeExpr);
    // On error, a diagnostic will be emitted, but we don't want to kill the
    // entire function definition.  We won't be able to correctly type check any
    // calls to this function though.
    if (!resultType) {
      if (reportError())
        return {};
      resultType = shared.getTypeCheckErrorType();
    }

    // Memory-only types get passed as the first argument to the function
    // by-reference.
    uint8_t rp =
        resultType.getRegisterPassability(resultTypeExpr->getLoc(), shared);
    if (rp == StructDeclOp::RP_MemoryOnly) {
      // Synthesize a result argument for this, and use None as the actual
      // function result.
      ParsedArgument resultArg;
      resultArg.loc = resultTypeExpr->getLoc();
      resultArg.name = StringAttr::get(shared.getContext(), "__result__");
      resultArg.convention = ParsedArgument::kConventionInOutResult;
      resultArg.typeExpr = resultTypeExpr;
      args.insert(args.begin(), resultArg);
      argTypes.push_back(resultType);
      skipIndex = 1;
      resultType = shared.getNoneType();
    } else if (rp != StructDeclOp::RP_RegisterPassableTrivial) {
      // We know the result type of the function is register passable (because
      // otherwise it would be promoted to an argument).  If the result of the
      // function is a non-trivial type, mark the function effect as having an
      // owned result so ownership tracking will notice it.
      effects = effects | FnEffects::OwnedResult;
    }
  }

  bool seenInitExpr = false;
  for (auto [idx, arg] : llvm::enumerate(llvm::drop_begin(args, skipIndex))) {
    ASTType type;
    if (arg.typeExpr) {
      type = typeEmitter.emitExprType(arg.typeExpr);

      // If the type couldn't be emitted, mark this argument erroneous (so uses
      // within the body of the function don't trigger secondary errors) and
      // mark the function erroneous so calls to it won't resolve.  Put in a
      // placeholder type so we can continue type checking.
      if (!type) {
        if (reportError())
          return {};
        type = shared.getTypeCheckErrorType();
      }
    }
    argTypes.push_back(type);

    // Determine the required function effects from the conventions.
    if (arg.vararg == VarArgKind::VarArg)
      effects = effects | FnEffects::Vararg;
    else if (arg.vararg == VarArgKind::PackVarArg)
      effects = effects | FnEffects::PackVararg;
    else if (arg.vararg == VarArgKind::KWVarArg)
      effects = effects | FnEffects::KWVararg;

    // If no convention was explicitly specified, provide a default.
    if (arg.convention == ParsedArgument::kConventionUnspec)
      arg.convention = ParsedArgument::kConventionBorrowed;

    // Emit default argument values.
    if (const ExprNode *initExpr = arg.initExpr) {
      seenInitExpr = true;
      Type argType = type;
      if (isDef && !argType)
        // Within a `def` and without any type expression specified, convert the
        // default argument to `object` type.
        argType = shared.lookupObjectType(arg.loc, scope);
      PValue value =
          typeEmitter.emitExprPValue(initExpr, EC_DefaultArgument, argType);
      if (!value)
        return {};
      defaults.push_back(value);
    } else if (seenInitExpr) {
      typeEmitter.emitError(arg.loc,
                            "non-default argument follows default argument")
          << arg.typeExpr->getRange();
    }
  }
  return resultType;
}

void ParsedArgument::computeArgumentConventions(
    SharedState &shared, MutableArrayRef<ParsedArgument> args,
    MutableArrayRef<Type> argTypes) {
  for (auto [arg, argType] : llvm::zip(args, argTypes)) {
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
    case ParsedArgument::kConventionInOut:
      arg.kgenConvention = ValueInputConvention::ByRef;
      break;
    case ParsedArgument::kConventionInOutResult:
      arg.kgenConvention = ValueInputConvention::ByRefResult;
      break;
    case ParsedArgument::kConventionInitSelfResult:
      arg.kgenConvention = ValueInputConvention::InitSelf;
      break;
    }

    // Adjust the MLIR type if needed.
    if (arg.kgenConvention != ValueInputConvention::OwnedInReg &&
        arg.kgenConvention != ValueInputConvention::BorrowedInReg)
      argType = POP::PointerType::get(argType);
    if (arg.vararg == VarArgKind::VarArg)
      argType = KGEN::VariadicType::get(argType);
  }
}

//===----------------------------------------------------------------------===//
// Doc String support logic
//===----------------------------------------------------------------------===//

void ParserBase::parseDocString(ASTDecl &decl) {
  // The doc string is simply a follow-on string literal.
  if (getToken().isNot(Token::string))
    return;
  decl.setDocString(consumeToken());
}

//===----------------------------------------------------------------------===//
// Decorator support logic
//===----------------------------------------------------------------------===//

SmallVector<std::pair<ExprNode *, LexerCursor>>
ParserBase::parseDecorators(ASTDecl &decl) {
  return parseDecorators(decl.getParentDecl()->getIndentation());
}

SmallVector<std::pair<ExprNode *, LexerCursor>>
ParserBase::parseDecorators(ssize_t indentation) {
  SmallVector<std::pair<ExprNode *, LexerCursor>> result;
  if (getToken().getIndentation())
    indentation = getToken().getIndentation().value();
  while (consumeIf(Token::at)) {
    ExprNode *decoratorExpr;
    LexerCursor cursor = lexer.getCursor();
    if (parseExpression(decoratorExpr, indentation))
      break;
    result.push_back({decoratorExpr, cursor});

    if (!getToken().getIndentation() ||
        ssize_t(getToken().getIndentation().value()) > indentation) {
      emitTokenError("unexpected tokens after decorator, each need to be on "
                     "their own line");
      skipUntilIndentation(indentation);
    }
  }
  return result;
}

static void
rejectDecorators(ArrayRef<std::pair<ExprNode *, LexerCursor>> decoratorExprs,
                 ASTDecl &decl, SharedState &shared) {
  if (decoratorExprs.empty())
    return;

  shared.emitError(decoratorExprs[0].first->getLoc(),
                   "decorators not supported on this statement")
      << SourceRange(decoratorExprs.front().first->getRangeStart(),
                     decoratorExprs.back().first->getRangeEnd());
}

//===----------------------------------------------------------------------===//
// Function Decl implementation
//===----------------------------------------------------------------------===//

/// For a LIT::FuncOp, this returns whether the function is a special function
/// like __init__.
void ASTDecl::setSpecialFunctionKind(SpecialFunctionKind kind) {
  assert(isa<LIT::FuncOp>(*this));
  specialFunctionKind = uint8_t(kind);
}

SpecialFunctionKind ASTDecl::getSpecialFunctionKind() const {
  assert(
      isa<LIT::FuncOp>(*this) && resolvedness >= DeclResolvedness::signature &&
      "Can only get special function kind from signature resolved functions");

  return SpecialFunctionKind(specialFunctionKind);
}

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
static void
verifyFunctionNameBinding(ASTDecl &decl, LIT::FuncOp funcOp, StringAttr name,
                          SmallVector<ParsedArgument> &args,
                          MutableArrayRef<Type> argTypes, ASTType &resultType,
                          const FnEffects &effects, SharedState &shared,
                          SpecialFunctionInfo fnInfo) {
  // On any semantic error we mark the declaration erroneous - so references to
  // it don't type check, and we clear our special function information.  This
  // reduces cascade errors.
  auto emitErrorLoc = [&](SMLoc loc,
                          const Twine &message = Twine()) -> InflightDiag {
    fnInfo = SpecialFunctionInfo();
    decl.hasReferenceError = true;
    return shared.emitError(loc, message);
  };
  auto emitError = [&](const Twine &message = Twine()) -> InflightDiag {
    fnInfo = SpecialFunctionInfo();
    decl.hasReferenceError = true;
    return shared.emitError(funcOp.getLoc(), message);
  };

  // This is true if the declared result type is modeled as the first argument
  // because it is returned in memory.
  bool hasMemoryResult =
      !args.empty() &&
      args[0].convention == ParsedArgument::kConventionInOutResult;

  // If this definition is a struct/class member, compute the self type.
  ASTType selfType;
  ssize_t selfArgNumber = -1;
  if (auto *parentDecl = decl.getParentDecl()) {
    if (isa<StructDeclOp>(*parentDecl)) {
      // The parent decl must be fully resolved in order to resolve any members
      // of it.
      assert(parentDecl->resolvedness == DeclResolvedness::fully);
      selfType = parentDecl->getSelfType();
      // If there is an in-memory result, self is passed as arg #1 otherwise #0.
      selfArgNumber = hasMemoryResult ? 1 : 0;
    }
  }

  // __*init__ methods are weird - for memory-primary results we define
  // init in convention Python style, but for @register_passable values, we
  // return it.  We handle this by mapping them to different enumerators so
  // things downstream have stronger invariants.
  if ((fnInfo.kind == SpecialFunctionKind::kInit ||
       fnInfo.kind == SpecialFunctionKind::kCopyInit ||
       fnInfo.kind == SpecialFunctionKind::kMoveInit) &&
      selfType && ASTType(selfType).isRegisterPassable(decl.getLoc(), shared)) {
    if (fnInfo.kind == SpecialFunctionKind::kCopyInit)
      fnInfo = SpecialFunctionInfo::get(SpecialFunctionKind::kCopyInitReg);
    else if (fnInfo.kind == SpecialFunctionKind::kInit)
      fnInfo = SpecialFunctionInfo::get(SpecialFunctionKind::kInitReg);
    else {
      assert(fnInfo.kind == SpecialFunctionKind::kMoveInit);
      emitError() << name
                  << " is not supported for @register_passable types, they "
                     "are always movable by copying a register";
    }
  }

  // Fill in any missing arguments or diagnose missing ones in fn's.
  for (auto [i, arg, type] : llvm::enumerate(args, argTypes)) {
    if (!type) {
      // If this is the 'self' argument in a struct, default the type to Self.
      if (static_cast<ssize_t>(i) == selfArgNumber && selfType &&
          !funcOp.getIsStatic()) {
        type = selfType;
      } else if (funcOp.getIsDef()) {
        // If we are in a 'def', we infer object type for Python compatibility.
        type = shared.lookupObjectType(arg.loc, *decl.getParentDecl());
        if (!type)
          type = shared.getTypeCheckErrorType();
      } else {
        // In an 'fn' we report an error.
        emitErrorLoc(arg.loc, "'fn' parameter type must be specified")
            << SourceRange(arg.loc, arg.loc);
        type = shared.getTypeCheckErrorType();
      }
    }
  }

  ASTType declaredResultType =
      hasMemoryResult ? ASTType(argTypes[0]) : resultType;

  // Check any special function information.

  // __new__ and similar methods are implicitly static.
  if (fnInfo.flags & SpecialFunctionInfo::kImplicitlyStaticMethod)
    funcOp.setIsStatic(true);

  // Check that the 'self' argument of a method was specified correctly.
  if (selfType && !funcOp.getIsStatic()) {
    if (selfArgNumber >= ssize_t(argTypes.size())) {
      // TODO('def' allows unused arguments): We can/should relax this for
      // 'def' declarations in the future, they should be able to implicit
      // ignore arguments like Python does.
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

  // Verify the argument count lines up.
  if (fnInfo.kind != SpecialFunctionKind::kNormal &&
      fnInfo.numArguments != -1 &&
      fnInfo.numArguments + std::max(selfArgNumber, ssize_t(0)) !=
          ssize_t(args.size())) {
    size_t numArguments = fnInfo.numArguments;
    emitError("special function ") << name << " must have " << numArguments
                                   << " operand" << plural(numArguments);
  }

  // Check other invariants based on method flags.
  if (fnInfo.isInstMethod()) {
    if (!selfType) {
      emitError("special function must be a method");
    } else if (funcOp.getIsStatic()) {
      if (!(fnInfo.flags & SpecialFunctionInfo::kImplicitlyStaticMethod))
        emitError("special method may not be a static method");
    } else if (fnInfo.requiresOwnedSelfInstMethod()) {
      if (args[selfArgNumber].convention != ParsedArgument::kConventionOwned) {
        emitErrorLoc(args[selfArgNumber].loc, "self argument must be 'owned'")
            << FixIt::insertBeforeToken(args[selfArgNumber].loc, "owned ");
        args[selfArgNumber].convention = ParsedArgument::kConventionOwned;
      }
    } else if (!fnInfo.allowsByRefSelfInstMethod() &&
               args[selfArgNumber].convention !=
                   ParsedArgument::kConventionBorrowed)
      emitErrorLoc(args[selfArgNumber].loc,
                   "self argument cannot be passed by reference");
  }

  // Some functions like __new__ require a Self result type.
  if (fnInfo.flags & SpecialFunctionInfo::kSelfResult &&
      !declaredResultType.isEqualCanon(selfType))
    emitError() << name << " result type must be " << selfType;

  // If the function is required to return None, verify that.
  if (fnInfo.hasNoneResult() &&
      !declaredResultType.isEqualCanon(shared.getNoneType())) {
    emitError() << name << " result type must be elided (or None)";
    resultType = shared.getNoneType();
  }

  // Reject special functions declared as throwing when that is invalid.
  if (bitEnumContainsAny(effects, FnEffects::Throws) &&
      fnInfo.flags & SpecialFunctionInfo::kCannotRaise) {
    // Specialize the error if raising is implicit because it was defined as a
    // def.
    if (funcOp.getIsDef()) {
      emitError() << "cannot define " << name
                  << " as 'def'; 'def' implicitly raises"
                  << FixIt::replaceToken(decl.getLoc(), "fn");
    } else {
      emitError() << name << " cannot be declared as raising an exception";
    }
  }

  // Diagnose a common errors and handle other special cases.
  switch (fnInfo.kind) {
  default:
    break;
  case SpecialFunctionKind::kNew:
    emitError("'__new__' is not supported on structs; use '__init__' instead");
    break;
  case SpecialFunctionKind::kMLIRI1:
    if (!resultType.mlirType.isSignlessInteger(1))
      emitError() << name << " result type must be __mlir_type.i1";
    break;
  case SpecialFunctionKind::kInit:
  case SpecialFunctionKind::kCopyInit:
  case SpecialFunctionKind::kMoveInit: {
    // The first/self argument is syntactically declared as a by-ref argument,
    // but we need to change it to InitSelf since it is not initialized coming
    // in.
    assert(!args.empty() && "arg count already checked above");
    SMLoc selfArgLoc = args[0].loc;
    // __init__/__copyinit__/__moveinit__ must take their self argument by-ref
    // syntactically.
    if (args[0].convention != ParsedArgument::kConventionInOut) {
      auto diag = emitErrorLoc(selfArgLoc, "'self' in struct ")
                  << name << " must be passed as mutable reference";
      if (args[0].convention == ParsedArgument::kConventionUnspec)
        diag << FixIt::insertAfterToken(selfArgLoc, "&", shared);
    }

    // Regardless force it to init_self so recovery follows the fix-it.
    args[0].convention = ParsedArgument::kConventionInitSelfResult;

    if (fnInfo.kind == SpecialFunctionKind::kCopyInit) {
      if (args[1].convention != ParsedArgument::kConventionBorrowed)
        emitErrorLoc(args[1].loc,
                     "existing value argument must be passed as borrowed");
    } else if (fnInfo.kind == SpecialFunctionKind::kMoveInit) {
      if (args[1].convention != ParsedArgument::kConventionInOut &&
          args[1].convention != ParsedArgument::kConventionOwned) {
        emitErrorLoc(
            args[1].loc,
            "existing value argument must be passed as by-ref or owned");
      }
    }
    break;
  }
  }

  // Now that all the types and signature information have been resolved,
  // compute the final MLIR types and KGEN conventions.
  ParsedArgument::computeArgumentConventions(shared, args, argTypes);

  // If we have a special function kind and didn't have any errors with it,
  // remember which kind it is.
  decl.setSpecialFunctionKind(fnInfo.kind);
}

// Mangle 'name', ensuring that overloaded methods get unique symbol names.
static StringAttr getMangledName(StringAttr baseName, SignatureType signature) {
  SmallString<64> mangledName(baseName.getValue().begin(),
                              baseName.getValue().end());
  mangledName += '(';
  size_t argNo = 0;
  for (auto [convention, argType] : llvm::zip(
           signature.getValueInputConventions(), signature.getValueInputs())) {
    // Update the mangled name for this argument.
    if (argNo != 0)
      mangledName += ",";

    // If this had adjustments added to it because of its argument convention /
    // variadic state, strip them off.
    ASTType type = argType;
    // FIXME(#13015, #13603): In general, we shouldn't be checking for variadic
    // types specifically, but this is a quick stop-gap to address a crash.
    if (signature.isVararg(argNo) && isa<VariadicType>(type.mlirType))
      type = type.getVariadicElementType();
    if (convention != ValueInputConvention::OwnedInReg &&
        convention != ValueInputConvention::BorrowedInReg)
      type = type.getPointerElementType();
    mangledName += type.getAsString();

    // Add suffix to disambiguate overloadable conventions.
    switch (convention) {
    case ValueInputConvention::OwnedInReg:
    case ValueInputConvention::OwnedInMem:
    case ValueInputConvention::BorrowedInReg:
    case ValueInputConvention::BorrowedInMem:
      break;
    case ValueInputConvention::ByRef:
      mangledName += '&';
      break;
    case ValueInputConvention::ByRefResult:
    case ValueInputConvention::InitSelf:
      mangledName += "=&";
      break;
    }

    if (signature.isVararg(argNo))
      mangledName += '*';
    ++argNo;
  }
  mangledName += ')';
  return StringAttr::get(baseName.getContext(), mangledName);
}

namespace {
struct FnDecorators : public SharedStateUser {
  FnDecorators(ASTDecl &decl, SharedState &shared)
      : SharedStateUser(shared), decl(decl), funcOp(cast<LIT::FuncOp>(decl)) {}

  void apply(SmallVector<std::pair<ExprNode *, LexerCursor>> &decoratorExprs);
  void applyLate(StringRef unmangledName, ExprNode *decorator,
                 SignatureType &signature);

private:
  void applyAdaptive(const DeclRefNode &node);
  void applyLateExport(Location loc, StringRef aliasName);
  void applyLateExport(Location loc, const CallNode &callNode);

  ASTDecl &decl;
  LIT::FuncOp funcOp;
};
} // namespace

void FnDecorators::applyAdaptive(const DeclRefNode &node) {
  if (funcOp.getIsAdaptive())
    emitError(node.getLoc(), "only one '@adaptive' decorator is allowed")
        << node.getRange();

  funcOp.setIsAdaptive(true);
}

// Apply all signature decorators.
void FnDecorators::apply(
    SmallVector<std::pair<ExprNode *, LexerCursor>> &decoratorExprs) {
  SmallVector<std::pair<ExprNode *, LexerCursor>> unprocessed;
  for (auto [decorator, cursor] : decoratorExprs) {
    bool processedIt = false;

    // Process all the decorators we know about.
    if (auto declRef = dyn_cast<DeclRefNode>(decorator)) {
      processedIt = true;
      if (declRef->spelling == "staticmethod")
        funcOp.setIsStatic(true);
      else if (declRef->spelling == "always_inline")
        funcOp.setAlwaysInlineLevel(AlwaysInlineLevel::Enabled);
      else if (declRef->spelling == "adaptive")
        applyAdaptive(*declRef);
      else if (declRef->spelling == "parameter")
        funcOp.setIsParametric(true);
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
            callNode->args[0].isPositionalStringLiteral("nodebug"))
          funcOp.setAlwaysInlineLevel(AlwaysInlineLevel::EnabledNoDebug);
        else
          processedIt = false;
      }
    }

    if (!processedIt)
      unprocessed.push_back({decorator, cursor});
  }
  decoratorExprs = unprocessed;
}

void FnDecorators::applyLateExport(Location loc, StringRef aliasName) {
  if (isa<StructDeclOp>(*decl.getParentDecl())) {
    emitError(funcOp.getLoc(), "methods cannot be exported");
    return;
  }

  auto symbolName = getFullyResolvedSymbolRef(funcOp);

  ASTDecl *containingDecl = decl.getParentDecl();
  auto builder = containingDecl->getDeclEndBuilder();
  auto exportOp = builder.create<ExportOp>(
      loc, symbolName, StringAttr::get(getContext(), aliasName),
      /*isCExport=*/true);
  getDeclResolver().registerAndCheckExport(exportOp);
}

void FnDecorators::applyLateExport(Location loc, const CallNode &node) {
  if (node.args.size() != 1 || node.args[0].kind != CallArgument::kPositional ||
      !isa<StringLiteralNode>(node.args[0].expr)) {
    emitError(
        node.getLoc(),
        "@export requires a string specifying the name of the exported symbol")
        << node.getParenRange();
    return;
  }
  std::string aliasName =
      cast<StringLiteralNode>(node.args[0].expr)->getValue();
  if (!isCIdentifier(aliasName)) {
    emitError(loc, aliasName) << " is not a valid C identifier";
    return;
  }
  applyLateExport(loc, aliasName);
}

void FnDecorators::applyLate(StringRef unmangledName, ExprNode *decorator,
                             SignatureType &signature) {
  Location loc = translateLocation(decorator->getLoc());
  // Process all the decorators we know about.
  if (auto declRef = dyn_cast<DeclRefNode>(decorator)) {
    if (declRef->spelling == "export") {
      applyLateExport(loc, unmangledName);
    } else if (declRef->spelling == "noncapturing") {
      signature = signature.getWithFnEffects(
          bitEnumClear(signature.getFnEffects(), FnEffects::Capturing));
    } else if (declRef->spelling == "closure") {
      signature = signature.getWithFnEffects(signature.getFnEffects() |
                                             FnEffects::Capturing);
    } else {
      emitError(decorator->getLoc(), "unsupported decorator: @")
          << declRef->spelling << declRef->getRange();
    }
    return;
  }

  if (auto callNode = dyn_cast<CallNode>(decorator)) {
    if (auto declRef = dyn_cast<DeclRefNode>(callNode->callee))
      if (declRef->spelling == "export") {
        applyLateExport(loc, *callNode);
        return;
      }
  }
  emitError(decorator->getLoc(), "unsupported decorator")
      << decorator->getRange();
}

/// A valid main function must have signature main().
/// No parameters are allowed and here must be only one main in the final
/// object file.
static bool isMainFunction(StringAttr &name, LIT::FuncOp func,
                           SharedState &shared) {
  SignatureType signature = func.getSignature();
  return name == kMainSymbolName && signature.getInputParamTypes().empty() &&
         signature.getResultParamTypes().empty() &&
         signature.getValueInputs().empty() &&
         ASTType(func.getResultTypeWithoutErrorVariant())
             .isEqualCanon(shared.getNoneType());
}

/// funcdef   ::=  [decorators] def_or_fn identifier [meta_signature]
///                "(" [argument_list] ")" ["->" expression] ":" suite
/// def_or_fn ::= "def" | "fn"
///
LogicalResult DeclResolver::resolveSignature(LIT::FuncOp funcOp, Lexer &lexer,
                                             ASTDecl &decl) {
  ParserBase p(lexer);
  auto decoratorExprs = p.parseDecorators(decl);
  assert(p.getToken().isAny(Token::kw_async, Token::kw_def, Token::kw_fn) &&
         "not a function definition?");
  FnEffects effects =
      p.consumeIf(Token::kw_async) ? FnEffects::Async : FnEffects::None;
  if (p.getToken().is(Token::kw_def))
    effects = effects | FnEffects::Throws;
  p.consumeToken();

  StringAttr baseName;
  if (p.parseIdentifier(baseName, "expected function name"))
    return failure();

  // Add meta parameters from an enclosing declaration to the symbol table.
  // These are /in/ our current scope because we do not want name conflicts with
  // them and they are instance (not type-level) values.
  // TODO: Generalize this to support nested structs and functions.
  bool paramVararg = false;
  bool inAStruct = isa<StructDeclOp>(*decl.getParentDecl());
  if (inAStruct) {
    auto structDecl = cast<StructDeclOp>(*decl.getParentDecl());
    auto parentLoc = decl.getParentDecl()->getLoc();
    for (auto param : structDecl.getInputParams()) {
      auto paramRef = ParamDeclRefAttr::get(param);
      addFullyResolvedDecl(PValue(paramRef), param.getName(), parentLoc, &decl);
    }
    paramVararg = structDecl.getParamVarargs();
  }

  // Parse declared meta parameters and add them to the current scope.
  SmallVector<ParamDeclAttr> inputParamDecls, resultParamDecls;
  SmallVector<ParsedArgument> args;

  // Add the meta parameters to the symbol table, and resolve their types.  We
  // add all of these after generic signature parsing so types used in the
  // signature list resolve to enclosing scopes, and we add them before the
  // value signature list so the types and parameters can resolve to the bound
  // values.
  if (parseOptionalParameterSignature(p, decl, inputParamDecls,
                                      resultParamDecls, paramVararg) ||
      p.parseToken(Token::l_paren, "expected '(' for parameter list"))
    return failure();

  if (paramVararg)
    effects = effects | FnEffects::ParamVararg;

  // Parse the argument list next if present.
  if (!p.consumeIf(Token::r_paren)) {
    if (ParsedArgument::parseAndResolvePresentArgumentList(
            p, args, /*isParameterList=*/false) ||
        p.parseToken(Token::r_paren, "expected ')' in argument list"))
      return failure();
  }

  // Check for function effects.
  if (p.getToken().is(Token::identifier)) {
    SMLoc loc = p.getToken().getLoc();
    if (p.getToken().getSpelling() == "raises") {
      if (bitEnumContainsAny(effects, FnEffects::Throws))
        p.emitError(loc, "function effect 'raises' was already specified");
      effects = effects | FnEffects::Throws;
    } else {
      emitError(loc, "unknown function effect '")
          << p.getToken().getSpelling() << "', expected 'raises'";
    }
    p.consumeToken();
  }

  // Parse the result type if present.
  ExprNode *resultTypeExpr = nullptr;
  SMLoc resultLoc = p.getToken().getLoc();
  if (p.consumeIf(Token::minus_greater)) {
    if (p.parseExpression(resultTypeExpr, std::nullopt))
      return failure();
  }
  if (p.parseToken(Token::colon, "expected ':' in function definition"))
    return failure();

  // Resolve the result parameter types now that the arguments are in scope.
  ExprEmitter typeEmitter(shared, decl, EC_Type, nullptr);

  // Now that we have figured out the lexical structure, allow decorators to
  // take a crack at the signature.
  // Okay, apply them now.
  FnDecorators(decl, shared).apply(decoratorExprs);

  // Emit the argument and result types.
  SmallVector<Type> argTypes;
  SmallVector<TypedAttr> defaults;
  auto reportError = [&] {
    decl.hasReferenceError = true;
    return success();
  };
  SpecialFunctionInfo fnInfo = SpecialFunctionInfo::get(baseName);
  ASTType resultType = ParsedArgument::emitFunctionArgumentsAndResults(
      reportError, shared, typeEmitter, resultTypeExpr, effects, args, argTypes,
      defaults, funcOp.getIsDef(), resultLoc, decl, fnInfo);
  if (!resultType)
    return failure();

  // Nested functions are capturing by default.
  if (funcOp->getParentOfType<FuncOp>())
    effects = effects | FnEffects::Capturing;

  // Now that all the structural properties are determined, perform any
  // name-binding specific checks over the declaration.  This happens after
  // decorator processing because that is how defs work in Python.  This also
  // fills in any implicitly declared types.
  verifyFunctionNameBinding(decl, funcOp, baseName, args, argTypes, resultType,
                            effects, shared, fnInfo);

  // Finally now that the full signature has been resolved, build our IR.

  // Handle function effects.
  SmallVector<Location> argLocs;
  SmallVector<StringAttr> argNames;

  // Any function that contains a capturing closure as a parameter is itself
  // capturing.
  // TODO: Check struct elements too.
  bool transivelyCaptures = llvm::any_of(
      llvm::concat<ParamDeclAttr>(inputParamDecls, resultParamDecls),
      [](ParamDeclAttr decl) {
        if (auto signature = dyn_cast<SignatureType>(decl.getType()))
          return signature.isCapturing();
        return false;
      });
  if (transivelyCaptures)
    effects = effects | FnEffects::Capturing;

  // If the function raises, it implicitly gets a variant result type.
  if (bitEnumContainsAny(effects, FnEffects::Throws)) {
    if (ASTType errorType = shared.getBuiltinErrorType(decl.getLoc())) {
      resultType = POP::VariantType::get({errorType, resultType});

      // FIXME(#12604): We cannot return an Error type from a function that also
      // throws. This is because Variant collapses the variant to one case and
      // we can't tell which is which.  We could fix this in a number of ways in
      // the future if/when it matters.
      if (cast<POP::VariantType>(resultType.mlirType).getNumTypes() == 1) {
        p.emitError(funcOp.getLoc(),
                    "cannot return and raise the same type from a function");
        resultType =
            POP::VariantType::get({errorType, shared.getTypeCheckErrorType()});
        decl.hasReferenceError = true;
      }
    } else {
      resultType = shared.getTypeCheckErrorType();
      decl.hasReferenceError = true;
    }
  }

  // Handle argument effects and build the ASTDecls for the arguments.
  SmallVector<ValueInputConvention> inputConventions;
  for (const ParsedArgument &arg : args) {
    argLocs.push_back(p.translateLocation(arg.loc));
    argNames.push_back(arg.name);
    inputConventions.push_back(arg.kgenConvention);

    // Add an ASTDecl for the argument.  This will actually be set up during
    // body resolution (when the vardecls and other things are set up) because
    // the argument types referenced are not necessarily fully resolved.  We
    // create the decls here in order to pass location information for each
    // argument over to body resolution.
    if (arg.kgenConvention != ValueInputConvention::ByRefResult)
      addDecl(DeclIRValue(), arg.loc, arg.name, &decl, LexerCursor(),
              LexerCursor(), /*indent*/ 0);
  }

  OpBuilder builder = decl.getDeclEndBuilder();
  NamedAttrList attrs = funcOp->getAttrDictionary();
  auto inputParamsAttr = builder.getAttr<ParamDeclArrayAttr>(inputParamDecls);
  auto resultParamsAttr = builder.getAttr<ParamDeclArrayAttr>(resultParamDecls);
  attrs.set(funcOp.getValueParamNamesAttrName(),
            builder.getAttr<StringArrayAttr>(argNames));
  attrs.set(funcOp.getInputParamsAttrName(), inputParamsAttr);
  attrs.set(funcOp.getResultParamsAttrName(), resultParamsAttr);
  FunctionType functionType =
      builder.getFunctionType(argTypes, {resultType.mlirType});
  attrs.set(funcOp.getFunctionTypeAttrName(), TypeAttr::get(functionType));

  // Compute the signature of the function.
  auto signature = IndexRefRemapper::remapToSignature(
      inputParamsAttr, resultParamsAttr, functionType,
      builder.getAttr<MetadataAttr>(inputConventions, defaults, effects),
      [&] { return mlir::emitError(funcOp.getLoc()); });
  if (!signature)
    return failure();

  attrs.set(funcOp.getSignatureAttrName(), TypeAttr::get(signature));

  // Set the symbol to the mangled name and check for redefinition.
  attrs.set(funcOp.getSymNameAttrName(), getMangledName(baseName, signature));

  // Remove the temporary "sym_namex" attribute set up in FuncOp::build, see
  // that method for an explanation.
  attrs.erase("sym_namex");

  // Bulk update the attributes.
  funcOp->setAttrs(attrs.getDictionary(funcOp.getContext()));

  // Set the symbol and notice if we are redeclaring something.
  if (Operation *existing = finalizeFuncSignature(funcOp, decl)) {
    const char *errorMessage = nullptr;
    auto existingFunc = cast<LIT::FuncOp>(existing);
    if (existingFunc.getResultType() != funcOp.getResultType()) {
      errorMessage = " cannot overload on return type only";
    } else if (existingFunc.getIsAdaptive()) {
      // If the thing is adaptive and exact matches, then we actually don't want
      // to error.
    } else {
      errorMessage = " with identical signature";
    }

    // On redefinition this is an overload of the same name.
    if (errorMessage) {
      auto diag = p.emitError(funcOp.getLoc(), "redefinition of function ")
                  << baseName << errorMessage;
      diag.attachNote(existing->getLoc()) << "previous definition here";
      decl.hasReferenceError = true;
    }
  }

  // If have a main function, fn main(), export it automatically.
  if (!inAStruct && isMainFunction(baseName, funcOp, shared))
    getDeclResolver().exportMain(decl);

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
        getContext(), llvm::map_to_vector(argTypes, mapUnresolvedType),
        mapUnresolvedType(resultType.mlirType));
    diScopeGuard = diBuilder->pushSubprogram(
        baseName, funcOp.getNameAttr(), diBuilder->createFile(fileLineCol),
        fileLineCol.getLine(), fileLineCol.getLine(), spFlags, type);
    funcOp->setLoc(diBuilder->createScopedLoc(fileLineCol));
  }

  // FIXME: Handle "late" decorators somehow else. They should be "body
  // decorators" that are applied after the decl is fully resolved.
  for (auto [decorator, cursor] : decoratorExprs) {
    FnDecorators(decl, shared).applyLate(baseName, decorator, signature);
    if (funcOp.getSignature() != signature)
      funcOp.setSignature(signature);
  }

  // If this is a nested function, set its parameter declaration. It will be
  // referenced via parameter references instead of symbol references.
  if (funcOp->getParentOfType<LIT::FuncOp>())
    funcOp.setParamDeclAttr(
        ParamDeclAttr::get(funcOp.getSymNameAttr(), signature));

  funcOp.getBody()->addArguments(argTypes, argLocs);

  // If the user tried to mark a transitive capturing closure as thin, emit an
  // error.
  if (transivelyCaptures && !signature.isCapturing())
    return p.emitError(funcOp.getLoc(), "cannot mark a function with capturing "
                                        "closure parameters as @noncapturing");

  if (!funcOp->getParentOfType<FuncOp>() || !signature.isCapturing())
    funcOp.setIsParametric(true);

  // Upon fully resolving a nonparametric closure, immediately materialize it
  // as a runtime value. It cannot be used as a parameter.
  if (!funcOp.getIsParametric()) {
    // Fully resolve the body so we can swap the IR value of the decl. Later on,
    // we will need this to determine the capture signature.
    decl.resolvedness = DeclResolvedness::signature;
    if (failed(resolveBody(funcOp, lexer, decl)))
      return failure();
    decl.resolvedness = DeclResolvedness::fully;
    if (failed(recursivelyResolveFully(decl, decl.getLoc())))
      return failure();

    // If the function doesn't actually capture anything, don't demote it to a
    // runtime value.
    bool hasCapture = false;
    mlir::visitUsedValuesDefinedAbove(funcOp.getBodyRegion(),
                                      [&](OpOperand *) { hasCapture = true; });
    if (!hasCapture)
      return success();

    if (funcOp.getIsAdaptive()) {
      decl.hasReferenceError = true;
      return emitError(funcOp.getLoc(),
                       "nonparametric closure cannot be marked @adaptive");
    }
    if (!inputParamDecls.empty() || !resultParamDecls.empty()) {
      emitError(funcOp.getLoc(),
                "nonparametric closure cannot have input or result parameters");
    }

    OpBuilder b(funcOp.getContext());
    b.setInsertionPointAfter(funcOp);
    decl.irValue = SBValue(b.create<CreateClosureOp>(
        funcOp.getLoc(), funcOp.getSignature(),
        ParamDeclRefAttr::get(*funcOp.getParamDecl()), ValueRange()));
  }

  return success();
}

/// Create a mutable VarDecl for a function argument that captures its value.
/// argValue specifies the argument with the correct valuetype.
static SLValue makeArgLValueVarSlot(CValue argValue, StringAttr argName,
                                    ASTDecl &parentDecl, OpBuilder &builder,
                                    SMLoc loc, SharedState &shared) {
  // Emit the initializer expression into the slot.
  ExprEmitter emitter(shared, parentDecl, builder, /*varDeclCursor*/ nullptr);

  ASTType declType = argValue.getRValueType();
  Type varType = POP::PointerType::get(declType);
  auto varDecl = builder.create<VarLetDeclOp>(shared.translateLocation(loc),
                                              varType, argName,
                                              /*isVar*/ true,
                                              /*isSynthesized*/ true);

  // Expr to provide location information.
  DeclRefNode srcExpr(StringRef(loc.getPointer(), argName.size()));
  ValueDest dest(SLValue(varDecl), EC_DefArgumentShadow);
  if (!emitter.emitBValue({argValue, &srcExpr}, dest))
    dest.resetForError();

  return SLValue(varDecl);
};

/// Emit a normal return (not a 'raise' return) out of the function, along with
/// any special logic that goes with it.
void ExprEmitter::emitNormalReturn(OpBuilder &builder, Location loc,
                                   Value value, const ASTDecl &funcDecl) {
  switch (funcDecl.getSpecialFunctionKind()) {
  default:
    break;

  /// In the __del__ method for a struct, we need to mark 'self' as being
  /// destroyed before any return operation.
  case SpecialFunctionKind::kDel: {
    auto func = cast<LIT::FuncOp>(funcDecl);
    assert(func.getBody()->getNumArguments() == 1 &&
           "__del__ should have one argument");
    Value selfArg = func.getBody()->getArgument(0);

    // If this is a @register_passable type, the value will be stored in a
    // box and we want to treat the box as the thing that we track.
    if (func.getSignature().getInputConvention(0) ==
        ValueInputConvention::OwnedInReg) {
      // Find the single store and ignore debug.value operations.
      POP::StoreOp store;
      for (auto user : selfArg.getUsers()) {
        if (isa<DebugInfo::ValueOp>(user))
          continue;
        assert(!store && "Should only have a single store");
        store = cast<POP::StoreOp>(user);
      }
      selfArg = store.getPtr();
    }
    builder.create<LIT::OwnershipMarkDestroyedOp>(loc, selfArg);
    break;
  }

  /// In the __moveinit__ method for a struct, we need to mark 'existing' as
  /// being destroyed before any return operation if it is owned convention.
  case SpecialFunctionKind::kMoveInit: {
    auto func = cast<LIT::FuncOp>(funcDecl);
    assert(func.getBody()->getNumArguments() == 2 &&
           "__moveinit__ should have to arguments");
    // Don't change `__moveinit__(owned self, inout existing: Self)`.
    if (func.getSignature().getInputConvention(1) !=
        ValueInputConvention::OwnedInMem)
      break;

    Value existingArg = func.getBody()->getArgument(1);
    builder.create<LIT::OwnershipMarkDestroyedOp>(loc, existingArg);
    break;
  }
  }

  // Finally we emit a normal return with lit.return.
  builder.create<LIT::ReturnOp>(loc, value);
}

/// This adds a default return (lit.return of None, potentially converted
/// to a variant) and emits a EndFuncOp.
static void appendDefaultReturnAndEndOp(LIT::FuncOp func, ASTDecl &funcDecl,
                                        SharedState &shared) {
  Block &body = *func.getBody();
  auto b = OpBuilder::atBlockEnd(&body);
  Location loc = func.getLoc();

  auto makeNoneReturn = [&] {
    // The function returns none.
    Value retVal = b.create<ParamConstantOp>(loc, b.getAttr<LIT::NoneAttr>());

    // Wrap the result value if necessary.
    if (func.isThrows())
      retVal =
          b.create<POP::VariantCreateOp>(loc, func.getResultType(), retVal);
    ExprEmitter::emitNormalReturn(b, loc, retVal, funcDecl);
  };

  // If the function returns None, insert a "return None".
  Type normalResult = func.getResultTypeWithoutErrorVariant();
  if (isa<LIT::NoneType>(normalResult) &&
      !func.getSignature().hasMemoryOnlyResult() &&
      // No default return needed if we ended in a return.
      (body.empty() || !isa<LIT::ReturnOp>(body.back()))) {
    makeNoneReturn();
  } else if (func.getIsDef() && func.getSignature().hasMemoryOnlyResult()) {
    // If this `def` returns an object but is missing a return, insert one
    // automatically.
    auto objType = shared.lookupObjectType(funcDecl.getLoc(), funcDecl);
    if (objType && objType.isEqualCanon(
                       cast<POP::PointerType>(func.getArgument(0).getType())
                           .getElementType())) {
      // Emit `object()` into the memory type return slot.
      ExprEmitter emitter(shared, funcDecl, EC_ReturnValue,
                          /*varDeclCursor=*/nullptr);
      emitter.builder = b;
      ValueDest resultDest(SLValue(func.getArgument(0)), EC_ReturnValue);
      // Create a dummy node to pass down.
      ExprNode *noneExpr = shared.allocPersistent<SimpleLiteralNode>(
          ExprNode::kNoneLiteral, funcDecl.getLoc());
      CValue result = emitter.emitConstructorCall(
          objType, {}, noneExpr, CallSyntax::kImplicitConvert, resultDest);
      if (!result || !emitter.emitResult(result, noneExpr, resultDest))
        resultDest.resetForError();
      else
        makeNoneReturn();
    }
  }

  // Insert the default end terminator.
  b.create<LIT::EndFuncOp>(loc);
}

ParseResult DeclResolver::resolveBody(LIT::FuncOp funcOp, Lexer &lexer,
                                      ASTDecl &decl) {
  // Push the debug scope for this function if necessary so that nested
  // operations have proper debug info.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (auto spAttr = DebugInfo::extractScope(funcOp))
    diScopeGuard = shared.diBuilder->pushScopeGuard(spAttr);

  // Set up information about for value arguments.
  Block *bodyBlock = funcOp.getBody();
  auto builder = OpBuilder::atBlockEnd(bodyBlock);

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

  SignatureType funcSignature = funcOp.getSignature();

  // Set up the body of the fn/def, creating declarations for the value
  // parameters and adding them to the symbol table.
  for (auto [argIndex, argName, bbArg, convention] :
       llvm::zip(llvm::detail::index_stream(), funcOp.getValueParamNames(),
                 funcOp.getBody()->getArguments(),
                 funcSignature.getValueInputConventions())) {

    // Don't bind byref-result, it is handled specially by 'return'.
    if (convention == ValueInputConvention::ByRefResult)
      continue;

    // Figure out which decl corresponds to this argument so we can finish it.
    ArrayRef<ASTDecl *> argDeclList = decl.lookupInCurrentScope(argName);
    assert(argDeclList.size() == 1 &&
           "Argument should be added by signature resolution");
    ASTDecl &argDecl = *argDeclList[0];

    // This function sets the argument decl to be fully resolved with the
    // specified IR representation.
    auto setDecl = [&](DeclIRValue value) {
      argDecl.setIRValue(value);
      argDecl.resolvedness = DeclResolvedness::fully;
      if (auto rv = argDecl.getIfRValue()) {
        if (isa<TypeCheckErrorType>(rv.getType().mlirType))
          argDecl.hasReferenceError = true;
      } else if (auto lv = argDecl.getIfLValue()) {
        if (isa<TypeCheckErrorType>(lv.getRValueType().mlirType))
          argDecl.hasReferenceError = true;
      } else if (auto bv = argDecl.getIfBValue()) {
        if (isa<TypeCheckErrorType>(bv.getRValueType().mlirType))
          argDecl.hasReferenceError = true;
      }
    };

    buildArgDIInfo(bbArg, argName, argIndex);

    // VarArg arguments are always treated as their pop.variadic type
    // by-value right now.  TODO(literals): Project to a list like thing.
    if (funcSignature.isVararg(argIndex) ||
        isa<POP::PackType>(bbArg.getType())) {
      setDecl(SRValue(bbArg));
      continue;
    }

    DeclIRValue argIRValue;
    switch (convention) {
    // Arguments passed by-reference can be directly used.
    case ValueInputConvention::ByRef:
    case ValueInputConvention::ByRefResult:
    case ValueInputConvention::InitSelf:
    case ValueInputConvention::OwnedInMem:
      // OwnedInMem passes ownership of the argument into the callee so we
      // can directly mutate it if we want to.
      argIRValue = SLValue(bbArg);
      break;

    case ValueInputConvention::OwnedInReg:
      argIRValue = makeArgLValueVarSlot(SRValue(bbArg), argName, decl, builder,
                                        argDecl.getLoc(), shared);
      break;

    case ValueInputConvention::BorrowedInReg:
    case ValueInputConvention::BorrowedInMem:
      // If this was passed by-value, then it becomes an rvalue in a `fn`.
      if (convention == ValueInputConvention::BorrowedInMem)
        argIRValue = MBValue(bbArg);
      else
        argIRValue = SBValue(bbArg);
      if (!funcOp.getIsDef())
        break;

      // In a `def`, we create a mutable var.decl lvalue to allow reassignment.
      // Figure out how to model the input value.
      CValue srcVal;
      if (convention == ValueInputConvention::BorrowedInMem)
        srcVal = MBValue(bbArg);
      else
        srcVal = SBValue(bbArg);

      // Check that the value is copyable - if not we want to emit a specific
      // error.
      if (!srcVal.getRValueType().isCopyable(argDecl.getLoc(), shared)) {
        auto diag =
            emitError(argDecl.getLoc())
            << "'def' requires argument type " << srcVal.getRValueType()
            << " to be copyable, but it doesn't provide a '__copyinit__' "
               "method";
        diag.attachNote(argDecl.getLoc())
            << "consider passing by reference instead"
            << FixIt::insertBeforeToken(argDecl.getLoc(), "inout ");
        break;
      }

      argIRValue = makeArgLValueVarSlot(srcVal, argName, decl, builder,
                                        argDecl.getLoc(), shared);
      break;
    }

    // Ok, now that we've figured out the IR representation of the ASTDecl,
    // install it.
    setDecl(argIRValue);
  }

  // With all the argument declarations set up, we can resolve the body of the
  // function.
  if (ParserBase::parseSuite(decl, lexer))
    return failure();

  auto loc = funcOp.getLoc();

  // Create a placeholder result bind op if the function has result parameters.
  ArrayRef<ParamDeclAttr> resultParams = funcOp.getResultParams();
  if (!resultParams.empty()) {
    SmallVector<TypedAttr> placeholders;
    for (ParamDeclAttr decl : resultParams)
      placeholders.push_back(UnknownAttr::get(decl.getType()));
    builder.create<ParamResultBindOp>(loc, placeholders);
  }

  // Emit a default "return None" if the function returns nothing, and add an
  // endop terminator.
  appendDefaultReturnAndEndOp(funcOp, decl, shared);

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

LogicalResult DeclResolver::recursivelyResolveFully(ASTDecl &decl,
                                                    llvm::SMLoc loc) {
  // Collect decls currently in scope.
  std::vector<ASTDecl *> initialDecls;
  for (auto &[name, decls] : decl.declsInScope)
    for (ASTDecl *decl : decls)
      initialDecls.push_back(decl);
  // Start resolving the decls. If any more decls get added, keep resolving
  // them and no more are added.
  size_t start = parsedDeclList.size();
  ArrayRef<ASTDecl *> declsToResolve = initialDecls;
  while (!declsToResolve.empty()) {
    for (ASTDecl *decl : declsToResolve)
      if (failed(resolveFully(*decl, loc)))
        return failure();
    declsToResolve = ArrayRef(parsedDeclList).drop_front(start);
    start = parsedDeclList.size();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Module Decl implementation
//===----------------------------------------------------------------------===//

ParseResult DeclResolver::resolveBody(LIT::FileModuleOp op, Lexer &lexer,
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

  return ParserBase::parseSuite(decl, lexer);
}

//===----------------------------------------------------------------------===//
// VarLetDecl implementation
//===----------------------------------------------------------------------===//

/// var_decl_stmt ::= var_or_let identifier ":" expression ["=" expression]
///                 | var_or_let identifier "=" expression
/// var_or_let    ::= "var" | "let"
LogicalResult DeclResolver::resolveSignature(VarLetDeclOp varOp, Lexer &lexer,
                                             ASTDecl &decl) {
  ParserBase p(lexer);
  auto decorators = p.parseDecorators(decl);

  p.consumeToken(); // eat the let/var.
  if (p.parseToken(Token::identifier, "internal error: checked by stmt parser"))
    return failure();

  //  Parse the type if present.
  ASTType parsedType;
  if (p.consumeIf(Token::colon)) {
    if (parseType(p, parsedType, *decl.getParentDecl(), decl.getIndentation()))
      return failure();
  }

  // Parse the initializer if present.
  ExprNode *initExpr = nullptr;
  if (p.consumeIf(Token::equal)) {
    if (p.parseExpression(initExpr, decl.getIndentation()))
      return failure();
  }

  // Now that parsing succeeded, we do IR emission and semantic processing.

  // Handle the initializer if present.
  if (initExpr) {
    // Insert before the var decl op. Ops can get deleted, so we have to ensure
    // the insertion point is stable.
    OpBuilder builder(varOp);
    ExprEmitter emitter(shared, *decl.getParentDecl(), builder,
                        /*varDeclCursor*/ nullptr);

    // If we have a type, then emit directly into the LValue.  Otherwise emit
    // into the varOp to infer its type.
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

    // Now move the var decl op to the end of the initializer IR.
    assert(varOp->hasOneUse() && "Should have one use");
    varOp->moveBefore(*varOp->user_begin());

    assert(!isa<UnresolvedType>(varOp.getType().getElementAsType()) &&
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

  // Now that this has been fully checked, we can promote to a LetRegDeclOp
  // if this was a non-parameteric register-passable `let` declaration with
  // an initializer.  We don't care about the address being available and
  // this produces smaller IR.
  ASTType inferredRValueType = ASTType(varOp.getType()).getPointerElementType();
  if (initExpr && !varOp.getIsVar() &&
      // NOTE: This is assuming type parameters are valid register types.  We
      // will need to build out better support when we have traits, but this is
      // important for kernels in practice today.
      inferredRValueType.isRegisterPassable(initExpr->getLoc(), shared)) {
    // There should be exactly one store to the original op, sanity check this.
    assert(varOp->hasOneUse() && "Should have one store use");
    auto theStore = cast<POP::StoreOp>(*varOp->user_begin());

    // Create new LetRegDeclOp and put it into the ASTDecl.
    OpBuilder builder(theStore);
    auto newLetOp = builder.create<LetRegDeclOp>(
        varOp.getLoc(), varOp.getNameAttr(), theStore.getArg());
    decl.setIRValue(newLetOp.getOperation());

    // Remove the store and the original VarLetDeclOp.
    theStore->erase();
    varOp->erase();
  }

  return success();
}

ParseResult DeclResolver::resolveBody(VarLetDeclOp op, Lexer &lexer,
                                      ASTDecl &decl) {
  return success();
}

ParseResult DeclResolver::resolveBody(LetRegDeclOp op, Lexer &lexer,
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
                                             Lexer &lexer, ASTDecl &decl) {
  ParserBase p(lexer);
  auto decoratorExprs = p.parseDecorators(decl);

  // Parse the type if present.
  if (p.parseToken(Token::kw_alias, "internal error: checked by stmt parser") ||
      p.parseToken(Token::identifier, "internal error: checked by stmt parser"))
    return failure();

  ASTType type;
  if (p.consumeIf(Token::colon)) {
    if (parseType(p, type, *decl.getParentDecl(), decl.getIndentation()))
      return failure();
  }

  // Handle the case where there is no initializer.
  if (!p.consumeIf(Token::equal)) {
    // If there was neither a type or initializer, reject the var.
    if (!type) {
      p.emitError(paramDeclOp.getLoc(),
                  "declaration must have either a type or an initializer");
      return failure();
    }

    // `alias x: Int` is a forward declaration of a return parameter from a
    // function call, so it must occur in a function.
    if (!paramDeclOp->getParentOfType<LIT::FuncOp>()) {
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

    // Process the doc string of the alias.
    p.parseDocString(decl);
    return success();
  }

  // Otherwise this is a normal `alias` declaration with an initializer.
  ExprNode *initExpr = nullptr;
  if (p.parseExpression(initExpr, decl.getIndentation()))
    return failure();

  ASTDecl &parentDecl = *decl.getParentDecl();
  ExprEmitter emitter(shared, parentDecl, EC_AliasValue,
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

  // Process the doc string of the alias.
  p.parseDocString(decl);
  return success();
}

ParseResult DeclResolver::resolveBody(ParamDeclareOp op, Lexer &lexer,
                                      ASTDecl &decl) {
  return success();
}

ParseResult DeclResolver::resolveBody(AliasForwardDeclOp aliasFwdDeclOp,
                                      Lexer &lexer, ASTDecl &decl) {
  return success();
}

//===----------------------------------------------------------------------===//
// Struct Decl implementation
//===----------------------------------------------------------------------===//

/// Process a decorator that is resolved at the signature phase of resolution
/// and return true, otherwise return false if it is an unknown or body
/// decorator.
static bool processStructSignatureDecorator(ExprNode *decorator,
                                            StructDeclOp structOp,
                                            DeclResolver &resolver) {
  if (auto declRef = dyn_cast<DeclRefNode>(decorator)) {
    if (declRef->spelling == "register_passable") {
      structOp.setRegisterPassable(StructDeclOp::RP_RegisterPassable);
      return true;
    }
  }

  // `x()` forms.
  if (auto callNode = dyn_cast<CallNode>(decorator)) {
    if (auto declRef = dyn_cast<DeclRefNode>(callNode->callee)) {
      // @register_passable("trivial")
      if (declRef->spelling == "register_passable" &&
          callNode->args.size() == 1 &&
          callNode->args[0].isPositionalStringLiteral("trivial")) {
        structOp.setRegisterPassable(StructDeclOp::RP_RegisterPassableTrivial);
        return true;
      }
    }
  }
  // Not handled in signature phase.
  return false;
}

/// structdef ::=
///   [decorators] "struct" identifier [meta_signature] ":" suite
///
LogicalResult DeclResolver::resolveSignature(StructDeclOp structOp,
                                             Lexer &lexer, ASTDecl &decl) {
  ParserBase p(lexer);
  auto decoratorExprs = p.parseDecorators(decl);

  SmallVector<ParamDeclAttr> inputParamDecls;
  SmallVector<ParamDeclAttr> resultParamDecls;
  bool paramVarargs = false;
  if (p.parseToken(Token::kw_struct,
                   "internal error: checked by stmt parser") ||
      p.parseToken(Token::identifier,
                   "internal error: checked by stmt parser") ||
      parseOptionalParameterSignature(p, decl, inputParamDecls,
                                      resultParamDecls, paramVarargs) ||
      p.parseToken(Token::colon, "expected ':' in struct definition"))
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
  structOp.setRegisterPassable(StructDeclOp::RP_MemoryOnly);

  // Now that we have the basic struct set up, process signature decorators.
  SmallVector<LexerCursor> bodyDecorators;
  for (auto [decorator, cursor] : decoratorExprs)
    if (!processStructSignatureDecorator(decorator, structOp, *this))
      bodyDecorators.push_back(cursor);
  decl.setBodyDecorators(bodyDecorators, shared);
  return success();
}

/// This method creates a FuncOp for a method inside of a struct with the
/// specified value signature information.  It handles the mechanics of creating
/// the function but also of registering it with the DeclResolver.
static std::pair<LIT::FuncOp, ASTDecl &>
synthesizeMethodInStruct(StringRef name, ArrayRef<Type> argTypes,
                         ArrayRef<ValueInputConvention> argConventions,
                         ArrayRef<StringAttr> argNames, Type resultType,
                         ImplicitLocOpBuilder &builder, ASTDecl &structDecl,
                         DeclResolver &resolver) {
  StructDeclOp structOp = cast<StructDeclOp>(structDecl);

  // Get the signature for the function.
  auto fnType = builder.getFunctionType(argTypes, resultType);

  FnEffects fnEffects = FnEffects();
  // If the result of the function is a non-trivial type, mark the function
  // effect as having an owned result so ownership tracking will notice it.
  if (!ASTType(resultType).isTrivial(structDecl.getLoc(), resolver.shared))
    fnEffects = fnEffects | FnEffects::OwnedResult;

  // TODO: Should raise if anything we invoke raises.
  auto metadata = builder.getAttr<MetadataAttr>(
      argConventions, /*no default args=*/ArrayRef<TypedAttr>(), fnEffects);
  auto signature = SignatureType::get({}, {}, fnType, metadata);

  // Create the empty function.
  StringAttr nameAttr = getMangledName(builder.getStringAttr(name), signature);
  auto funcOp = builder.create<LIT::FuncOp>(nameAttr, signature, argNames);

  // Register the method in the struct.
  ASTDecl &funcDecl = resolver.addFullyResolvedDecl(
      funcOp.getOperation(), name, structDecl.getLoc(), &structDecl);
  funcDecl.setSpecialFunctionKind(SpecialFunctionInfo::getKind(name));

  // Set the symbol and notice if we are redeclaring something.
  if (Operation *existing = resolver.finalizeFuncSignature(funcOp, funcDecl)) {
    resolver.emitError(
        existing->getLoc(),
        "internal compiler error: synthesized member that already exists");
  }

  // If the struct is register_passable("trivial"), make this
  // @always_inline("nodebug").
  if (structOp.getRegisterPassable() ==
      StructDeclOp::RP_RegisterPassableTrivial)
    funcOp.setAlwaysInlineLevel(AlwaysInlineLevel::EnabledNoDebug);

  return {funcOp, funcDecl};
}

/// Look up the __del__ destructor for the specified `type` which is needed
/// for the specified declaration (typically a var or argument declaration).
/// This returns the destructor if successful, diagnoses an error if not, and
/// returns null if there is no defined destructor.
static TypedAttr lookupDestructor(ASTDecl &structDecl, SharedState &shared) {
  auto dels = shared.lookupAndResolveDecl(
      "__del__", structDecl.getLoc(), structDecl, /*searchParentScopes=*/false);
  ArrayRef<ASTDecl *> entries = dels.getIfSuccess();
  // If there are no __del__ methods, return null.  This is valid.
  if (entries.empty())
    return {};
  if (entries.size() != 1) {
    auto diag = shared.emitError(structDecl.getLoc(),
                                 "invalid overloaded '__del__' method");
    for (auto candidate : entries)
      diag.attachNote(candidate->getLoc()) << "candidate declared here";
    return {};
  }
  ASTDecl &delDecl = *entries[0];
  LIT::FuncOp func = dyn_cast<LIT::FuncOp>(delDecl);
  if (!func) {
    shared.emitError(delDecl.getLoc(), "'__del__' must be a method");
    return {};
  }
  return func.getBoundReference();
}

/// Given a struct that has no explicitly defined __del__ member, define a new
/// one with an empty body.  This allows the CheckLifetimes pass to insert field
/// dels as needed, and makes sure that anything that refers to this struct
/// properly runs its destructor.
static TypedAttr synthesizeEmptyDtor(StructDeclOp structOp, ASTDecl &structDecl,
                                     DeclResolver &resolver) {
  auto builder = ImplicitLocOpBuilder::atBlockEnd(
      structOp.getLoc(), &structOp.getFields().front());

  // Figure out the type of the 'self' argument.  It is the struct's `Self`
  // type for register passable things, or indirect for a memory-only type.
  ASTType selfType = structDecl.getSelfType();
  // The argument is always owned.
  ValueInputConvention convention = ValueInputConvention::OwnedInReg;
  if (!selfType.isRegisterPassable(structDecl.getLoc(), resolver.shared)) {
    selfType = POP::PointerType::get(selfType);
    convention = ValueInputConvention::OwnedInMem;
  }

  StringAttr selfName = builder.getStringAttr("self");

  // Create the FuncOp and ASTDecl for the method.
  auto [funcOp, funcDecl] = synthesizeMethodInStruct(
      "__del__", selfType.mlirType, convention, selfName,
      resolver.shared.getNoneType(), builder, structDecl, resolver);

  // Set up the body.
  Block *body = funcOp.getBody();
  BlockArgument arg = body->addArgument(selfType, structOp.getLoc());

  // We need to make a var box + store for register_passable values since that
  // is what lifetime tracking expects.
  if (convention == ValueInputConvention::OwnedInReg) {
    builder.setInsertionPointToStart(body);
    (void)makeArgLValueVarSlot(SRValue(arg), selfName, funcDecl, builder,
                               structDecl.getLoc(), resolver.shared);
  }

  // Finish off the function with a return + lit.endfunc.
  appendDefaultReturnAndEndOp(funcOp, funcDecl, resolver.shared);

  return funcOp.getBoundReference();
}

struct StructBodyDecorators : public SharedStateUser {
  StructBodyDecorators(
      StructDeclOp structOp, ASTDecl &structDecl, DeclResolver &resolver,
      ArrayRef<std::pair<StructFieldOp, ASTDecl *>> structFields)
      : SharedStateUser(resolver.shared), structOp(structOp),
        structDecl(structDecl), resolver(resolver), structFields(structFields) {
  }

  void processBodyDecorator(LexerCursor decoratorCursor) {
    // Don't run decorators if the struct is invalid.
    if (structDecl.hasReferenceError)
      return;

    Lexer lexer(resolver.shared, decoratorCursor);
    ParserBase parser(lexer);
    ExprNode *expr = nullptr;
    if (failed(parser.parseExpression(expr, structDecl.getIndentation())))
      return;

    processDecorator(expr);
  }

private:
  void processValueDecorator(SMLoc decoratorLoc);
  void processRegisterPassableDecorator(bool isTrivial);
  void processDecorator(ExprNode *expr);

  StructDeclOp structOp;
  ASTDecl &structDecl;
  DeclResolver &resolver;
  ArrayRef<std::pair<StructFieldOp, ASTDecl *>> structFields;
};

/// Check to see if the specified struct already has a memberwise initializer
/// with the specified fields.
static bool
hasMemberwiseInit(SMLoc loc, ASTDecl &structDecl, bool isMemoryOnly,
                  ArrayRef<std::pair<StructFieldOp, ASTDecl *>> structFields,
                  SharedState &shared) {
  LookupResult inits =
      shared.lookupAndResolveDecl("__init__", loc, structDecl,
                                  /*searchParentScopes=*/false);
  // Check all the available candidates to see if they have any memberwise inits
  // yet.
  for (ASTDecl *decl : inits.getIfSuccess()) {
    // Don't synthesize if we see anything fishy.
    if (decl->hasReferenceError)
      return true;
    auto func = dyn_cast<LIT::FuncOp>(*decl);
    if (!func)
      continue;
    auto signature = func.getSignature();

    ArrayRef<Type> inputTypes = signature.getValueInputs();
    ArrayRef<ValueInputConvention> convs = signature.getValueInputConventions();

    // If this is @register_passable struct, we'd have an init like:
    //   fn __init__(field1: Int, field2: Mem) -> Self
    // If memory-only, we should have:
    //   fn __init__(inout self, field1: Int, field2: Mem)
    // The result type of all inits / self are already checked.
    if (isMemoryOnly) {
      inputTypes = inputTypes.drop_front();
      convs = convs.drop_front();
    }
    // TODO: Handle default arguments.
    if (inputTypes.size() != structFields.size())
      continue;

    bool isMatch = true;
    for (auto [type, conv, field] :
         llvm::zip(inputTypes, convs, structFields)) {
      // Strip the pointer type if present.
      Type argType = type;
      if (conv != ValueInputConvention::OwnedInReg &&
          conv != ValueInputConvention::BorrowedInReg)
        argType = ASTType(argType).getPointerElementType();
      StructFieldOp op = field.first;
      if (argType != op.getType()) {
        isMatch = false;
        break;
      }
    }
    if (isMatch)
      return true;
  }

  return false;
}

/// This synthesizes an __init__ method that accepts values for every field of
/// a struct, making it easy for external clients to initialize it.
static void synthesizeMemberwiseInit(
    SMLoc decoratorLoc, ASTDecl &structDecl, bool isMemoryOnly,
    ArrayRef<std::pair<StructFieldOp, ASTDecl *>> structFields,
    DeclResolver &resolver) {
  StructDeclOp structOp = cast<StructDeclOp>(structDecl);
  auto builder = ImplicitLocOpBuilder::atBlockEnd(
      resolver.translateLocation(decoratorLoc), &structOp.getFields().front());

  SmallVector<Type> argTypes;
  SmallVector<ValueInputConvention> argConventions;
  SmallVector<StringAttr> argNames;
  Type resultType;

  // Figure out the type of the 'self' argument/result.
  ASTType selfType = structDecl.getSelfType();
  if (isMemoryOnly) {
    argTypes.push_back(POP::PointerType::get(selfType));
    argConventions.push_back(ValueInputConvention::InitSelf);
    argNames.push_back(builder.getStringAttr("self"));
    resultType = resolver.shared.getNoneType();
  } else {
    resultType = selfType;
  }

  // We declare all of the operands to the init constructor as owned.  This
  // enables it to work with move-only fields, and, for copyable types, forces
  // the copy into the caller, which can then be elided with a consume or
  // RValue.
  for (auto [fieldOp, fieldDecl] : structFields) {
    ASTType fieldType = fieldOp.getType();
    ValueInputConvention conv;
    switch (fieldType.getRegisterPassability(decoratorLoc, resolver.shared)) {
    default:
      llvm_unreachable("unknown case");
    case StructDeclOp::RP_MemoryOnly:
      fieldType = POP::PointerType::get(fieldType);
      conv = ValueInputConvention::OwnedInMem;
      break;
    case StructDeclOp::RP_RegisterPassable:
      conv = ValueInputConvention::OwnedInReg;
      break;
    case StructDeclOp::RP_RegisterPassableTrivial:
      conv = ValueInputConvention::BorrowedInReg;
      break;
    }
    argTypes.push_back(fieldType);
    argConventions.push_back(conv);
    argNames.push_back(fieldOp.getNameAttr());
  }

  // Create the FuncOp and ASTDecl for the method.
  auto [funcOp, funcDecl] =
      synthesizeMethodInStruct("__init__", argTypes, argConventions, argNames,
                               resultType, builder, structDecl, resolver);

  // Set up the body.
  Block *body = funcOp.getBody();
  builder.setInsertionPointToStart(body);
  ExprEmitter emitter(resolver.shared, funcDecl, builder,
                      /*varDeclCursor*/ nullptr);

  // For a memory-only initializer, we emit a bunch of stores to fields indexing
  // self.
  if (isMemoryOnly) {
    BlockArgument selfArg = body->addArgument(argTypes[0], funcOp.getLoc());
    for (size_t idx = 1, e = argTypes.size(); idx != e; ++idx) {
      // Add the block argument, get it as an RValue since it is owned.
      BlockArgument arg = body->addArgument(argTypes[idx], funcOp.getLoc());
      CValue argVal;
      if (argConventions[idx] == ValueInputConvention::OwnedInReg)
        argVal = SRValue(arg);
      else if (argConventions[idx] == ValueInputConvention::BorrowedInReg)
        argVal = SBValue(arg);
      else
        argVal = MRValue(arg);

      // Project self to the right field and store the RValue.
      StructFieldOp field = structFields[idx - 1].first;
      auto fieldPtr = builder.create<StructGEPOp>(selfArg, field);
      DeclRefNode srcExpr(StringRef(decoratorLoc.getPointer(), 1));
      emitter.emitStoreToLValue({argVal, &srcExpr}, SLValue(fieldPtr),
                                EC_AttributeRefBase);
    }

    // Finish off the function with a return + lit.endfunc.
    appendDefaultReturnAndEndOp(funcOp, funcDecl, resolver.shared);
    return;
  }

  funcOp.setIsStatic(true);

  // Otherwise, emit all the values and finish with a struct create.  We know
  // all the subfields must be register passable.
  SmallVector<Value> fieldVals;
  for (size_t idx = 0, e = argTypes.size(); idx != e; ++idx) {
    // Add the block argument, get it as an RValue since it is owned.
    BlockArgument arg = body->addArgument(argTypes[idx], structOp.getLoc());
    fieldVals.push_back(arg);
  }

  auto result = SRValue(builder.create<StructCreateOp>(
      selfType.mlirType, fieldVals,
      StringArrayAttr::get(emitter.getContext(), argNames)));

  ExprEmitter::emitNormalReturn(builder, structOp.getLoc(), result, funcDecl);
  builder.create<LIT::EndFuncOp>();
}

/// This synthesizes a __copyinit__/__moveinit__ method that recursively
/// copies/moves each field of a struct.
static void synthesizeCopyMoveInit(
    bool isMove, SMLoc decoratorLoc, ASTDecl &structDecl, bool isMemoryOnly,
    ArrayRef<std::pair<StructFieldOp, ASTDecl *>> structFields,
    DeclResolver &resolver) {
  StructDeclOp structOp = cast<StructDeclOp>(structDecl);
  auto builder = ImplicitLocOpBuilder::atBlockEnd(
      resolver.translateLocation(decoratorLoc), &structOp.getFields().front());

  // The signature of __copyinit__ is either:
  //   fn __copyinit__(inout self, borrowed existing: Self) -> None
  //   fn __copyinit__(borrowed existing: Self) -> Self
  // The signature of __moveinit__ (only for memory types) is:
  //   fn __moveinit__(inout self, owned existing: Self) -> None
  SmallVector<Type> argTypes;
  SmallVector<ValueInputConvention> argConventions;
  SmallVector<StringAttr> argNames;
  Type resultType;

  // Figure out the type of the 'self' argument/result.
  ASTType selfType = structDecl.getSelfType();
  if (isMemoryOnly) {
    argNames.push_back(builder.getStringAttr("self"));
    argTypes.push_back(POP::PointerType::get(selfType));
    argConventions.push_back(ValueInputConvention::InitSelf);

    argNames.push_back(builder.getStringAttr("existing"));
    argTypes.push_back(POP::PointerType::get(selfType));
    argConventions.push_back(isMove ? ValueInputConvention::OwnedInMem
                                    : ValueInputConvention::BorrowedInMem);
    resultType = resolver.shared.getNoneType();
  } else {
    argNames.push_back(builder.getStringAttr("existing"));
    argTypes.push_back(selfType);
    argConventions.push_back(ValueInputConvention::BorrowedInReg);
    resultType = selfType;
  }

  // Create the FuncOp and ASTDecl for the method.
  auto [funcOp, funcDecl] = synthesizeMethodInStruct(
      isMove ? "__moveinit__" : "__copyinit__", argTypes, argConventions,
      argNames, resultType, builder, structDecl, resolver);

  // Set up the body.
  Block *body = funcOp.getBody();
  builder.setInsertionPointToStart(body);
  ExprEmitter emitter(resolver.shared, funcDecl, builder,
                      /*varDeclCursor*/ nullptr);
  DeclRefNode srcExpr(StringRef(decoratorLoc.getPointer(), 1));

  // For a memory-only initializer, we emit a bunch of copies/moves to fields
  // indexing self.
  if (isMemoryOnly) {
    BlockArgument selfArg = body->addArgument(argTypes[0], funcOp.getLoc());
    BlockArgument existingArg = body->addArgument(argTypes[1], funcOp.getLoc());

    for (auto [fieldOp, fieldDecl] : structFields) {
      auto selfField = builder.create<StructGEPOp>(selfArg, fieldOp);
      auto existingField = builder.create<StructGEPOp>(existingArg, fieldOp);
      CValue src = isMove ? CValue(MRValue(existingField))
                          : CValue(MBValue(existingField));
      emitter.emitStoreToLValue({src, &srcExpr}, SLValue(selfField),
                                EC_AttributeRefBase);
    }

    // Finish off the function with a return + lit.endfunc.
    appendDefaultReturnAndEndOp(funcOp, funcDecl, resolver.shared);
    return;
  }

  funcOp.setIsStatic(true);

  // Otherwise, extract all the values and finish with a struct create.  We know
  // all the subfields must be register passable.
  BlockArgument existingArg = body->addArgument(argTypes[0], funcOp.getLoc());

  SmallVector<Value> fieldVals;
  SmallVector<StringAttr> fieldNames;
  for (auto [fieldOp, fieldDecl] : structFields) {
    auto extractVal = builder.create<StructExtractOp>(existingArg, fieldOp);
    // Emit an SBValue -> SRValue conversion to get ownership of the value.
    auto copiedVal =
        emitter.emitSRValue({SBValue(extractVal), &srcExpr}, EC_CallArgValue);
    if (!copiedVal)
      return;
    fieldVals.push_back(copiedVal);
    fieldNames.push_back(fieldOp.getNameAttr());
  }

  auto result = SRValue(builder.create<StructCreateOp>(
      selfType.mlirType, fieldVals,
      StringArrayAttr::get(emitter.getContext(), fieldNames)));

  ExprEmitter::emitNormalReturn(builder, structOp.getLoc(), result, funcDecl);
  builder.create<LIT::EndFuncOp>();
}

/// Process the @value body decorator on structs.  This synthesizes the
/// memberwise init, copy ctor and move ctor if requested.
void StructBodyDecorators::processValueDecorator(SMLoc decoratorLoc) {
  // Check to see the classification of the fields, the result type will be
  // copyable/movable iff all the fields are.
  bool isCopyable = true, isMovable = true;
  for (auto [fieldOp, fieldDecl] : structFields) {
    ASTType fieldType(fieldOp.getType());
    isCopyable &= fieldType.isCopyable(fieldDecl->getLoc(), shared);
    isMovable &= fieldType.isMovable(fieldDecl->getLoc(), shared);

    // If this field is neither copyable or movable, then we cannot do
    // anything in this decorator.
    if (!isCopyable && !isMovable) {
      auto diag =
          emitError(decoratorLoc, "'@value' cannot synthesize members: ")
          << fieldOp.getNameAttr() << " has non-copyable, non-movable type "
          << fieldType;
      diag.attachNote(fieldDecl->getLoc())
          << fieldOp.getNameAttr() << " declared here";
      return;
    }
  }

  bool isMemoryOnly =
      structOp.getRegisterPassable() == StructDeclOp::RP_MemoryOnly;

  // Ok, we know the struct has copyable or movable fields.  Check to see if
  // it has a memberwise initializer.
  if (!hasMemberwiseInit(decoratorLoc, structDecl, isMemoryOnly, structFields,
                         shared)) {
    synthesizeMemberwiseInit(decoratorLoc, structDecl, isMemoryOnly,
                             structFields, resolver);
  }

  // If the struct is not already copyable, but its members are, add a
  // __copyinit__ method.
  if (isCopyable && !structDecl.getSelfType().isCopyable(decoratorLoc, shared))
    synthesizeCopyMoveInit(/*isMove=*/false, decoratorLoc, structDecl,
                           isMemoryOnly, structFields, resolver);

  // If the struct is not already movable and is memory-only, synthesize a move
  // operation.
  if (isMemoryOnly && isMovable &&
      !structDecl.getSelfType().isMovable(decoratorLoc, shared))
    synthesizeCopyMoveInit(/*isMove=*/true, decoratorLoc, structDecl,
                           isMemoryOnly, structFields, resolver);
}

void StructBodyDecorators::processDecorator(ExprNode *decorator) {
  if (auto declRef = dyn_cast<DeclRefNode>(decorator)) {
    if (declRef->spelling == "value")
      return processValueDecorator(decorator->getRangeStart());

    emitError(decorator->getLoc(), "unsupported decorator: '@")
        << declRef->spelling << "'" << declRef->getRange();
    return;
  }

  emitError(decorator->getLoc(), "unsupported decorator")
      << decorator->getRange();
}

/// Process the @register_passable decorator on structs.  This finalizes
/// semantic checks.
static void processRegisterPassableDecorator(
    StructDeclOp structOp, ASTDecl &structDecl,
    ArrayRef<std::pair<StructFieldOp, ASTDecl *>> structFields,
    DeclResolver &resolver, StructDeclOp::RegisterPassable structPassability) {

  bool isTrivial =
      structPassability == StructDeclOp::RP_RegisterPassableTrivial;
  for (auto [fieldOp, fieldDecl] : structFields) {
    ASTType fieldType = fieldOp.getType();

    // Register-passable structs may only contain register-passable stored
    // values.
    // TODO(traits): We need to type constrain mlirtype parameters to being
    // register-only types to support things like this correctly:
    //  struct P[T: mlirtype]:
    //    var storage : T

    // If the field is at least as register-passable as the container then
    // we're happy.
    if (fieldType.getRegisterPassability(fieldDecl->getLoc(), resolver.shared) <
        structPassability) {
      StringRef trivialSuffix;
      if (isTrivial)
        trivialSuffix = "(\"trivial\")";

      auto diag = resolver.emitError(structOp.getLoc())
                  << "all members of '@register_passable" << trivialSuffix
                  << "' struct must themselves be '@register_passable"
                  << trivialSuffix << "'";
      diag.attachNote(fieldDecl->getLoc())
          << fieldOp.getNameAttr() << " declared with type " << fieldType;

      // We cannot support IRGen'ing references to this type, since it will
      // break invariant about being register passable without being composed
      // of such types.
      fieldDecl->getParentDecl()->hasReferenceError = true;
      return;
    }
  }

  // Trivial types may not have __copyinit__ or __del__ members.
  if (isTrivial) {
    auto rejectMemberIfPresent = [&](StringRef name) {
      auto members = structDecl.lookupInCurrentScope(name);
      if (!members.empty())
        resolver.emitError(members[0]->getLoc())
            << "'@register_passable(\"trivial\")' types may not have a '"
            << name << "' method";
    };

    rejectMemberIfPresent("__copyinit__");
    rejectMemberIfPresent("__del__");
  }
}

ParseResult DeclResolver::resolveBody(StructDeclOp structOp, Lexer &lexer,
                                      ASTDecl &structDecl) {
  if (ParserBase::parseSuite(structDecl, lexer))
    return failure();

  // Mark the declaration as fully resolved so we can lookup into it.
  structDecl.resolvedness = DeclResolvedness::fully;

  // Track whether any field needs destruction, if so, we need a __del__
  // method.
  bool needsDtorForFields = false;

  /// This collects all the resolved struct fields.
  SmallVector<std::pair<StructFieldOp, ASTDecl *>> structFields;

  // Now that the body is completely resolved, check the declared fields for
  // extra invariants.
  for (StructFieldOp field : structOp.getFieldDecls()) {
    // Make sure the field is signature resolved so we can get its type.
    auto fieldEntries = structDecl.lookupInCurrentScope(field.getNameAttr());
    assert(fieldEntries.size() == 1 && "field decls cannot be overloaded");
    ASTDecl &fieldASTDecl = *fieldEntries[0];
    if (failed(resolveSignature(fieldASTDecl, fieldASTDecl.getLoc())))
      continue;

    // If any field of this struct has a destructor, then the struct needs
    // one.
    needsDtorForFields |=
        ASTType(field.getType()).hasDestructor(fieldASTDecl.getLoc(), shared);

    structFields.push_back({field, &fieldASTDecl});
  }

  // If the struct is @register_passable, check invariants imposed by it before
  // checking other decorators.  This ensures that we reject invalid
  // register_passable types before processing them.
  if (auto passability = structOp.getRegisterPassable()) {
    // TODO: Split trivial and register_passable appart.
    processRegisterPassableDecorator(
        structOp, structDecl, structFields, *this,
        (StructDeclOp::RegisterPassable)passability);
  }

  // If there are any body decorators, resolve them now.
  for (auto decoratorCursor : structDecl.getBodyDecorators(shared)) {
    StructBodyDecorators(structOp, structDecl, *this, structFields)
        .processBodyDecorator(decoratorCursor);
  }

  if (structDecl.hasReferenceError)
    return success();

  // Now that the struct body has been resolved, check to see if there is a
  // destructor and install it into the StructDeclOp if so.
  if (auto dtorAttr = lookupDestructor(structDecl, shared)) {
    // Check to see if we have an explicitly declared destructor.
    structOp.setDestructorAttr(dtorAttr);
  } else if (needsDtorForFields) {
    // If one of the fields needs to be destroyed, then we synthesize an empty
    // del function so that lifetime checking can handle field destruction.
    structOp.setDestructorAttr(
        synthesizeEmptyDtor(structOp, structDecl, *this));
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
                                             Lexer &lexer, ASTDecl &decl) {
  ParserBase p(lexer);
  auto decoratorExprs = p.parseDecorators(decl);

  ASTType type;
  // Parse the type if present.
  p.consumeToken(); // let or var.
  if (p.parseToken(Token::identifier,
                   "internal error: checked by stmt parser") ||
      p.parseToken(Token::colon, "struct field declaration must have a type") ||
      parseType(p, type, *decl.getParentDecl(), decl.getIndentation()))
    return failure();

  fieldOp.setType(type);
  rejectDecorators(decoratorExprs, decl, shared);
  return success();
}

ParseResult DeclResolver::resolveBody(StructFieldOp op, Lexer &lexer,
                                      ASTDecl &decl) {
  return success();
}

//===----------------------------------------------------------------------===//
// UnresolvedImport Decl implementation
//===----------------------------------------------------------------------===//

ParseResult DeclResolver::resolveSignature(LIT::UnresolvedImportOp op,
                                           Lexer &lexer, ASTDecl &decl) {
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
