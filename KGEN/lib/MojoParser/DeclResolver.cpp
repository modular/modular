//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Declaration parsing and name binding logic.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser/DeclResolver.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/CallEmission.h"
#include "KGEN/MojoParser/ClosureEmitter.h"
#include "KGEN/MojoParser/DocString.h"
#include "KGEN/MojoParser/ExprEmitter.h"
#include "KGEN/MojoParser/ExprNodes.h"
#include "KGEN/MojoParser/IRValues.h"
#include "KGEN/MojoParser/Lexer.h"
#include "KGEN/MojoParser/ParserBase.h"
#include "KGEN/MojoParser/ParserParamEvaluator.h"
#include "KGEN/MojoParser/SharedState.h"
#include "KGEN/MojoParser/StructEmitter.h"
#include "Utils.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/LITDialect/SpecialFunctions.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "Support/Profiling/TimeProfiler.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SourceMgr.h"
#include <deque>

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

  ExprEmitter emitter(p.shared, declScope, EC_Type);
  result = emitter.emitExprType(expr);
  if (!result)
    return failure();

  return success();
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

ASTDecl &DeclResolver::createUnlistedDecl(DeclIRValue irValue, SMLoc loc,
                                          ASTDecl *parentDecl,
                                          LexerCursor cursor,
                                          LexerCursor endCursor,
                                          ssize_t indentation) {
  ASTDecl *decl = shared.allocPersistent<ASTDecl>(
      irValue, loc, parentDecl, cursor, endCursor, indentation);
  parsedDeclList.push_back(decl);

  // If this is a declaration which has a TypeCheckErrorType, then all
  // references to it are invalid.
  if (auto rv = decl->getIfRValue()) {
    if (rv.getType().isTypeCheckErrorType())
      decl->hasReferenceError = true;
  } else if (auto lv = decl->getIfMLValue()) {
    if (lv.getRValueType().isTypeCheckErrorType())
      decl->hasReferenceError = true;
  } else if (auto bv = decl->getIfBValue()) {
    if (bv.getRValueType().isTypeCheckErrorType())
      decl->hasReferenceError = true;
  }

  return *decl;
}
ASTDecl &DeclResolver::createUnlistedDecl(Operation *declOp, SMLoc loc,
                                          ASTDecl *parentDecl,
                                          LexerCursor cursor,
                                          LexerCursor endCursor,
                                          ssize_t indentation) {
  return createUnlistedDecl(DeclIRValue(declOp), loc, parentDecl, cursor,
                            endCursor, indentation);
}

void DeclResolver::attachDeclToParentNameTable(ASTDecl *decl, StringAttr name) {
  ASTDecl *parentDecl = decl->getParentDecl();
  // Remember the named decl in the symbol table so it can be looked up.
  TinyPtrVector<ASTDecl *> &entries = parentDecl->declsInScope[name];
  if (entries.empty()) {
    entries.push_back(decl);

    // If the decl is a type or alias that has a symbol, remember it.  This
    // allows us to look up decls by symbol when referenced as types. Functions
    // don't have symbols until they are fully resolved, but decls inside
    // functions cannot be accessed anyways.
    if (auto symbolDecl = dyn_cast<mlir::SymbolOpInterface>(decl);
        symbolDecl && !isa<LIT::FuncOp>(*decl)) {
      // Make sure there are no name conflicts with the MLIR symbol.  If there
      // are, then addDecl will have rejected it with an error.
      shared.setResolvedDeclSymbol(symbolDecl);

      SymbolRefAttr symbol = decl->getSymbolRef();
      assert(!declForTypeSymbol.count(symbol) &&
             "Symbol redefinition/collision");
      declForTypeSymbol[symbol] = decl;
    }

    return;
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
        return;
      }
    }

    // Otherwise, we're good, charge forwards.
    entries.push_back(decl);
    return;
  }

  // Check if we are adding an identical unresolved import.
  if (auto import = dyn_cast<UnresolvedImportOp>(decl)) {
    auto prevOp = dyn_cast<UnresolvedImportOp>(entries.front());
    if (prevOp && import.getModuleNameAttr() == prevOp.getModuleNameAttr() &&
        import.getDeclNameAttr() == prevOp.getDeclNameAttr()) {
      entries.push_back(decl);
      return;
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
}

/// Add a new declaration that needs to be resolved.
ASTDecl &DeclResolver::addDecl(DeclIRValue irValue, SMLoc loc,
                               StringAttr baseName, ASTDecl *parentDecl,
                               LexerCursor cursor, LexerCursor endCursor,
                               ssize_t indentation) {
  ASTDecl &decl = createUnlistedDecl(irValue, loc, parentDecl, cursor,
                                     endCursor, indentation);
  // If this has a parent and a name, insert it into the parents name table so
  // name lookup will resolve it.  If it doesn't, then we're done.
  if (baseName)
    attachDeclToParentNameTable(&decl, baseName);
  return decl;
}

void DeclResolver::moveDecls(ASTDecl &dst, ASTDecl &src) {
  dst.hasReferenceError |= src.hasReferenceError;
  for (auto &[name, children] : src.declsInScope)
    for (ASTDecl *child : children)
      child->parentDecl = &dst;
  dst.declsInScope = std::move(src.declsInScope);
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
  if (auto importOp = dyn_cast<UnresolvedImportOp>(frontDecl)) {
    // If the import is overlapping with an existing declaration, let it slide.
    // FIXME: This is assuming that the import would resolve to the same decl.
    if (ArrayRef<ASTDecl *> decls = context.lookupInCurrentScope(name);
        !decls.empty())
      return success();

    ASTDecl &importDecl = addDecl(
        frontDecl->getIfOperation(), frontDecl->getLoc(), name, &context,
        frontDecl->getCursor(), frontDecl->getCursor(), /*indentation=*/-1);
    return success(!importDecl.hasReferenceError);
  }

  auto [it, inserted] = context.declsInScope.insert({name, decls});
  if (inserted)
    return success();
  TinyPtrVector<ASTDecl *> &entries = it->second;

  // We hit an overlap, check to see if this is just resolving a module import.
  // If so, replace the unresolved import with the real decls.
  if (moduleName) {
    auto importOp = dyn_cast<UnresolvedImportOp>(it->second.back());
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

        LITSignatureType declSignature = declOp.getFullSignature();
        LITSignatureType existingSignature = existingOp.getFullSignature();
        // If the value input types match exactly *and* the input parameter
        // types match exactly, then we don't want to merge this decl into the
        // set. We also need to remove the by-ref result type from the
        // input types, so that aliasing is strictly based on the actual
        // inputs.
        auto getActualValueInputs =
            [](SignatureType signature) -> ArrayRef<mlir::Type> {
          ArrayRef<Type> inputTypes = signature.getValueInputs();
          ArrayRef<ValueInputConvention> inputConventions =
              signature.getInputConventions();
          // If there's a by-ref result type, it'll be the first argument.
          if (!inputConventions.empty() &&
              inputConventions.front() == ValueInputConvention::ByRefResult) {
            inputTypes = inputTypes.drop_front();
          }
          return inputTypes;
        };

        if (getActualValueInputs(declSignature) ==
                getActualValueInputs(existingSignature) &&
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

LogicalResult DeclResolver::importModule(ASTDecl &dest,
                                         PackageOp currentPackage,
                                         StringAttr moduleName,
                                         StringAttr importName, SMLoc loc,
                                         SMLoc importNameLoc) {
  ASTDecl &module = shared.importModule(moduleName, currentPackage, loc);
  shared.notifyListenerOnModuleImport(module, moduleName, loc);
  shared.notifyListenerOnRef(&module, importName, importNameLoc);

  return aliasImportDecls(TinyPtrVector<ASTDecl *>(&module), importName,
                          /*declName=*/StringAttr(), moduleName, importNameLoc,
                          dest);
}

LogicalResult
DeclResolver::importDeclFromModule(ASTDecl &dest, PackageOp currentPackage,
                                   StringAttr moduleName, StringAttr sourceName,
                                   StringAttr destName, SMLoc loc,
                                   SMLoc sourceNameLoc, SMLoc destNameLoc) {
  ASTDecl &module = shared.importModule(moduleName, currentPackage, loc);
  shared.notifyListenerOnModuleImport(module, moduleName, loc);

  FailureOr<ArrayRef<ASTDecl *>> results =
      lookupDeclInModule(module, sourceName, sourceNameLoc);
  if (failed(results))
    return failure();
  shared.notifyListenerOnRef(*results, sourceName, sourceNameLoc);
  shared.notifyListenerOnRef(*results, destName, destNameLoc);

  return aliasImportDecls(TinyPtrVector<ASTDecl *>(*results), destName,
                          sourceName, moduleName, destNameLoc, dest);
}

LogicalResult DeclResolver::importWildCardDeclsFromModule(ASTDecl &context,
                                                          StringAttr moduleName,
                                                          bool isFullImport,
                                                          llvm::SMLoc loc) {
  PackageOp currentPackage = dyn_cast<PackageOp>(context);
  if (!currentPackage && context.getIfOperation())
    currentPackage = context.getIfOperation()->getParentOfType<PackageOp>();

  // Make sure the module has been resolved.
  ASTDecl &module = shared.importModule(moduleName, currentPackage, loc);
  if (failed(resolveFully(module, loc)))
    return failure();

  // Resolve pending wildcard imports in this module.
  if (failed(resolveAllWildcardImports(module)))
    return failure();

  // Wildcard imports don't import decls with a leading '_'.
  LogicalResult result = success();
  for (const auto &[name, decls] : module.declsInScope) {
    if (!isFullImport && name.getValue()[0] == '_')
      continue;
    if (failed(aliasImportDecls(decls, name, name, moduleName, loc, context)))
      result = failure();
  }
  return result;
}

FailureOr<ArrayRef<ASTDecl *>>
DeclResolver::lookupDeclInModule(ASTDecl &module, StringAttr sourceName,
                                 SMLoc loc) {
  if (failed(resolveFully(module, loc)))
    return failure();

  // Check to see if the module has the construct we are importing.
  auto result = shared.lookupAndResolveDecl(sourceName, loc, module,
                                            /*searchParentScopes=*/false);
  if (result.isErroneous())
    return failure();
  if (result.isFailure()) {
    // Emit an error with the module name without the leading `$` mangle.
    StringRef name = cast<mlir::SymbolOpInterface>(module).getName();
    StringRef declType = isa<PackageOp>(module) ? "package" : "module";
    assert(name.startswith("$") && "unexpected module/package name mangling");
    emitError(loc, declType + " '" + name.drop_front() +
                       "' does not contain '" + sourceName.getValue() + "'");
    return failure();
  }
  return result.getIfSuccess();
}

/// Add a new declaration that needs to be resolved.
ASTDecl &DeclResolver::addDecl(Operation *op, SMLoc loc, StringAttr baseName,
                               ASTDecl *parentDecl, LexerCursor cursor,
                               LexerCursor endCursor, ssize_t indentation) {
  return addDecl(DeclIRValue(op), loc, baseName, parentDecl, cursor, endCursor,
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

void DeclResolver::resolveAllReferencedFrom(ASTDecl &decl) {
  TimeTraceScope traceScope("resolveAllReferencedFrom", [&] {
    return decl.getNameIfOperation().value_or("").str();
  });

  // The first stage is to fully resolve all of the decls recursively defined
  // within the main container. These decls provide the anchor for resolution.
  std::deque<ASTDecl *> worklist({&decl});
  while (!worklist.empty()) {
    ASTDecl *declIt = worklist.back();
    worklist.pop_back();

    // Resolve the decl.
    (void)resolveFully(*declIt, declIt->getLoc());

    // When validating doc strings, we wish to only validate those defined on
    // decl in the main container. As this point the main container decl has
    // been fully resolved, so it's an opportune time to validate.
    validateDocString(shared, *declIt);

    // If this is a package, resolve all of the modules within it as a pre-step.
    // Normally these get lazily resolved, but if we're forcing pulling them in,
    // we need to do it now.
    if (isa<PackageOp>(*declIt)) {
      for (auto &decls : llvm::make_second_range(declIt->declsInScope))
        if (isa<UnresolvedImportOp>(*decls.front()))
          (void)resolveFully(*decls.front(), declIt->getLoc());
    }

    // Traverse the children. We don't resolve alias children, these will be
    // resolved separately if they actually got referenced.
    for (auto &decls : llvm::make_second_range(declIt->declsInScope)) {
      for (ASTDecl *decl : decls)
        if (decl->getParentDecl() == declIt)
          worklist.push_front(decl);
    }
  }

  // After all of the children within `decl` have been fully resolved, we can
  // now iteratively resolve all of the outside decls that got referenced.
  llvm::SetVector<ASTDecl *> deferredDecls;
  size_t parsedDeclIt = 0;
  do {
    // Resolve all of the newly parsed decls that got referenced.
    for (; parsedDeclIt != parsedDeclList.size(); ++parsedDeclIt) {
      ASTDecl &decl = *parsedDeclList[parsedDeclIt];

      // If the decl was never touched and we pulled it in from bytecode, treat
      // it as unreachable and don't resolve it now.
      if (decl.loadedFromBytecode &&
          decl.resolvedness == DeclResolvedness::unparsed) {
        // Some decls always need to be resolved if their parents were resolved,
        // allowlist the decls that we can safely ignore when unparsed.
        if (isa<FuncOp, FileModuleOp, PackageOp, UnresolvedImportOp,
                UnresolvedWildcardImportOp, StructDeclOp, TraitDeclOp,
                AliasDeclOp, GlobalVarDeclOp>(decl)) {
          deferredDecls.insert(&decl);
          continue;
        }
      }

      (void)resolveFully(decl, decl.getLoc());
    }

    // After resolving the newly parsed decls, make sure we resolve any
    // previously parsed decls that are newly referenced.
    bool resolvedAnything = false;
    do {
      resolvedAnything = false;
      for (ASTDecl *decl : deferredDecls) {
        // Fully resolve this decl if it was only midway resolved during normal
        // parsing resolution.
        if (decl->resolvedness == DeclResolvedness::signature) {
          (void)resolveFully(*decl, decl->getLoc());
          resolvedAnything = true;
        }
      }
    } while (resolvedAnything);
  } while (parsedDeclIt != parsedDeclList.size());
}

void DeclResolver::registerAndCheckExport(StringRef aliasName, SMLoc loc) {
  auto [it, inserted] = exportedSymbolNames.try_emplace(aliasName, loc);
  if (!inserted) {
    auto diag = emitError(loc, "invalid re-export of ") << aliasName;
    diag.attachNote(it->second) << "previous export here";
    return;
  }
}

void DeclResolver::exportMain(ASTDecl &funcDecl) {
  LIT::FuncOp userMainFn = cast<LIT::FuncOp>(funcDecl);
  SignatureType userMainSignature = userMainFn.getSignature();
  ASTDecl *containingDecl = funcDecl.getParentDecl();
  Location loc = userMainFn.getLoc();

  // The type of main function described by the given func decl.
  enum MainKind {
    // A non-raising function that returns None.
    kNonRaisingNoneMain,
    // A raising function that returns None.
    kRaisingNoneMain,
    // A raising function that returns object.
    kRaisingObjectMain,
  };
  MainKind mainKind = kNonRaisingNoneMain;

  // Validate that main has the expected signature.
  if (!userMainSignature.getInputParamTypes().empty() ||
      !userMainSignature.getResultParamTypes().empty()) {
    shared.emitError(loc, "expected 'main' function to have no parameters");
    return;
  }
  ASTType userResultType(userMainFn.getUserResultType());
  ArrayRef<Type> valueInputs = userMainSignature.getValueInputs();

  // Process a main returning none.
  if (userResultType.isNoneType()) {
    if (userMainSignature.isThrows())
      mainKind = kRaisingNoneMain;

    // Process a main returning object.
  } else if (userResultType.isEqualCanon(
                 shared.lookupObjectType(funcDecl.getLoc(), *containingDecl))) {
    // Check that the function is raising, e.g. the `def main()` mode.
    if (!userMainSignature.isThrows()) {
      shared.emitError(
          loc, "expected 'main' function returning object to be raising");
      return;
    }
    mainKind = kRaisingObjectMain;

    // Drop the result type from the value inputs.
    valueInputs = valueInputs.drop_front();

    // Otherwise, this is an unrecognized main.
  } else {
    shared.emitError(loc, "expected 'main' function to return 'None'");
    return;
  }
  if (!valueInputs.empty()) {
    shared.emitError(loc, "expected 'main' function to have no arguments");
    return;
  }

  // Utility for resolving a decl within the Startup module.
  ASTDecl &startupModule = shared.importModule(
      "builtin._startup", /*currentPackage=*/nullptr, funcDecl.getLoc());
  auto resolveStartDecl = [&](StringRef name) -> ASTDecl * {
    auto result = shared.lookupAndResolveDecl(
        name, funcDecl.getLoc(), startupModule, /*searchParentScopes=*/false);
    if (result.getIfSuccess().empty()) {
      if (result.isFailure()) {
        shared.emitError(funcDecl.getLoc(),
                         "unable to resolve `Builtin.Startup` module when "
                         "exporting 'main'");
      }
      return nullptr;
    }
    ASTDecl *decl = result.getIfSuccess().front();
    if (failed(resolveFully(*decl, decl->getLoc())))
      return nullptr;
    return result.getIfSuccess().front();
  };

  // Generate a shim for main that handles parsing command line arguments,
  // capturing uncaught exceptions, and returning the exit code. The shim
  // defines a C-ABI compatible function that sets up the mojo runtime.
  OpBuilder builder = containingDecl->getDeclEndBuilder();

  // The Startup module provides a stubbed out shim for us to use, so pull that
  // in.
  ASTDecl *mainShimProtoDecl = resolveStartDecl("__mojo_main_prototype");
  if (!mainShimProtoDecl)
    return;
  FuncOp mainShimProtoFn = cast<FuncOp>(*mainShimProtoDecl);

  // Builder function.
  StringAttr mainAttr = StringAttr::get(getContext(), kMainSymbolName);
  auto shimMainFn = cast<FuncOp>(builder.clone(*mainShimProtoFn));
  shimMainFn.setSymNameAttr(mainAttr);
  shimMainFn.setLinkageNameAttr(mainAttr);
  shimMainFn.setCExported();
  shimMainFn.getBody()->clear();

  // The shim may be parsed from the precompiled standard library package, make
  // sure to drop any of the package metadata.
  shimMainFn.setPreCompiledModuleRef(std::nullopt);

  // Populate the body of the shim. For this we designate the internal
  // implementation to one of the wrapper helpers in the Startup module,
  // depending on how the user specified their main function.
  StringRef mainWrapperName;
  switch (mainKind) {
  case kNonRaisingNoneMain:
    mainWrapperName = "__wrap_and_execute_main";
    break;
  case kRaisingNoneMain:
    mainWrapperName = "__wrap_and_execute_raising_main";
    break;
  case kRaisingObjectMain:
    mainWrapperName = "__wrap_and_execute_object_raising_main";
    break;
  }
  ASTDecl *mainWrapperDecl = resolveStartDecl(mainWrapperName);
  if (!mainWrapperDecl)
    return;
  FuncOp mainWrapperFn = cast<FuncOp>(*mainWrapperDecl);

  // Generate a reference to the main wrapper function, which expects the user
  // main to be provided via an input parameter.
  SymbolConstantAttr wrapperFnRef = SymbolConstantAttr::get(
      getFullyResolvedSymbolRef(mainWrapperFn),
      {SymbolConstantAttr::get(getFullyResolvedSymbolRef(userMainFn),
                               userMainSignature)},
      mainWrapperFn.getSignature().dropParamValues());

  auto shimBodyBuilder = ImplicitLocOpBuilder::atBlockBegin(
      shimMainFn->getLoc(), shimMainFn.getBody());
  auto wrappedCall = shimBodyBuilder.create<CallOp>(
      shimMainFn.getArgumentTypes()[0], wrapperFnRef,
      /*lifetimeParams=*/std::nullopt, /*paramDecls=*/ArrayRef<ParamDeclAttr>(),
      shimMainFn.getArguments());
  shimBodyBuilder.create<LIT::ReturnOp>(wrappedCall.getResults());
  shimBodyBuilder.create<EndFuncOp>();

  exportedSymbolNames.insert({mainAttr, funcDecl.getLoc()});
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

  // Handle decls that are loaded from bytecode. These decls are not parsed like
  // decls originating from source files.
  if (decl.loadedFromBytecode) {
    if (failed(shared.resolveDeclFromBytecode(decl, howResolved)))
      decl.hasReferenceError = true;

    declsCurrentlyProcessing.erase(&decl);
    return success(!decl.hasReferenceError);
  }

  // If the signature hasn't been parsed, do so.
  if (decl.resolvedness < DeclResolvedness::signature) {
    // Handle each operation that can be name bound.  We handle this by
    // restoring the lexer to the position where parsing can continue, calling
    // the `resolveSignature` method for the op, and re-saving the new cursor
    // for the next stage of resolution.
    TypeSwitch<ASTDecl &>(decl)
        .Case<LIT::FuncOp, StructDeclOp, StructFieldOp, TraitDeclOp,
              GlobalVarDeclOp, AliasDeclOp>([&](auto op) {
          Lexer lexer(shared.diags, decl.getCursor());

          // Generate pretty stack traces if a crash happens in this scope.
          LexerCrashReporter crashReporter(lexer, decl.getLoc(),
                                           "resolving decl signature");

          // Resolve the signature: on a parse error, we note that the decl
          // is malformed and should not be referenced to silence downstream
          // errors.
          if (failed(resolveSignature(op, lexer, decl)))
            decl.hasReferenceError = true;
          decl.getCursor() = lexer.getCursor();
        })
        .Case<UnresolvedImportOp>([&](auto op) {
          // Resolve the signature: on a parse error, we note that the decl
          // is malformed and should not be referenced to silence downstream
          // errors.
          if (failed(resolveSignature(op, decl)))
            decl.hasReferenceError = true;
        })
        .Case<LIT::FileModuleOp, ModuleOp, PackageOp,
              UnresolvedWildcardImportOp>([&](auto op) { /*Nothing*/ })
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
                                   Token::kw_trait, Token::kw_class,
                                   Token::kw_var)) {
          lexer.emitTokenError(
              "definition isn't on its own line at the correct "
              "indentation");
        } else if (lexer.getToken().is(Token::eof)) {
          lexer.emitTokenError(
                   "internal error: decl parsing skipped beyond end "
                   "of declaration")
                  .attachNote(decl.getLoc())
              << "declaration started here";
        } else {
          lexer.emitTokenError("unknown tokens at the end of a declaration");
        }
      }
    };

    // Mark the body as already resolved so that name lookup can be performed
    // in the decl during resolution.
    decl.resolvedness = DeclResolvedness::fully;

    // Handle each operation that can be name bound.
    TypeSwitch<ASTDecl &>(decl)
        .Case<FileModuleOp, LIT::FuncOp, StructDeclOp, StructFieldOp,
              TraitDeclOp, GlobalVarDeclOp, LetRegDeclOp, AliasDeclOp,
              AliasForwardDeclOp>([&](auto op) {
          // Parse the body of the declaration from the correct point.
          Lexer lexer(shared.diags, decl.getCursor());

          // Generate pretty stack traces if a crash happens in this scope.
          LexerCrashReporter crashReporter(lexer, decl.getLoc(),
                                           "resolving decl body");

          if (resolveBody(op, lexer, decl))
            return;

          checkEndOfBodyCursor(lexer);
        })
        .Case([&](PackageOp op) { (void)resolveBody(op, decl); })
        .Case<ModuleOp, UnresolvedImportOp, UnresolvedWildcardImportOp>(
            [&](auto op) { /*Nothing*/ })
        .Default([&](auto &attr) {
          if (!decl.hasReferenceError)
            emitError(decl.getLoc(),
                      "do not know how to resolve the body of this decl!");
        });
  }

  declsCurrentlyProcessing.erase(&decl);
  // If decl is busted, then return failure.
  return success(!decl.hasReferenceError);
}

LogicalResult DeclResolver::resolveAllWildcardImports(ASTDecl &module) {
  while (!module.unresolvedWildcardImports.empty()) {
    auto it = module.unresolvedWildcardImports.begin();
    auto [moduleName, locAndIsFullImport] = *it;
    module.unresolvedWildcardImports.erase(it);

    if (failed(importWildCardDeclsFromModule(module, moduleName,
                                             locAndIsFullImport.second,
                                             locAndIsFullImport.first)))
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ParserParamEvaluator implementation
//===----------------------------------------------------------------------===//

ParserParamEvaluator::ParserParamEvaluator(DeclResolver &resolver,
                                           ArrayRef<ParamDeclAttr> paramDecls,
                                           ArrayRef<TypedAttr> paramValues)
    : ParameterEvaluator(paramDecls, paramValues),
      InterpreterState(resolver.getContext()), resolver(resolver) {}

FailureOr<TypedAttr>
ParserParamEvaluator::evaluateFunctionCall(SymbolRefAttr symbol,
                                           ArrayRef<Attribute> arguments) {
  ErrorOr<Region *> body = lookupFunctionBody(symbol);
  if (body.isError()) {
    // Swallow the error.
    DEBUG_WITH_TYPE("lit-parameter-evaluator", llvm::errs()
                                                   << "[ParserParamEvaluator] "
                                                   << body.getError() << "\n");
    return failure();
  }

  ErrorTreeOr<SmallVector<Attribute>> result =
      executeRegion(*body.takeValue(), arguments);
  if (result.isError()) {
    // Swallow the error.
    DEBUG_WITH_TYPE("lit-parameter-evaluator",
                    result.takeError().emit(
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
  if (func.getInlineLevel() == InlineLevel::Automatic)
    return Error("function is not always_inline");
  LITSignatureType fullSig = func.getFullSignature();
  if (!fullSig.getInputParamTypes().empty() ||
      !fullSig.getResultParamTypes().empty())
    return Error("function is parametric");

  // Use of the interpreter's memory model requires a target specification,
  // which the parser does not have.
  if (fullSig.hasMemoryOnlyResult() || fullSig.hasInitSelfResult())
    return Error("function has memory-only result");

  // Make sure to fully resolve the body and everything within it.
  if (failed(resolver.resolveFully(*decl, decl->getLoc())))
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
                                  ArgListKind kind) {
  loc = p.getToken().getLoc();
  cursor = p.getLexer().getCursor();

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
  if (kind == ArgListKind::kFnTypeArgList ||
      kind == ArgListKind::kFnTypeParamList) {
    StringAttr maybeArgName;
    SMLoc nextLocation;
    if (succeeded(p.parseOptionalIdentifier(maybeArgName, Token::colon,
                                            &nextLocation))) {
      name = maybeArgName;
      loc = nextLocation;
    }
  } else {
    if (p.parseIdentifier(name, "expected parameter name", &loc)) {
      // TODO: Scan ahead for better recovery.
      return failure();
    }
  }

  // Parse an optional type annotation: `":" ["*"] expression`. Omit the colon
  // if a name was not specified.  Bare lambda arg lists do not allow types.
  if (kind != ArgListKind::kBareLambdaArgList) {
    if (!name || p.consumeIf(Token::colon)) {
      SMLoc starLoc = p.getToken().getLoc();
      if (p.getToken().getKind() == Token::star) {
        if (vararg != VarArgKind::VarArg) {
          InflightDiag diag = p.emitError(
              starLoc, "only variadic arguments' types can be unpacked");
          if (name) {
            diag.attachNote(loc)
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
  }

  // Set the name to empty string if it wasn't specified.
  if (!name)
    name = StringAttr::get(p.getContext());

  // Parse an optional default argument value: `"=" expression`.
  SMLoc equalLoc;
  if (p.consumeIf(Token::equal, &equalLoc)) {
    if (p.parseExpression(initExpr))
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
    ParserBase &p, SmallVectorImpl<ParsedArgument> &args, ArgListKind kind) {
  // Figure out where to stop scanning.
  SmallVector<Token::Kind, 2> stopTokens;
  switch (kind) {
  case ArgListKind::kParamList:
  case ArgListKind::kFnTypeParamList:
    stopTokens.append({Token::r_square, Token::minus_greater});
    break;
  case ArgListKind::kFnTypeArgList:
  case ArgListKind::kArgList:
    stopTokens.push_back(Token::r_paren);
    break;
  case ArgListKind::kBareLambdaArgList:
    stopTokens.push_back(Token::colon);
    break;
  }

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
  StringRef argOrParam =
      kind == ArgListKind::kParamList || kind == ArgListKind::kFnTypeParamList
          ? "parameter"
          : "argument";
  auto handleSlashMarker = [&](SMLoc loc) {
    if (hasSlashMarker) {
      p.emitError(loc, "cannot have two '/' markers in the same ")
          << argOrParam << " list";
      return;
    }
    if (hasStarMarker) {
      p.emitError(loc, "cannot specify '/' marker after '*' marker in ")
          << argOrParam << " list";
      return;
    }
    if (args.empty()) {
      p.emitError(loc, "'/' marker cannot be used at the start of the ")
          << argOrParam << " list";
    }

    // Ok, process it by changing all arguments we've seen to be positional
    // only.  The remaining ones will stay kPositionalOrKeyword though.
    for (ParsedArgument &arg : args)
      arg.kwArgHandling = KWArgHandling::kPositionalOnly;
    hasSlashMarker = true;
  };

  // This is invoked when we see a '*' marker or '*arg' argument.
  auto handleStarMarker = [&](SMLoc loc, bool isMarker) {
    if (hasStarMarker) {
      p.emitError(loc, "cannot have two '*' markers in the same ")
          << argOrParam << " list";
    }

    // Diagnose '*' marker at end of argument list for completeness.
    if (p.getToken().isAny(stopTokens) && isMarker) {
      p.emitError(loc, "'*' marker is not allowed at end of ")
          << argOrParam << " list";
    }

    // From now on, any parsed arguments are keyword only.
    defaultKWArgHandling = KWArgHandling::kKeywordOnly;
    hasStarMarker = true;
  };

  // This parses either an argument or a keyword argument specifier.
  bool foundName = false;
  auto parseArgument = [&]() -> ParseResult {
    KWArgMarkerInfo marker = KWArgMarkerInfo::kNotMarker;
    ParsedArgument arg;
    arg.kwArgHandling = defaultKWArgHandling;
    if (arg.parse(p, marker, kind))
      return failure();

    // If this argument is just a marker, process it.
    if (marker == KWArgMarkerInfo::kSlash)
      return handleSlashMarker(arg.loc), success();
    if (marker == KWArgMarkerInfo::kStar)
      return handleStarMarker(arg.loc, /*isMarker=*/true), success();

    if (arg.name.empty()) {
      if (foundName) {
        return p.emitError(arg.loc, "unnamed ")
               << argOrParam << " cannot follow named " << argOrParam;
      }
      if (hasSlashMarker || hasStarMarker) {
        return p.emitError(arg.loc, "unnamed ")
               << argOrParam << " cannot follow '/' or '*'";
      }
    } else {
      foundName = true;
    }

    // Otherwise, if this is a varargs marker, handle it as a marker and an
    // argument.
    if (arg.vararg == VarArgKind::VarArg ||
        arg.vararg == VarArgKind::PackVarArg)
      handleStarMarker(arg.loc, /*isMarker=*/false);

    // If we have a **arg then it must be the last argument.
    if (arg.vararg == VarArgKind::KWVarArg && p.getToken().isNot(stopTokens)) {
      p.emitError(arg.loc, "'**' marker must be at end of ")
          << argOrParam << " list";
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

  // We allow specifying signatures with only positional-only arguments if all
  // the argument names are omitted, i.e. `fn(Int, Int) -> Int` is the same as
  // `fn(Int, Int, /) -> Int`.
  bool allUnnamedPosOnly = !foundName && !hasSlashMarker && !hasStarMarker;
  for (ParsedArgument &arg : args) {
    if (!arg.name.empty() ||
        arg.kwArgHandling == KWArgHandling::kPositionalOnly || arg.vararg)
      continue;
    if (!allUnnamedPosOnly)
      return p.emitError(arg.loc, "unnamed ")
             << argOrParam << " must be positional-only";
    arg.kwArgHandling = KWArgHandling::kPositionalOnly;
  }

  // TODO(Keyword Args): now that we parsed a fully generic parameter list,
  // reject keyword-only arguments. Remove them from the signature since the
  // representation does not support them either.
  auto trailingKwarg = [&] {
    return !args.empty() &&
           args.back().kwArgHandling == KWArgHandling::kKeywordOnly;
  };
  if (trailingKwarg()) {
    p.emitError(args.back().loc, "keyword-only ")
        << argOrParam << "s not supported yet";
    do {
      args.pop_back();
    } while (trailingKwarg());
  }
  return success();
}

/// Parse an argument list, including the parentheses around them.  The
/// argument list is allowed to be empty.  If `fnEffects` is non-null, then this
/// parses 'raises' and other effects.
ParseResult ParsedArgument::parseAndResolveParenthesizedArgumentList(
    ParserBase &p, SmallVectorImpl<ParsedArgument> &args, ArgListKind kind,
    FnEffects &fnEffects) {

  if (p.parseToken(Token::l_paren, "expected '(' for argument list"))
    return failure();

  if (!p.consumeIf(Token::r_paren)) {
    if (parseAndResolvePresentArgumentList(p, args, kind) ||
        p.parseToken(Token::r_paren, "expected ')' in argument list"))
      return failure();
  }

  // If the client supports function effects, parse them as well.
  // Parse other function effects.
  while (p.getToken().isIdentifier()) {
    SMLoc loc = p.getToken().getLoc();
    StringRef spelling = p.getToken().getSpelling();

    auto handleEffect = [&](auto hasFn, auto setFn) {
      if ((fnEffects.*hasFn)())
        p.emitError(loc, "function effect '")
            << spelling << "' was already specified";
      (fnEffects.*setFn)(true);
    };

    if (spelling == "raises") {
      handleEffect(&FnEffects::isThrows, &FnEffects::setThrows);
    } else if (spelling == "capturing") {
      handleEffect(&FnEffects::isCapturing, &FnEffects::setCapturing);
    } else if (spelling == "escaping") {
      handleEffect(&FnEffects::isEscaping, &FnEffects::setEscaping);
    } else {
      p.emitError(loc, "unknown function effect '")
          << spelling << "', expected 'raises' or 'capturing'";
    }

    p.consumeIdentifier();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Parameter signature implementation
//===----------------------------------------------------------------------===//

/// Core implementation of the parameter argument parsing logic.
static void processParameterArgs(ExprEmitter &emitter, ASTDecl &declScope,
                                 ArrayRef<ParsedArgument> args,
                                 SmallVectorImpl<ParamDeclAttr> &params,
                                 SmallVectorImpl<StringAttr> &names,
                                 SmallVectorImpl<PassingKind> &passingKinds,
                                 SmallVectorImpl<TypedAttr> &defaults,
                                 bool isResultParams, bool &paramVarArg) {
  bool seenInitExpr = false;
  for (const ParsedArgument &arg : args) {
    // Check for things supported in arguments that are not supported in
    // parameters.

    ASTType type;
    if (arg.typeExpr)
      type = emitter.emitExprType(arg.typeExpr);
    else
      emitter.emitError(arg.loc, "parameters must always have a type");
    if (!type)
      type = emitter.shared.getTypeCheckErrorType();

    VarArgKind vararg = arg.vararg;
    if (vararg != VarArgKind::None && isResultParams)
      emitter.emitError(arg.loc, "result parameters may not be variadic");
    if (vararg == VarArgKind::PackVarArg)
      emitter.emitError(arg.loc, "parameters may not be variadic packs");

    if (vararg == VarArgKind::VarArg && !type.isTypeCheckErrorType()) {
      type = VariadicType::get(type);
      paramVarArg = true;
    }

    if (const ExprNode *initExpr = arg.initExpr) {
      seenInitExpr = true;
      Type paramType = type;
      PValue value =
          emitter.emitExprPValue(initExpr, EC_DefaultParam, paramType);
      if (!value)
        return;
      defaults.push_back(value);
      if (isResultParams) {
        emitter.emitError(arg.loc,
                          "unexpected default value for result parameter");
      }
    } else if (seenInitExpr) {
      emitter.emitError(arg.loc,
                        "non-default parameter follows default parameter")
          << arg.typeExpr->getRange();
    }

    // TODO: Parameter decls should support conventions at some point.
    if (arg.convention != ParsedArgument::kConventionUnspec)
      emitter.emitError(arg.loc, "parameters must always be passed by-value");

    // Bind the parsed type expression so references from other parameters
    // can be resolved. The parameter names in ParamDeclAttr are mangled with
    // the location so that parameter names in mojo are unique in the IR.
    auto newDecl = ParamDeclAttr::get(
        emitter.shared.getMangledParameterName(arg.name.getValue(), arg.loc),
        type);
    params.push_back(newDecl);

    // The unmangled names are also collected to aid keyword parameter binding.
    if (!isResultParams) {
      passingKinds.emplace_back(
          ParsedArgument::mapToPassingKind(arg.kwArgHandling));
      names.push_back(arg.name);
    }

    ASTDecl &resolvedDecl = emitter.getDeclResolver().addFullyResolvedDecl(
        PValue(ParamDeclRefAttr::get(newDecl)), arg.name, arg.loc, &declScope);
    emitter.shared.notifyListenerOnParameterDecl(resolvedDecl, arg.loc);
  }
}

void ParsedArgument::processParameterInputArgs(
    ExprEmitter &emitter, ASTDecl &declScope, ArrayRef<ParsedArgument> args,
    SmallVectorImpl<ParamDeclAttr> &params, SmallVectorImpl<StringAttr> &names,
    SmallVectorImpl<PassingKind> &passingKinds,
    SmallVectorImpl<TypedAttr> &defaults, bool &paramVarArg) {
  processParameterArgs(emitter, declScope, args, params, names, passingKinds,
                       defaults, /*isResultParams=*/false, paramVarArg);
}

void ParsedArgument::processParameterResultArgs(
    ExprEmitter &emitter, ASTDecl &declScope, ArrayRef<ParsedArgument> args,
    SmallVectorImpl<ParamDeclAttr> &params, bool &paramVarArg) {
  SmallVector<TypedAttr> defaults;
  SmallVector<StringAttr> names;
  SmallVector<PassingKind> passingKinds;
  processParameterArgs(emitter, declScope, args, params, names, passingKinds,
                       defaults, /*isResultParams=*/true, paramVarArg);
}

/// param_signature    ::= "[" param_list ("->" param_result_types)? "]"
/// param_list   ::= argument_list | "(" ")"
/// param_result_types ::= expression ("," expression)*
static ParseResult
parseOptionalParameterSignature(ParserBase &p, ASTDecl &declScope,
                                SmallVectorImpl<ParamDeclAttr> &inputParams,
                                SmallVectorImpl<ParamDeclAttr> &resultParams,
                                SmallVectorImpl<StringAttr> &names,
                                SmallVectorImpl<PassingKind> &passingKinds,
                                SmallVectorImpl<TypedAttr> &defaults,
                                bool &paramVarArg) {
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
            p, args, ParsedArgument::ArgListKind::kParamList))
      return failure();
  }

  // Resolve each of the parameter declarations.
  ExprEmitter emitter(p.shared, declScope, EC_Type);
  ParsedArgument::processParameterInputArgs(emitter, declScope, args,
                                            inputParams, names, passingKinds,
                                            defaults, paramVarArg);

  // Parse the meta results if present.
  if (p.consumeIf(Token::minus_greater)) {
    args.clear();
    // Parse a result parameter list.
    if (ParsedArgument::parseAndResolvePresentArgumentList(
            p, args, ParsedArgument::ArgListKind::kParamList))
      return failure();
    ParsedArgument::processParameterResultArgs(emitter, declScope, args,
                                               resultParams, paramVarArg);
  }
  return p.parseToken(Token::r_square, "expected ']' for parameter list");
}

/// Given a type that potentially has all of its parameters unbound, implicitly
/// add the parameter declarations to the function input parameters.
static ASTType
addImplicitTypeParams(SharedState &shared, ASTType type,
                      const ParsedArgument &arg,
                      SmallVectorImpl<StringAttr> &inputParamNames,
                      SmallVectorImpl<PassingKind> &inputParamPassingKinds,
                      SmallVectorImpl<ParamDeclAttr> &inputParamDecls) {
  // Check if the type has unbound parameters.
  auto metatype = dyn_cast_or_null<MetaTypeType>(type.getMetaType());
  if (!metatype)
    return type;
  ArrayRef<Type> inputParams = metatype.getSignature().getInputParamTypes();
  if (inputParams.empty())
    return type;

  unsigned nameCounter = 0;
  SmallVector<TypedAttr> paramValues;
  for (Type type : inputParams) {
    auto funcDecl = ParamDeclAttr::get(
        shared.getMangledParameterName(
            arg.name.getValue() + Twine(nameCounter++), arg.loc),
        type);
    inputParamNames.push_back(StringAttr::get(type.getContext()));
    inputParamPassingKinds.push_back(PassingKind::Implicit);
    inputParamDecls.push_back(funcDecl);
    paramValues.push_back(ParamDeclRefAttr::get(funcDecl));
  }
  return BindTypeAttr::get(PValue(type), paramValues);
}

ASTType ParsedArgument::emitFunctionArgumentsAndResults(
    function_ref<ParseResult()> reportError, ExprEmitter &typeEmitter,
    SmallVectorImpl<StringAttr> &inputParamNames,
    SmallVectorImpl<PassingKind> &inputParamPassingKinds,
    SmallVectorImpl<ParamDeclAttr> &inputParamDecls,
    const ExprNode *resultTypeExpr, FnEffects &effects,
    SmallVectorImpl<ParsedArgument> &args, SmallVectorImpl<Type> &argTypes,
    SmallVectorImpl<TypedAttr> &defaults, bool isDef, SMLoc resultLoc,
    ASTDecl *fnDecl, SpecialFunctionInfo fnInfo,
    function_ref<void()> processSignature) {
  SharedState &shared = typeEmitter.shared;
  ASTDecl &sigDecl = typeEmitter.declScope;
  // If this definition is a struct/class member, compute the self type.
  ASTType selfType;
  if (fnDecl) {
    ASTDecl *parent = fnDecl->getParentDecl();
    if (isa<StructDeclOp, TraitDeclOp>(*parent)) {
      // The parent decl must be fully resolved in order to resolve any of its
      // members.
      assert(parent->resolvedness == DeclResolvedness::fully);
      selfType = parent->getSelfType();
    }
  }

  // HACK: Create a dummy value to assign to argument declarations during
  // argument and result type emission.
  MLIRContext *ctx = typeEmitter.shared.getContext();
  SmallVector<OwningOpRef<ParamConstantOp>> argVals;
  auto makeDummy = [&](Type type) -> Value {
    return *argVals.emplace_back(OpBuilder(ctx).create<ParamConstantOp>(
        UnknownLoc::get(ctx), UnboundAttr::get(type)));
  };

  // Resolve all argument types, generating type check error types for any types
  // that could not be correctly resolved.
  bool seenInitExpr = false;
  for (auto [idx, arg] : llvm::enumerate(args)) {
    ASTType type;
    if (arg.typeExpr) {
      // Emit the argument type. Allow argument types to be "automatically"
      // parameterized: if the type is fully unbound, its input parameters are
      // appended to the function input parameters.
      type = typeEmitter.emitExprType(arg.typeExpr, /*allowUnbound=*/true);

      // If the type couldn't be emitted, mark this argument erroneous (so uses
      // within the body of the function don't trigger secondary errors) and
      // mark the function erroneous so calls to it won't resolve.  Put in a
      // placeholder type so we can continue type checking.
      if (!type) {
        if (reportError())
          return {};
        type = shared.getTypeCheckErrorType();
      }
      type = addImplicitTypeParams(shared, type, arg, inputParamNames,
                                   inputParamPassingKinds, inputParamDecls);
    } else if (!idx && selfType && !cast<LIT::FuncOp>(fnDecl).getIsStatic()) {
      // If this is the 'self' argument in a struct, default the type to Self.
      type = selfType;
    } else if (isDef) {
      // In 'def', arguments with no types default to 'object'.
      type = shared.lookupObjectType(arg.loc, sigDecl);
      if (!type) {
        if (reportError())
          return {};
        type = shared.getTypeCheckErrorType();
      }
    } else {
      // In an 'fn' we report an error.
      shared.emitError(arg.loc, "'fn' argument type must be specified")
          << SourceRange(arg.loc, arg.loc);
      if (reportError())
        return {};
      type = shared.getTypeCheckErrorType();
    }
    assert(type && "must have an argument type");
    argTypes.push_back(type);

    // Determine the required function effects from the conventions.
    if (arg.vararg == VarArgKind::VarArg)
      effects.setVarArgs();
    else if (arg.vararg == VarArgKind::PackVarArg)
      effects.setPackVarArgs();
    else if (arg.vararg == VarArgKind::KWVarArg)
      effects.setKWVarArgs();

    // If no convention was explicitly specified, provide a default.  We default
    // to borrowed in an 'fn' or owned in a 'def'.
    if (arg.convention == ParsedArgument::kConventionUnspec) {
      arg.convention = isDef ? ParsedArgument::kConventionOwned
                             : ParsedArgument::kConventionBorrowed;
    }

    // Emit default argument values.
    if (const ExprNode *initExpr = arg.initExpr) {
      seenInitExpr = true;
      PValue value =
          typeEmitter.emitExprPValue(initExpr, EC_DefaultArgument, type);
      if (!value)
        return {};
      defaults.push_back(value);
    } else if (seenInitExpr) {
      InflightDiag diag = typeEmitter.emitError(
          arg.loc, "non-default argument follows default argument");
      // Depending on `reportError`, the type might also be missing.
      if (arg.typeExpr)
        diag << arg.typeExpr->getRange();
    }

    // Add the declaration for the argument, now that is has been resolved. Use
    // a placeholder value to allow the value to be referenced, but in function
    // body resolution, it will be replaced with the actual function argument
    // SSA value.
    if (!arg.name.empty()) {
      typeEmitter.getDeclResolver().addFullyResolvedDecl(
          SRValue(makeDummy(type)), arg.name, arg.loc, &typeEmitter.declScope);
    }
  }

  // Compute the result type. If it is memory-only, insert it into the argument
  // list to be added to the signature.
  ASTType resultType;
  if (!resultTypeExpr) {
    resultType = shared.getNoneType();
    // Don't insert the return value for certain special functions.
    if (isDef && !fnInfo.hasNoneResult() && !fnInfo.isInitializer()) {
      // Insert an object memory-only result type.
      ParsedArgument resultArg;
      resultArg.loc = resultLoc;
      resultArg.name = StringAttr::get(shared.getContext(), "__result__");
      resultArg.convention = ParsedArgument::kConventionInOutResult;
      resultArg.kwArgHandling = ParsedArgument::KWArgHandling::kPositionalOnly;
      args.insert(args.begin(), resultArg);
      argTypes.insert(argTypes.begin(),
                      shared.lookupObjectType(resultLoc, sigDecl));
      if (!argTypes.front()) {
        if (reportError())
          return {};
        argTypes.front() = shared.getTypeCheckErrorType();
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
    TypeConvention rp =
        resultType.getRegisterPassability(resultTypeExpr->getLoc(), shared);
    if (rp == TypeConvention::MemoryOnly) {
      // Synthesize a result argument for this, and use None as the actual
      // function result.
      ParsedArgument resultArg;
      resultArg.loc = resultTypeExpr->getLoc();
      resultArg.name = StringAttr::get(shared.getContext(), "__result__");
      resultArg.convention = ParsedArgument::kConventionInOutResult;
      resultArg.kwArgHandling = ParsedArgument::KWArgHandling::kPositionalOnly;
      resultArg.typeExpr = resultTypeExpr;
      args.insert(args.begin(), resultArg);
      argTypes.insert(argTypes.begin(), resultType);
      resultType = shared.getNoneType();
    } else if (rp != TypeConvention::RegisterPassableTrivial) {
      // We know the result type of the function is register passable (because
      // otherwise it would be promoted to an argument).  If the result of the
      // function is a non-trivial type, mark the function effect as having an
      // owned result so ownership tracking will notice it.
      effects.setOwnedRegisterResult();
    }
  }
  // While the signature decls are still in scope, do additional signature
  // processing.
  processSignature();
  return resultType;
}

void DeclResolver::computeArgumentConventions(
    SmallVectorImpl<ParamDeclAttr> &inputParamDecls,
    MutableArrayRef<ParsedArgument> args, MutableArrayRef<Type> argTypes) {
  for (auto [i, arg, argType] : llvm::enumerate(args, argTypes)) {
    switch (arg.convention) {
    case ParsedArgument::kConventionUnspec:
      llvm_unreachable("should be resolved by now");
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
      // We also force the passing kind of self to positional-only.
      arg.kwArgHandling = ParsedArgument::KWArgHandling::kPositionalOnly;
      arg.kgenConvention = ValueInputConvention::InitSelf;
      break;
    }

    // Adjust the MLIR type if needed.  Non-register values need to be passed
    // by pointer/reference.
    if (arg.kgenConvention != ValueInputConvention::OwnedInReg &&
        arg.kgenConvention != ValueInputConvention::BorrowedInReg) {

      // Values passed by memory need an associated lifetime parameter, and need
      // to be passed by reference.  Fun fact: explicit ref/mutref arguments
      // have register conventions, so they won't get these.
      if (false &&
          // FIXME: This is currently disabled because it causes literally
          // everything to explode.  We'll need to stage stuff in more
          // aggressively before going down this path and we don't want to
          // hork useExperimentalLifetimes beyond testing ability.
          shared.useExperimentalLifetimes()) {
        // Given a memory argument named "foo" we give the implicit lifetime a
        // name of "`foo".  We do this because of Rust precedent, but also
        // because you can't spell this identifier in Mojo, even with backticks!
        StringAttr lifetimeName;
        if (arg.name)
          lifetimeName = StringAttr::get(getContext(), "`" + arg.name.str());
        else // Used by function types, for example.
          lifetimeName = StringAttr::get(getContext(), "`" + llvm::utostr(i));
        auto lifetimeDecl =
            ParamDeclAttr::get(lifetimeName, shared.getLifetimeType());
        inputParamDecls.push_back(lifetimeDecl);

        // The parameter implicitly gets a reference type.
        bool isMutable = arg.convention != ParsedArgument::kConventionBorrowed;
        argType = RefType::get(
            isMutable, argType,
            ParamDeclRefAttr::get(lifetimeName, lifetimeDecl.getType()));
      } else {
        argType = PointerType::get(argType);
      }
    }
    if (arg.vararg == VarArgKind::VarArg)
      argType = KGEN::VariadicType::get(argType);
  }
}

//===----------------------------------------------------------------------===//
// Doc String support logic
//===----------------------------------------------------------------------===//

void ParserBase::parseDocString(ASTDecl &decl) {
  // The doc string is simply a follow-on string literal.
  Token docToken = getToken();
  if (!consumeIf(Token::string))
    return;
  if (auto astDeclOp = dyn_cast<ASTDeclInterface>(decl)) {
    StringRef docSpelling = docToken.getSpelling();
    Location loc = shared.diags.translateLocation(
        lexer.getStringLiteralStartLoc(docSpelling));

    astDeclOp.setDocStringAttr(DocStringAttr::get(
        StringAttr::get(getContext(), lexer.getStringLiteralValue(docSpelling)),
        dyn_cast<FileLineColLoc>(loc)));
  }
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
  // Decorators are applied to a decl starting from the one closest to it, so
  // reverse the vector.
  std::reverse(result.begin(), result.end());
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

/// Apply `@export` to an exportable declaration and register it with the shared
/// state to ensure no duplicate exports.
static void applyExport(SMLoc loc, SharedState &shared, ASTDecl &decl,
                        StringRef unmangledName, StringRef aliasName,
                        ExportInterface itf, bool isCExport = false) {
  // Handle the unique case of main. We implicitly export main, so this is
  // simply checking that the user didn't try to export it as something else.
  if (aliasName == kMainSymbolName) {
    if (unmangledName != kMainSymbolName)
      shared.emitError(loc, "only 'main' can be exported as 'main'");
    if (!isa<LIT::FuncOp>(decl))
      shared.emitError(loc, "exported 'main' must be a function");
    return;
  }
  if (unmangledName == kMainSymbolName) {
    shared.emitError(loc, "'main' can only be exported as 'main'");
    return;
  }

  llvm::TypeSwitch<ASTDecl &, void>(decl).Case<LIT::FuncOp, GlobalVarDeclOp>(
      [aliasName](auto op) { op.setLinkageName(aliasName); });
  if (isCExport)
    itf.setCExported();
  else
    itf.setExported();

  shared.declResolver->registerAndCheckExport(aliasName, loc);
}

/// Apply `@export("linkageName")` to an exportable declaration and register it
/// with the shared state to ensure no duplicate exports.
static void applyExport(SMLoc loc, SharedState &shared, ASTDecl &decl,
                        StringRef unmangledName, const CallNode &node,
                        ExportInterface itf) {
  ArrayRef<Operand> operands = node.operands;
  if (operands.empty() || operands.size() > 2) {
    shared.emitError(node.getLoc(), "@export requires 1 or 2 arguments");
    return;
  }

  std::optional<std::string> exportABI;
  std::optional<std::string> aliasName;
  for (const Operand &operand : operands) {
    auto strNode = dyn_cast<StringLiteralNode>(operand.value);
    if (strNode && operand.isKeyword() && operand.name == "ABI") {
      exportABI = strNode->getValue();
      if (*exportABI != "C") {
        shared.emitError(operand.getLoc(),
                         "only \"C\" ABI is supported at the moment");
        return;
      }
    } else if (strNode && operand.isPositional()) {
      aliasName = strNode->getValue();
    } else {
      shared.emitError(node.getLoc(),
                       "@export requires a string specifying the "
                       "name of the exported symbol");
      return;
    }
  }

  if (exportABI && aliasName && !isCIdentifier(*aliasName)) {
    shared.emitError(loc, *aliasName) << " is not a valid C identifier";
    return;
  }
  applyExport(loc, shared, decl, unmangledName,
              aliasName ? StringRef(*aliasName) : unmangledName, itf,
              exportABI.has_value());
}

namespace {
/// Decorators attached to a declaration may be "signature" decorators, "body"
/// decorators, compiler decorators, or dynamic decorators.
///
/// - Signature decorators are applied during the resolution of the signature of
///   a declaration before it is name bound.
/// - Body decorators are applied after the body of the declaration is fully
///   resolved.
/// - Compiler decorators (TODO) are applied at some stage in the Mojo
///   compilation pipeline.
/// - Dynamic decorators (TODO) are applied at the object at runtime.
///
/// This is the base class for handling decorators on declarations. Signature
/// decorators are processed first and then leftover decorators are persisted
/// until body resolution is complete via the SharedState.
struct Decorators : public SharedStateUser {
  Decorators(ASTDecl &decl, SharedState &shared)
      : SharedStateUser(shared), decl(decl) {}

  /// Process signature decorators on the declaration using the provided
  /// functor. The functor should return success if the decorator was processed
  /// as a signature decorator.
  void applySignatureDecorators(
      ArrayRef<std::pair<ExprNode *, LexerCursor>> decoratorExprs,
      function_ref<LogicalResult(ExprNode *)> process);

  /// Process body decorators on the declaration using the provided functor.
  /// The functor should return success if the decorator was processed as a
  /// signature decorator. Any leftover decorators are emitted and deferred to
  /// the operation.
  void applyBodyDecorators(function_ref<LogicalResult(ExprNode *)> process);

  /// The declaration this class is applying decorators to.
  ASTDecl &decl;
};
} // namespace

void Decorators::applySignatureDecorators(
    ArrayRef<std::pair<ExprNode *, LexerCursor>> decoratorExprs,
    function_ref<LogicalResult(ExprNode *)> process) {
  // Process decorators in the order they are seen. Stop at the first decorator
  // that needs to be deferred.
  while (true) {
    // Return if we are out of decorators.
    if (decoratorExprs.empty())
      return;
    if (failed(process(decoratorExprs.front().first)))
      break;
    decoratorExprs = decoratorExprs.drop_front();
  }
  // Ensure that there are no other signature decorators afterwards. This is
  // an error.
  SmallVector<ExprNode *> bodyDecorators;
  bodyDecorators.push_back(decoratorExprs.front().first);
  for (auto [i, decorator] :
       llvm::enumerate(llvm::make_first_range(decoratorExprs.drop_front()))) {
    if (failed(process(decorator))) {
      bodyDecorators.push_back(decorator);
      continue;
    }
    // If the decorator applies, we have an error.
    InflightDiag diag =
        emitError(decorator->getLoc(),
                  "signature decorator cannot come after body decorator")
        << decorator->getRange();
    ExprNode *bodyDecorator = decoratorExprs[i].first;
    diag.attachNote(bodyDecorator->getLoc())
        << "previous body decorator applied here" << bodyDecorator->getRange();
    break;
  }
  // Defer the rest of the decorators through the shared state.
  decl.setBodyDecorators(bodyDecorators, shared);
}

void Decorators::applyBodyDecorators(
    function_ref<LogicalResult(ExprNode *)> process) {
  // Don't run decorators if the declaration is invalid.
  if (decl.hasReferenceError)
    return;

  ArrayRef<ExprNode *> decoratorExprs = decl.getBodyDecorators(shared);
  while (true) {
    // If there are no decorators left, just exit.
    if (decoratorExprs.empty())
      return;
    if (failed(process(decoratorExprs.front())))
      break;
    decoratorExprs = decoratorExprs.drop_front();
  }

  // Emit the expressions and persist the resulting PValue into the IR. For now,
  // assume that all decorators are "compiler" decorators.
  // TODO: Emit an attempt to call the decorator value.
  SmallVector<TypedAttr> decoPValues;
  decoPValues.reserve(decoratorExprs.size());
  ExprEmitter emitter(shared, decl, EC_Decorator);
  for (auto [i, decorator] : llvm::enumerate(decoratorExprs)) {
    // Make sure we don't have another body decorator.
    if (failed(process(decorator))) {
      if (PValue decoVal = emitter.emitExprPValue(decorator, EC_Decorator))
        decoPValues.push_back(decoVal);
      continue;
    }
    // If the decorator applies, we have an error.
    InflightDiag diag =
        emitError(decorator->getLoc(),
                  "body decorator cannot come after compiler decorator")
        << decorator->getRange();
    ExprNode *bodyDecorator = decoratorExprs[i - 1];
    diag.attachNote(bodyDecorator->getLoc())
        << "previous compiler decorator applied here"
        << bodyDecorator->getRange();
    break;
  }

  TypeSwitch<ASTDecl &, void>(decl)
      .Case<LIT::FuncOp, StructDeclOp, GlobalVarDeclOp>([&](auto op) {
        op.setDecoratorsAttr(DecoratorsAttr::get(op.getContext(), decoPValues));
      });
}

//===----------------------------------------------------------------------===//
// Function Decl implementation
//===----------------------------------------------------------------------===//

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
                          ArrayRef<Type> argTypes, ASTType &resultType,
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
  if (ASTDecl *parent = decl.getParentDecl();
      parent && isa<StructDeclOp, TraitDeclOp>(*parent)) {
    // The parent decl must be fully resolved in order to resolve any of its
    // members.
    assert(parent->resolvedness == DeclResolvedness::fully);
    selfType = parent->getSelfType();
    // If there is an in-memory result, self is passed as arg #1 otherwise #0.
    selfArgNumber = hasMemoryResult ? 1 : 0;
  }

  // __*init__ methods are weird - for memory-only results we define
  // init in convention Python style, but for @register_passable values, we
  // return it.  We handle this by mapping them to different enumerators so
  // things downstream have stronger invariants.
  if ((fnInfo.kind == SpecialFunctionKind::kInit ||
       fnInfo.kind == SpecialFunctionKind::kCopyInit ||
       fnInfo.kind == SpecialFunctionKind::kMoveInit ||
       fnInfo.kind == SpecialFunctionKind::kTakeInit) &&
      selfType && ASTType(selfType).isRegisterPassable(decl.getLoc(), shared)) {
    if (fnInfo.kind == SpecialFunctionKind::kCopyInit)
      fnInfo = SpecialFunctionInfo::get(SpecialFunctionKind::kCopyInitReg);
    else if (fnInfo.kind == SpecialFunctionKind::kInit)
      fnInfo = SpecialFunctionInfo::get(SpecialFunctionKind::kInitReg);
    else {
      assert(fnInfo.kind == SpecialFunctionKind::kMoveInit ||
             fnInfo.kind == SpecialFunctionKind::kTakeInit);
      emitError() << name
                  << " is not supported for @register_passable types, they "
                     "are always movable by copying a register";
    }
  }

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
  if (fnInfo.kind != SpecialFunctionKind::kNormal) {
    size_t numActualArgs = args.size() - std::max(selfArgNumber, ssize_t(0));
    size_t numMin = fnInfo.minNumArguments;
    ssize_t numMax = fnInfo.maxNumArguments;
    if (numMin == size_t(numMax) && numActualArgs != numMin) {
      emitError("special function ")
          << name << " must have " << numMin << " operand" << plural(numMin);
    } else if (numActualArgs < numMin) {
      emitError("special function ") << name << " must have at least " << numMin
                                     << " operand" << plural(numMin);
    } else if (numMax != -1 && numActualArgs > size_t(numMax)) {
      emitError("special function ")
          << name << " must have at most " << size_t(numMax) << " operand"
          << plural(numMax);
    }
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

  ASTType declaredResultType =
      hasMemoryResult ? ASTType(argTypes[0]) : resultType;

  // Some functions like __new__ require a Self result type.
  if (fnInfo.flags & SpecialFunctionInfo::kSelfResult &&
      !declaredResultType.isEqualCanon(selfType))
    emitError() << name << " result type must be " << selfType;

  // If the function is required to return None, verify that.
  if (fnInfo.hasNoneResult() && !declaredResultType.isNoneType()) {
    emitError() << name << " result type must be elided (or None)";
    resultType = shared.getNoneType();
  }

  // Reject special functions declared as throwing when that is invalid.
  if (effects.isThrows() && fnInfo.flags & SpecialFunctionInfo::kCannotRaise) {
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

  // Diagnose common errors and handle other special cases.
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
  case SpecialFunctionKind::kMoveInit:
  case SpecialFunctionKind::kTakeInit: {
    // The first/self argument is syntactically declared as a by-ref argument,
    // but we need to change it to InitSelf since it is not initialized coming
    // in.
    assert(!args.empty() && "arg count already checked above");
    SMLoc selfArgLoc = args[0].loc;
    // __init__ methods must take their self argument by-ref syntactically.
    if (args[0].convention != ParsedArgument::kConventionInOut) {
      auto diag = emitErrorLoc(selfArgLoc, "'self' in struct ")
                  << name << " must be passed as mutable reference";
      if (args[0].convention == ParsedArgument::kConventionUnspec)
        diag << FixIt::insertAfterToken(selfArgLoc, "&", shared.diags);
    }

    // Regardless force it to init_self so recovery follows the fix-it.
    args[0].convention = ParsedArgument::kConventionInitSelfResult;
    // We also force the passing kind of self to positional-only.
    args[0].kwArgHandling = ParsedArgument::KWArgHandling::kPositionalOnly;

    if (fnInfo.kind == SpecialFunctionKind::kCopyInit) {
      if (args[1].convention != ParsedArgument::kConventionBorrowed)
        emitErrorLoc(args[1].loc,
                     "existing value argument must be passed as borrowed");
    } else if (fnInfo.kind == SpecialFunctionKind::kMoveInit) {
      if (args[1].convention != ParsedArgument::kConventionOwned)
        emitErrorLoc(args[1].loc,
                     "existing value argument must be passed as owned");
    } else if (fnInfo.kind == SpecialFunctionKind::kTakeInit) {
      if (args[1].convention != ParsedArgument::kConventionInOut)
        emitErrorLoc(args[1].loc,
                     "existing value argument must be passed as by-ref");
    }
    break;
  }
  }

  // If we have a special function kind and didn't have any errors with it,
  // remember which kind it is.
  if (fnInfo.kind != SpecialFunctionKind::kNormal)
    funcOp.setSpecialFnKind(uint8_t(fnInfo.kind));
}

/// Mangle 'name', ensuring that overloaded methods get unique symbol names.
/// TODO(#16040): Struct names mangled into the signature should be parameter
/// name-erased.
StringAttr DeclResolver::getMangledName(StringAttr baseName,
                                        SignatureType signature) {
  SmallString<64> mangledName(baseName.getValue().begin(),
                              baseName.getValue().end());
  llvm::raw_svector_ostream os(mangledName);
  ArrayRef<Type> inputParams = signature.getInputParamTypes();
  if (!inputParams.empty()) {
    os << '[';
    llvm::interleave(
        inputParams, os,
        [&](ASTType type) {
          os << type.getAsString(/*forDiag=*/false, /*demangleParams=*/true);
        },
        ",");
    os << ']';
  }

  mangledName += '(';
  for (auto [argNo, convention, argType] : llvm::enumerate(
           signature.getInputConventions(), signature.getValueInputs())) {
    // We do not mangle byref results into the signature.
    if (convention == ValueInputConvention::ByRefResult)
      continue;

    // Update the mangled name for this argument.
    if (argNo != 0)
      mangledName += ",";

    // If this had adjustments added to it because of its argument convention /
    // variadic state, strip them off.
    ASTType type = argType;
    // FIXME(#13015, #13603): In general, we shouldn't be checking for variadic
    // types specifically, but this is a quick stop-gap to address a crash.
    if (signature.isVarArg(argNo) && isa<VariadicType>(type.mlirType))
      type = type.getVariadicElementType();
    if (convention != ValueInputConvention::OwnedInReg &&
        convention != ValueInputConvention::BorrowedInReg)
      type = type.getReferenceElementType();
    mangledName += type.getAsString(/*forDiag=*/false, /*demangleParams=*/true);

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
    case ValueInputConvention::InitSelf:
      mangledName += "=&";
      break;
    case ValueInputConvention::ByRefResult:
      llvm_unreachable("byref_result should be skipped");
    case ValueInputConvention::None:
      llvm_unreachable("none convention not permitted in lit");
    }

    if (signature.isVarArg(argNo))
      mangledName += '*';
  }
  mangledName += ')';
  return StringAttr::get(baseName.getContext(), mangledName);
}

namespace {
struct FnDecorators : public SharedStateUser {
  FnDecorators(ASTDecl &decl, ASTDecl &sigDecl, SharedState &shared,
               StringRef baseName)
      : SharedStateUser(shared), decl(decl), sigDecl(sigDecl),
        funcOp(cast<LIT::FuncOp>(decl)), baseName(baseName) {}

  /// Apply a function signature decorator.
  LogicalResult apply(ExprNode *decorator, FnEffects &effects);

private:
  void applyAdaptive(const DeclRefNode &node);
  void applyMoveCapture(const CallNode &node);
  void applyLLVMMetadata(const CallNode &node);

  ASTDecl &decl;
  ASTDecl &sigDecl;
  LIT::FuncOp funcOp;
  StringRef baseName;
};
} // namespace

LogicalResult FnDecorators::apply(ExprNode *decorator, FnEffects &effects) {
  // Process all the decorators we know about.
  if (auto declRef = dyn_cast<DeclRefNode>(decorator)) {
    if (declRef->spelling == "export")
      applyExport(decorator->getLoc(), shared, decl, baseName, baseName,
                  funcOp);
    else if (declRef->spelling == "staticmethod")
      funcOp.setIsStatic(true);
    else if (declRef->spelling == "always_inline")
      funcOp.setInlineLevel(InlineLevel::Always);
    else if (declRef->spelling == "no_inline")
      funcOp.setInlineLevel(InlineLevel::Never);
    else if (declRef->spelling == "adaptive")
      applyAdaptive(*declRef);
    else if (declRef->spelling == "parameter")
      funcOp.setIsParametric(true);
    else if (declRef->spelling == "noncapturing")
      effects.setCapturing(false);
    else if (declRef->spelling == "closure")
      effects.setCapturing();
    else
      return failure();
    return success();
  }

  // `x()` forms.
  if (auto callNode = dyn_cast<CallNode>(decorator)) {
    if (auto declRef = dyn_cast<DeclRefNode>(callNode->callee)) {
      // @always_inline("nodebug")
      if (declRef->spelling == "always_inline" &&
          callNode->operands.size() == 1 &&
          callNode->operands[0].isPositionalStringLiteral("nodebug"))
        funcOp.setInlineLevel(InlineLevel::AlwaysNoDebug);
      else if (declRef->spelling == "export")
        applyExport(decorator->getLoc(), shared, decl, baseName, *callNode,
                    funcOp);
      else if (declRef->spelling == "__move_capture")
        applyMoveCapture(*callNode);
      else if (declRef->spelling == "__llvm_metadata")
        applyLLVMMetadata(*callNode);
      else
        return failure();
      return success();
    }
  }
  return failure();
}

void FnDecorators::applyAdaptive(const DeclRefNode &node) {
  if (funcOp.getIsAdaptive())
    emitError(node.getLoc(), "only one '@adaptive' decorator is allowed")
        << node.getRange();

  funcOp.setIsAdaptive(true);
}

void FnDecorators::applyMoveCapture(const CallNode &node) {
  // HACK(#16110): Need to implement proper capture list syntax rather than rely
  // on a special decorator.
  for (const Operand &operand : node.operands) {
    auto *declRef = dyn_cast<DeclRefNode>(operand.value);
    if (!declRef) {
      emitError(operand.getLoc(), "'@__move_capture' expected a declaration");
      continue;
    }

    LookupResult lookup = shared.lookupAndResolveDecl(
        declRef->spelling, declRef->getLoc(), *decl.getParentDecl(),
        /*searcInParentScopes=*/true);
    if (ArrayRef<ASTDecl *> decls = lookup.getIfSuccess(); !decls.empty()) {
      ExprEmitter emitter(shared, decl, EC_CaptureCopy);
      ValueDest dest(EC_CaptureCopy);
      std::optional<Capture> capture;
      if (emitter.emitDeclReference(declRef->spelling, decls, declRef, dest,
                                    capture) &&
          capture) {
        shared.addCaptureToScope(decl, decls.front(),
                                 Capture(capture->getValue(), /*isMove=*/true));
        continue;
      }
    }
    emitError(declRef->getLoc(), "cannot capture '")
        << declRef->spelling << "'";
  }
}

void FnDecorators::applyLLVMMetadata(const CallNode &node) {
  NamedAttrList attrs;
  ExprEmitter emitter(shared, sigDecl, EC_Decorator);
  for (Operand value : node.operands) {
    if (!value.name) {
      emitError(value.getLoc(), "LLVM metadata requires a name");
      continue;
    }
    if (PValue attr = emitter.emitExprPValue(value.value, EC_Decorator))
      attrs.append(value.name, attr);
  }
  funcOp.setLLVMMetadataAttr(attrs.getDictionary(getContext()));
}

/// Given the lexical context of a function, return true if the default bit
/// for the function is capturing.
/// FIXME: The language modeling here is a mess. It needs more thought.
static bool isCapturingByDefault(LIT::FuncOp funcOp, StructDeclOp parent,
                                 ArrayRef<ParamDeclAttr> inputParamDecls,
                                 ArrayRef<ParamDeclAttr> resultParamDecls) {
  // Nested functions are capturing by default.
  if (funcOp->getParentOfType<LIT::FuncOp>())
    return true;
  // Any function that contains a capturing closure as a parameter is itself
  // capturing, include parent struct parameters.
  mlir::AttrTypeWalker walker;
  walker.addWalk([](SignatureType sig) {
    if (sig.isCapturing())
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return llvm::any_of(
      llvm::concat<const ParamDeclAttr>(inputParamDecls, resultParamDecls,
                                        parent ? parent.getInputParams()
                                               : std::nullopt),
      [&](ParamDeclAttr decl) { return walker.walk(decl).wasInterrupted(); });
}

std::pair<SmallVector<ParamDeclRefAttr>, LITSignatureType>
DeclResolver::createSelfContainedSignature(LITSignatureType original) {
  // Collect the subset of referenced parameters. Use a set vector to keep the
  // order deterministic.
  llvm::SetVector<ParamDeclRefAttr, SmallVector<ParamDeclRefAttr>> capturedRefs;
  original.walk([&](ParamDeclRefAttr ref) { capturedRefs.insert(ref); });

  SmallVector<ParamDeclRefAttr> captured = capturedRefs.takeVector();
  // Unbind the N capture parameters, creating a new signature with N new input
  // parameters prepended.
  auto unbound = LITSignatureType::prependParams(
      original, llvm::map_to_vector(captured, [](ParamDeclRefAttr ref) {
        return ParamDeclAttr::get(ref);
      }));
  return {std::move(captured), unbound};
}

static Value emitClosureInstance(SignatureType closureSignature,
                                 SharedState &shared, ASTDecl &nestedFnDecl,
                                 SMLoc loc) {
  LIT::FuncOp nestedFn = cast<LIT::FuncOp>(nestedFnDecl);
  auto parentFn = nestedFn->getParentOfType<LIT::FuncOp>();
  assert(parentFn && "expected nested function to have a parent FuncOp");

  // Save the insertion point before closure creation since closure creation
  // nukes the nested function.
  ImplicitLocOpBuilder builder =
      ImplicitLocOpBuilder::atBlockEnd(parentFn.getLoc(), parentFn.getBody());
  builder.setInsertionPointAfter(nestedFn);
  OpBuilder::InsertPoint insertPoint = builder.saveInsertionPoint();
  ASTDecl *moduleDecl = nestedFnDecl.getNearestDeclOfType<FileModuleOp>();

  auto [capturedRefs, wrapperSig] =
      DeclResolver::createSelfContainedSignature(nestedFn.getSignature());
  if (!wrapperSig)
    return {};
  StructDeclOp closureWrapper =
      shared.getOrCreateClosureWrapper(loc, wrapperSig, moduleDecl);
  if (!closureWrapper)
    return {};

  // In order to emit a closure instance, we need the captures and in order to
  // compute the captures we need to resolve the body.
  if (failed(shared.declResolver->resolveFully(nestedFnDecl, loc)))
    return {};
  // Find all parameter captures in the function body.
  ParameterCollector::Analysis collectorCache;
  ParameterUseDefGraph graph(nestedFn.getBodyRegion());
  graph.calculate(collectorCache);
  SmallVector<ParamDeclRefAttr> paramCaptures =
      graph.usesFromAbove.takeVector();

  // Create an instance of the closure implementation in the parent function
  // right after the nested function definition.
  ClosureEmitter emitter(*moduleDecl, shared);
  StructDeclOp closureImpl =
      emitter.replaceNestedFunctionWithClosureImplStructDecl(
          loc, nestedFnDecl, paramCaptures, wrapperSig);

  // Map the closure wrapper captures to the impl captures.
  SmallDenseMap<unsigned, unsigned> fromImplToWrapperParameterMap;
  emitter.createWrapperInitWithImpl(closureWrapper, closureImpl,
                                    fromImplToWrapperParameterMap, loc);

  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (DebugInfo::DIScopeAttr spAttr = parentFn.getLocScope())
    diScopeGuard = shared.diBuilder->pushScopeGuard(spAttr);
  builder.restoreInsertionPoint(insertPoint);

  ExprEmitter exprEmitter(shared, *nestedFnDecl.getParentDecl(), builder);
  SyntheticNode node(loc);

  // Create a copy of the captured value.
  auto captureIteratorRange = shared.getCaptureRangeInScope(nestedFnDecl);
  SmallVector<ASTExprAnd<AnyValue>> closureImplInitArgs;
  for (auto &[_, capture] : captureIteratorRange) {
    AnyValue arg = capture.getValue();
    if (capture.isMoveCapture()) {
      // HACK(#16110): This transfers ownership without an explicit `^` from the
      // user, because we don't have capture list syntax.
      UnaryOpNode transfer(ExprNode::kTransfer, loc, &node);
      ValueDest dest(EC_CaptureCopy);
      arg = transfer.emitTransfer(arg, dest, exprEmitter);
    }
    closureImplInitArgs.push_back({arg, &node});
  }

  ValueDest closureDest;

  // Create Closure Impl type by adding captured parameters to the ClosureImpl
  // DeclType.
  Type closureImplType = closureImpl.bindReference(llvm::map_to_vector(
      paramCaptures, [](ParamDeclRefAttr ref) -> TypedAttr { return ref; }));
  CValue value = exprEmitter.emitConstructorCall(
      ASTType(closureImplType), closureImplInitArgs, &node,
      CallSyntax::kTypeCall, closureDest, /*allowImplicitConversion=*/false);
  // Emit the Closure Wrapper instance.
  ValueDest closureWrapperDest;
  SmallVector<ASTExprAnd<AnyValue>> closureWrapperInitArgs;
  closureWrapperInitArgs.push_back({value, &node});

  // Create the ClosureWrapper type by binding parent parameters to the
  // ClosureWrapper type.
  // TODO: Handle partial binding.
  DeclRefType closureWrapperType =
      closureWrapper.bindReference(llvm::map_to_vector(
          capturedRefs, [](ParamDeclRefAttr ref) -> TypedAttr { return ref; }));
  CValue closureWrapperInstance = exprEmitter.emitConstructorCall(
      ASTType(closureWrapperType), closureWrapperInitArgs, &node,
      CallSyntax::kTypeCall, closureWrapperDest,
      /*allowImplicitConversion=*/false);

  return closureWrapperInstance.getIfMRValue();
}

PassingKind ParsedArgument::mapToPassingKind(KWArgHandling handling) {
  switch (handling) {
  case ParsedArgument::KWArgHandling::kPositionalOnly:
    return PassingKind::PosOnly;
  case ParsedArgument::KWArgHandling::kKeywordOnly:
    return PassingKind::KwOnly;
  case ParsedArgument::KWArgHandling::kPositionalOrKeyword:
    return PassingKind::PosOrKw;
  }
  llvm_unreachable("unhandled ParsedArgument::KWArgHandling");
}

/// Silence internal verifier errors when constructing types from the parser. We
/// don't want to show these to the user.
static auto silenceErrors(MLIRContext *ctx) {
  return [ctx] {
    InFlightDiagnostic diag = mlir::emitError(UnknownLoc::get(ctx));
    diag.abandon();
    return diag;
  };
}

/// funcdef   ::=  [decorators] def_or_fn identifier [param_signature]
///                "(" [argument_list] ")" ["->" expression] ":" suite
/// def_or_fn ::= "def" | "fn"
///
LogicalResult DeclResolver::resolveSignature(LIT::FuncOp funcOp, Lexer &lexer,
                                             ASTDecl &decl) {
  ParserBase p(shared, lexer);
  auto decoratorExprs = p.parseDecorators(decl);
  assert(p.getToken().isAny(Token::kw_async, Token::kw_def, Token::kw_fn) &&
         "not a function definition?");
  FnEffects effects;
  effects.setAsync(p.consumeIf(Token::kw_async));
  if (p.getToken().is(Token::kw_def))
    effects.setThrows();
  p.consumeToken();

  StringAttr baseName;
  SMLoc identifierLoc;
  if (p.parseIdentifier(baseName, "expected function name", &identifierLoc))
    return failure();

  // The function signature is a self-contained scope where the input and result
  // parameters of the function are visible by all types.
  ASTDecl &sigDecl = addFullyResolvedDecl(nullptr, StringAttr(), decl.getLoc(),
                                          decl.getParentDecl());

  // Add meta parameters from an enclosing declaration to the symbol table.
  // These are /in/ our current scope because we do not want name conflicts with
  // them and they are instance (not type-level) values.
  // TODO: Generalize this to support nested structs and functions.
  bool paramVarArg = false;
  auto structDecl = dyn_cast<StructDeclOp>(decl.getParentDecl());
  if (structDecl) {
    for (ParamDeclAttr param : structDecl.getInputParams()) {
      auto paramRef = ParamDeclRefAttr::get(param);
      addFullyResolvedDecl(PValue(paramRef), param.getName(), decl.getLoc(),
                           &sigDecl);
    }
    paramVarArg = structDecl.getSignature().getParamVarArg();
  }

  // Parse declared meta parameters and add them to the current scope.
  SmallVector<ParamDeclAttr> inputParamDecls, resultParamDecls;
  SmallVector<ParsedArgument> args;
  SmallVector<StringAttr> paramNames;
  SmallVector<PassingKind> paramPassingKinds;
  SmallVector<TypedAttr> paramDefaults;

  // Add the meta parameters to the symbol table, and resolve their types.  We
  // add all of these after generic signature parsing so types used in the
  // signature list resolve to enclosing scopes, and we add them before the
  // value signature list so the types and parameters can resolve to the bound
  // values.
  if (parseOptionalParameterSignature(
          p, sigDecl, inputParamDecls, resultParamDecls, paramNames,
          paramPassingKinds, paramDefaults, paramVarArg))
    return failure();

  // Parse the argument list next if present.
  if (ParsedArgument::parseAndResolveParenthesizedArgumentList(
          p, args, ParsedArgument::ArgListKind::kArgList, effects))
    return failure();

  if (paramVarArg)
    effects.setParamVarArgs();

  // This doesn't support the capturing effect, reject it.
  if (effects.isCapturing()) {
    emitError(decl.getLoc(),
              "'capturing' effect not supported on this declaration");
    effects.setCapturing(false);
  }

  // Parse the result type if present.
  ExprNode *resultTypeExpr = nullptr;
  SMLoc resultLoc = p.getToken().getLoc();
  if (p.consumeIf(Token::minus_greater)) {
    if (p.parseExpression(resultTypeExpr))
      return failure();
  }

  // Check for a missing colon now, but don't yet bail out. We want to be able
  // to diagnose errors in the signature before we bail out, users often haven't
  // finished writing the signature when they hit the missing colon.
  ParseResult missingColon =
      p.parseToken(Token::colon, "expected ':' in function definition");

  // Emit the argument and result types.
  SmallVector<Type> argTypes;
  SmallVector<TypedAttr> argDefaults;
  auto reportError = [&] {
    decl.hasReferenceError = true;
    return success();
  };
  SpecialFunctionInfo fnInfo = SpecialFunctionInfo::get(baseName);

  // Process signature decorators in the same scope as signature resolution.
  auto processSignature = [&] {
    if (isCapturingByDefault(funcOp, structDecl, inputParamDecls,
                             resultParamDecls) &&
        !effects.isEscaping())
      effects.setCapturing();

    // Now that we have figured out the lexical structure, allow decorators to
    // take a crack at the signature.
    FnDecorators fnDecorators(decl, sigDecl, shared, baseName);
    Decorators(decl, shared)
        .applySignatureDecorators(decoratorExprs, [&](ExprNode *decorator) {
          return fnDecorators.apply(decorator, effects);
        });
  };

  ExprEmitter typeEmitter(shared, sigDecl, EC_Type);
  ASTType resultType = ParsedArgument::emitFunctionArgumentsAndResults(
      reportError, typeEmitter, paramNames, paramPassingKinds, inputParamDecls,
      resultTypeExpr, effects, args, argTypes, argDefaults, funcOp.getIsDef(),
      resultLoc, &decl, fnInfo, processSignature);
  if (!resultType)
    return failure();

  // Propagate errors and the parsed decls in the signature.
  moveDecls(decl, sigDecl);

  // Now that all the structural properties are determined, perform any
  // name-binding specific checks over the declaration.  This happens after
  // decorator processing because that is how defs work in Python.  This also
  // fills in any implicitly declared types.
  verifyFunctionNameBinding(decl, funcOp, baseName, args, argTypes, resultType,
                            effects, shared, fnInfo);

  // Now that all the types and signature information have been resolved,
  // compute the final MLIR types and KGEN conventions.  This also introduces
  // implicit lifetime parameters for borrows/inout/owned arguments.
  computeArgumentConventions(inputParamDecls, args, argTypes);

  // Now that we've processed the signature, bail if we had a missing colon.
  if (missingColon)
    return failure();

  // Finally now that the full signature has been resolved, build our IR.

  // First, handle function effects. If the function raises, it implicitly gets
  // a variant result type.
  if (effects.isThrows()) {
    ASTType errorType =
        shared.getBuiltinErrorType(*decl.getParentDecl(), decl.getLoc());
    if (errorType.isTypeCheckErrorType())
      decl.hasReferenceError = true;

    resultType = VariantType::get({errorType, resultType});
  }

  // Handle argument effects and build the ASTDecls for the arguments.
  SmallVector<Location> argLocs;
  SmallVector<StringAttr> argNames;
  SmallVector<PassingKind> argPassingKinds;
  SmallVector<ValueInputConvention> inputConventions;
  for (const ParsedArgument &arg : args) {
    argLocs.push_back(shared.diags.translateLocation(arg.loc));
    argPassingKinds.emplace_back(
        ParsedArgument::mapToPassingKind(arg.kwArgHandling));
    argNames.push_back(arg.name);
    inputConventions.push_back(arg.kgenConvention);
  }

  OpBuilder builder = decl.getDeclEndBuilder();
  NamedAttrList attrs = funcOp->getAttrDictionary();
  auto inputParamsAttr = builder.getAttr<ParamDeclArrayAttr>(inputParamDecls);
  auto resultParamsAttr = builder.getAttr<ParamDeclArrayAttr>(resultParamDecls);

  attrs.set(funcOp.getInputParamsAttrName(), inputParamsAttr);
  attrs.set(funcOp.getResultParamsAttrName(), resultParamsAttr);
  FunctionType functionType =
      builder.getFunctionType(argTypes, {resultType.mlirType});
  attrs.set(funcOp.getFunctionTypeAttrName(), TypeAttr::get(functionType));

  // Compute the signature of the function.
  auto metadata = FnMetadataAttr::get(
      builder.getContext(), argNames, argPassingKinds, paramNames,
      paramPassingKinds, argDefaults, paramDefaults);
  LITSignatureType signature = SignatureType::remapToSignature(
      inputParamsAttr, resultParamsAttr, functionType, inputConventions,
      effects, metadata, silenceErrors(getContext()));
  if (!signature)
    return failure();

  attrs.set(funcOp.getSignatureAttrName(), TypeAttr::get(signature));

  // Set the symbol to the mangled name and check for redefinition.
  attrs.set(funcOp.getSymNameAttrName(), getMangledName(baseName, signature));
  attrs.set(funcOp.getSourceNameAttrName(), baseName);

  // Remove the temporary "sym_namex" attribute set up in FuncOp::build, see
  // that method for an explanation.
  attrs.erase("sym_namex");

  // Bulk update the attributes.
  funcOp->setAttrs(attrs.getDictionary(funcOp.getContext()));

  // Set the symbol and notice if we are redeclaring something.
  if (Operation *existing = finalizeFuncSignature(funcOp, decl)) {
    const char *errorMessage = nullptr;
    auto existingFunc = cast<LIT::FuncOp>(existing);

    // We need to compare the (name erased) user result types, since memory-only
    // types may result in `!kgen.none` in the mlir signature result.
    auto resTy = ASTType(signature).getSignatureUserResultType();
    auto existingResTy =
        ASTType(existingFunc.getSignature()).getSignatureUserResultType();
    if (!resTy.isEqualCanon(existingResTy)) {
      errorMessage = " cannot overload on return type only";
    } else if (!existingFunc.getIsAdaptive()) {
      // If the results match, we only error if the function is not adaptive.
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
  if (!structDecl && baseName == kMainSymbolName)
    getDeclResolver().exportMain(decl);

  // Generate a debug subprogram for this function.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  shared.setLocationDebugScope(diScopeGuard, funcOp);

  // If this is a nested function, set its parameter declaration. It will be
  // referenced via parameter references instead of symbol references.
  if (funcOp->getParentOfType<LIT::FuncOp>())
    funcOp.setParamDeclAttr(
        ParamDeclAttr::get(funcOp.getSymNameAttr(), signature));

  funcOp.getBody()->addArguments(argTypes, argLocs);

  if (!funcOp->getParentOfType<FuncOp>() ||
      (!signature.isCapturing() && !signature.isEscaping()))
    funcOp.setIsParametric(true);

  // Upon fully resolving a nonparametric closure, immediately materialize it
  // as a runtime value. It cannot be used as a parameter.
  if (!funcOp.getIsParametric()) {
    // Fully resolve the body so we can swap the IR value of the decl. Later on,
    // we will need this to determine the capture signature.
    decl.resolvedness = DeclResolvedness::fully;
    if (failed(resolveBody(funcOp, lexer, decl)))
      return failure();

    // If the function doesn't actually capture anything, don't demote it to a
    // runtime value.
    bool hasCapture = false;
    mlir::visitUsedValuesDefinedAbove(funcOp.getBodyRegion(),
                                      [&](OpOperand *) { hasCapture = true; });
    if (hasCapture || signature.isEscaping()) {
      if (funcOp.getIsAdaptive()) {
        decl.hasReferenceError = true;
        return emitError(
            funcOp.getLoc(),
            "nonparametric capturing closure cannot be marked @adaptive");
      }
      if (!signature.isEscaping() &&
          (!inputParamDecls.empty() || !resultParamDecls.empty())) {
        return emitError(funcOp.getLoc(),
                         "nonparametric capturing closure cannot have input or "
                         "result parameters");
      }

      OpBuilder b(funcOp.getContext());
      b.setInsertionPointAfter(funcOp);
      auto parent = funcOp->getParentOfType<LIT::FuncOp>();
      if (!parent)
        return failure();
      // Emit Closure structures necessary for instantiating an escaping
      // closure.
      if (signature.isEscaping()) {
        if (!inputParamDecls.empty() || !resultParamDecls.empty())
          return emitError(
              funcOp.getLoc(),
              "escaping closures cannot have input or result parameters yet");
        if (auto closure =
                emitClosureInstance(signature, shared, decl, decl.getLoc()))
          decl.irValue = MBValue(closure);
        else
          return failure();
      } else {
        decl.irValue = SBValue(b.create<CreateClosureOp>(
            parent.getLoc(), signature,
            ParamDeclRefAttr::get(*funcOp.getParamDecl()), ValueRange()));
      }
    }
  }

  shared.notifyListenerOnFunctionDecl(decl, identifierLoc);
  return success();
}

static LetRegDeclOp makeVarArgLValueVarSlot(const CValue &argValue,
                                            StringAttr argName,
                                            ASTDecl &parentDecl,
                                            OpBuilder &builder, SMLoc loc,
                                            SharedState &shared) {
  Location mloc = shared.translateLocation(loc);
  ASTType varListType = shared.getBuiltinVariadicListInstantiation(
      parentDecl, loc, argValue.getRValueType().getVariadicElementType());

  // Emit the initializer expression into the slot.
  ExprEmitter emitter(shared, parentDecl, builder);

  // Expr to provide location information.
  DeclRefNode srcExpr(StringRef(loc.getPointer(), argName.size()));
  SRValue val = emitter.emitSRValue({argValue, &srcExpr}, EC_DefArgumentShadow,
                                    varListType);
  LetRegDeclOp declOp = builder.create<LetRegDeclOp>(mloc, argName, val);

  return declOp;
}

/// Create a mutable VarDecl for a function argument that captures its value.
/// argValue specifies the argument with the correct valuetype.
static VarLetDeclOp makeArgLValueVarSlot(const CValue &argValue,
                                         StringAttr argName,
                                         ASTDecl &parentDecl,
                                         OpBuilder &builder, SMLoc loc,
                                         SharedState &shared) {
  Location mloc = shared.translateLocation(loc);

  // Emit the initializer expression into the slot.
  ExprEmitter emitter(shared, parentDecl, builder);
  VarLetDeclOp varDecl = emitter.emitVarLetDecl(
      argName, argValue.getRValueType(), mloc, VarLetDeclKind::Implicit);

  // Expr to provide location information.
  DeclRefNode srcExpr(StringRef(loc.getPointer(), argName.size()));
  ValueDest dest(XLValue(varDecl), EC_DefArgumentShadow);
  if (!emitter.emitBValue({argValue, &srcExpr}, dest))
    dest.resetForError();

  return varDecl;
};

/// This adds a default return (lit.return of None, potentially converted
/// to a variant) and emits a EndFuncOp.
static void appendDefaultReturnAndEndOp(LIT::FuncOp func, ASTDecl &funcDecl,
                                        SharedState &shared) {
  Block &body = *func.getBody();
  auto b = ImplicitLocOpBuilder::atBlockEnd(func.getLoc(), &body);

  auto makeNoneReturn = [&] {
    // The function returns none.
    Value retVal = b.create<ParamConstantOp>(shared.getNoneAttr());

    // Wrap the result value if necessary.
    if (func.isThrows())
      retVal = b.create<VariantCreateOp>(func.getMLIRResultType(), retVal, 1);
    ExprEmitter::emitNormalReturn(b, retVal, funcDecl);
  };

  // If the function returns None, insert a "return None".
  ASTType normalResult = func.getUserResultType();
  if (normalResult.isNoneType() &&
      // No default return needed if we ended in a return.
      (body.empty() || !isa<LIT::ReturnOp>(body.back()))) {
    makeNoneReturn();
  } else if (func.getIsDef() && func.getSignature().hasMemoryOnlyResult()) {
    // If this `def` returns an object but is missing a return, insert one
    // automatically.
    auto objType = shared.lookupObjectType(funcDecl.getLoc(), funcDecl);
    if (objType &&
        objType.isEqualCanon(cast<PointerType>(func.getArgument(0).getType())
                                 .getElementType())) {
      // Emit `object()` into the memory type return slot.
      ExprEmitter emitter(shared, funcDecl, EC_ReturnValue);
      emitter.builder = b;
      ValueDest resultDest(MLValue(func.getArgument(0)), EC_ReturnValue);
      // Create a dummy node to pass down.
      SyntheticNode locExpr(funcDecl.getLoc());
      CValue result = emitter.emitConstructorCall(
          objType, {}, &locExpr, CallSyntax::kImplicitConvert, resultDest);
      if (!result || !emitter.emitResult(result, &locExpr, resultDest))
        resultDest.resetForError();
      else
        makeNoneReturn();
    }
  }

  // Insert the default end terminator.
  b.create<LIT::EndFuncOp>();
}

ParseResult DeclResolver::resolveBody(LIT::FuncOp funcOp, Lexer &lexer,
                                      ASTDecl &decl) {
  // Push the debug scope for this function if necessary so that nested
  // operations have proper debug info.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (DebugInfo::DIScopeAttr spAttr = funcOp.getLocScope())
    diScopeGuard = shared.diBuilder->pushScopeGuard(spAttr);

  // Set up information about value arguments.
  Block *bodyBlock = funcOp.getBody();
  auto builder = OpBuilder::atBlockEnd(bodyBlock);

  LITSignatureType funcSignature = funcOp.getSignature();

  // Set up the body of the fn/def, creating declarations for the value
  // parameters and adding them to the symbol table.
  for (auto [argName, bbArg, convention] :
       llvm::zip(funcSignature.getArgNames(), funcOp.getBody()->getArguments(),
                 funcSignature.getInputConventions())) {
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
    auto setDecl = [&](DeclIRValue value) -> LogicalResult {
      argDecl.setIRValue(value);
      argDecl.resolvedness = DeclResolvedness::fully;
      if (auto rv = argDecl.getIfRValue()) {
        if (rv.getType().isTypeCheckErrorType())
          argDecl.hasReferenceError = true;
      } else if (auto lv = argDecl.getIfMLValue()) {
        if (lv.getRValueType().isTypeCheckErrorType())
          argDecl.hasReferenceError = true;
      } else if (auto bv = argDecl.getIfBValue()) {
        if (bv.getRValueType().isTypeCheckErrorType())
          argDecl.hasReferenceError = true;
      }
      shared.notifyListenerOnArgumentDecl(argDecl, argDecl.getLoc());
      return success();
    };

    shared.buildArgDebugInfo(builder, bbArg, argName);

    // VarArg arguments are projected into a VariadicList.
    if (funcSignature.isVarArg(bbArg.getArgNumber())) {
      auto declOp = makeVarArgLValueVarSlot(SRValue(bbArg), argName, decl,
                                            builder, argDecl.getLoc(), shared);
      if (failed(setDecl(DeclIRValue(declOp))))
        return failure();
      continue;
    }

    // PackVarArg arguments are always treated as their kgen.pack type
    // by-value right now.  TODO(literals): Project to a tuple like thing.
    if (isa<PackType>(bbArg.getType())) {
      if (failed(setDecl(SRValue(bbArg))))
        return failure();
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
      argIRValue = MLValue(bbArg);
      break;

    case ValueInputConvention::OwnedInReg:
      argIRValue = makeArgLValueVarSlot(SRValue(bbArg), argName, decl, builder,
                                        argDecl.getLoc(), shared);
      break;

    case ValueInputConvention::BorrowedInReg:
      argIRValue = SBValue(bbArg);
      break;

    case ValueInputConvention::BorrowedInMem:
      argIRValue = MBValue(bbArg);
      break;
    case ValueInputConvention::None:
      llvm_unreachable("none convention not permitted in lit");
    }

    // Ok, now that we've figured out the IR representation of the ASTDecl,
    // install it.
    if (failed(setDecl(argIRValue)))
      return failure();
  }

  Block *body = funcOp.getBody();

  Operation *lastOpIterBefore =
      body->empty() ? nullptr : &body->getOperations().back();

  // With all the argument declarations set up, we can resolve the body of the
  // function.
  if (ParserBase(shared, lexer).parseSuite(decl))
    return failure();

  // Function body is empty if the body block is empty or the last operation in
  // the block is still the same as it was before parseSuite.
  bool emptyBody =
      body->empty() || (lastOpIterBefore == &body->getOperations().back());

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

  if (emptyBody && isa<TraitDeclOp>(*decl.getParentDecl())) {
    // Wipe out the body which may already contain some compiler generated
    // operations for handling argLValueVarSlot.
    body->walk([&](LIT::VarLetDeclOp op) {
      // Remove the value from parent's declsInScope first before destroying the
      // value.
      auto iter = decl.declsInScope.find(op.getNameAttr());
      if (iter != decl.declsInScope.end())
        iter->second.clear();
    });

    body->clear();
    // Don't append anything to an empty function if this is a trait function.
  } else {
    appendDefaultReturnAndEndOp(funcOp, decl, shared);
  }

  // Now that the body of the function is parsed, run any body decorators.
  Decorators(decl, shared).applyBodyDecorators([](ExprNode *decorator) {
    return failure();
  });

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

ParseResult DeclResolver::resolveBody(LIT::FileModuleOp op, Lexer &lexer,
                                      ASTDecl &decl) {
  // Push a scope for the file of this module.
  DebugInfo::DIBuilder::ScopeGuard fileGuard;
  if (shared.diBuilder) {
    auto &sourceMgr = shared.getSourceMgr();
    int fileId = sourceMgr.FindBufferContainingLoc(lexer.getToken().getLoc());
    if (fileId) {
      StringRef filename =
          sourceMgr.getMemoryBuffer(fileId)->getBufferIdentifier();
      fileGuard = shared.diBuilder->pushFile(filename, "/");
    }
  }

  return ParserBase(shared, lexer).parseSuite(decl);
}

//===----------------------------------------------------------------------===//
// Package Decl implementation
//===----------------------------------------------------------------------===//

ParseResult DeclResolver::resolveBody(LIT::PackageOp op, ASTDecl &decl) {
  // A source package corresponds to a directory, resolving the body requires
  // iterating the filesystem directory and importing the corresponding
  // children.

  // Grab the directory that this package is defined in.
  std::optional<std::string> directoryStr = shared.getModuleSourcePath(decl);
  if (!directoryStr)
    return emitError(op.getLoc(), "unable to locate package directory");

  std::error_code ec;
  std::filesystem::path directory(*directoryStr);
  if (!std::filesystem::is_directory(directory, ec) || ec)
    return emitError(op.getLoc(), "unable to locate package directory");

  // Iterate the directory and import nested modules.
  OpBuilder builder = decl.getDeclEndBuilder();
  SmallVector<std::string> nestedModules;
  for (const auto &entry : std::filesystem::directory_iterator(directory, ec)) {
    if (ec || !SharedState::isModuleOrPackagePath(entry.path()))
      continue;
    nestedModules.emplace_back(
        entry.path().filename().replace_extension().generic_string());
  }

  // Sort the nested modules to ensure that we get a deterministic filesystem
  // ordering across the different platforms.
  llvm::stable_sort(nestedModules);

  // Create an unresolved relative import for each nested module. That way we
  // only need to actually pull anything in from the filesystem if it gets
  // referenced.
  for (StringRef name : nestedModules) {
    StringAttr importName = builder.getStringAttr("." + name);
    StringAttr boundName = builder.getStringAttr("$" + name);
    auto importDecl = builder.create<LIT::UnresolvedImportOp>(
        op->getLoc(), importName, boundName, /*declName=*/StringAttr(),
        /*importNameLoc=*/mlir::LocationAttr(),
        /*destNameLoc=*/mlir::LocationAttr());
    getDeclResolver().addDecl(importDecl, decl.loc, boundName, &decl,
                              LexerCursor(), LexerCursor(), /*indentation=*/-1);

    // Create an alias for the unmangled module name to allow for simplified
    // indexing into this module.
    boundName = builder.getStringAttr(name);
    importDecl = builder.create<LIT::UnresolvedImportOp>(
        op->getLoc(), importName, boundName, /*declName=*/StringAttr(),
        /*importNameLoc=*/mlir::LocationAttr(),
        /*declNameLoc=*/mlir::LocationAttr());
    getDeclResolver().addDecl(importDecl, decl.loc, boundName, &decl,
                              LexerCursor(), LexerCursor(), /*indentation=*/-1);
  }

  // Create a full wildcard import from the __init__, as the symbols defined
  // there are visible from the package.
  StringAttr importModule = builder.getStringAttr(".__init__");
  builder.create<UnresolvedWildcardImportOp>(op->getLoc(), importModule,
                                             /*fullImport=*/true);
  decl.addUnresolvedWildCardImport(importModule, /*isFullImport=*/true,
                                   decl.loc);

  // Resolve the body of the __init__ within the package, and inherit some
  // attributes from it if they are present.
  LookupResult initResult =
      shared.lookupAndResolveDecl("__init__", decl.loc, decl,
                                  /*searchParentScopes=*/false);
  if (initResult.isSuccess()) {
    ASTDecl &initDecl = *initResult.getIfSuccess().front();
    if (failed(resolveFully(initDecl, decl.loc)))
      return failure();
    if (auto initDeclOp = dyn_cast<ASTDeclInterface>(initDecl)) {
      // Inherit the docstring from the __init__ if it is present.
      if (auto docstring = initDeclOp.getDocStringAttr())
        op.setDocStringAttr(docstring);
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// VarLetDecl implementation
//===----------------------------------------------------------------------===//

ParseResult DeclResolver::resolveBody(LetRegDeclOp op, Lexer &lexer,
                                      ASTDecl &decl) {
  return success();
}

//===----------------------------------------------------------------------===//
// GlobalVarDecl implementation
//===----------------------------------------------------------------------===//

LogicalResult DeclResolver::resolveSignature(GlobalVarDeclOp op, Lexer &lexer,
                                             ASTDecl &decl) {
  ParserBase p(shared, lexer);
  SmallVector<std::pair<ExprNode *, LexerCursor>> decoratorExprs =
      p.parseDecorators(decl);

  // Re-parse the preamble. The syntax should have been checked already.
  if (!p.consumeIf(Token::kw_var) && !p.consumeIf(Token::kw_let)) {
    return shared.emitError(
        decl.getLoc(), "internal error: should be checked by statement parser");
  }
  StringAttr name;
  SMLoc identifierLoc;
  if (p.parseIdentifier(name,
                        "internal error: should be checked by statement parser",
                        &identifierLoc))
    return failure();

  // Parse the type if present.
  ASTType parsedType;
  ExprEmitter emitter(shared, *decl.getParentDecl(), EC_VarInit);
  if (p.consumeIf(Token::colon)) {
    ExprNode *typeExpr = nullptr;
    if (p.parseExpression(typeExpr, decl.getIndentation()))
      return failure();
    parsedType = emitter.emitExprType(typeExpr);
    if (!parsedType)
      return failure();
  }

  // Global variables require an initializer.
  ExprNode *initExpr = nullptr;
  if (p.parseToken(Token::equal, "expected '=' in global variable") ||
      p.parseVarLetInitExpression(initExpr, decl.getIndentation()))
    return failure();

  // Emit the initializer into an initializer function. If we have a type, then
  // emit directly into the LValue. Otherwise emit into the global to infer its
  // type.
  ValueDest dest;
  ExprContext exprContext = op.getIsVar() ? EC_VarInit : EC_LetInit;
  if (parsedType) {
    op.setType(parsedType);
    DLValue result(
        RCRef<GlobalDLValue>::create(op, parsedType, initExpr->getLoc()));
    dest = ValueDest(result, exprContext);
  } else {
    // If we don't, we emit into the varOp itself, because this will infer the
    // type of the varOp from the initializer expression.
    dest = ValueDest(op, exprContext);
  }

  op.getCtor().push_back(new Block);
  emitter.builder = OpBuilder::atBlockBegin(&op.getCtor().front());
  if (!initExpr->emitIR(dest, emitter)) {
    dest.resetForError();
    return failure();
  }
  assert(!isa<UnresolvedType>(op.getType()) &&
         "RValue emission should have inferred var type");

  // Emit the destructor call, if present, into the destructor function.
  OverloadSet dtorFn(ASTType(op.getType()), "__del__", initExpr,
                     CallSyntax::kDestructor, shared,
                     /*no error on failure*/ {});
  op.getDtor().push_back(new Block);
  if (!dtorFn.isNull()) {
    emitter.builder = OpBuilder::atBlockBegin(&op.getDtor().front());
    MRValue owned(emitter.builder->create<GlobalVarRefOp>(op.getLoc(), op));
    PValue callee = dtorFn.filterOverloadSet(
        CallOperands({{owned, initExpr}}), /*allowImplicitConversions=*/true,
        /*emitDiagnosticOnFailure=*/true, emitter);
    if (!callee)
      return failure();
    ValueDest dest(EC_Destructor);
    if (!emitter.emitIndirectCall(callee, CallOperands({{owned, initExpr}}),
                                  dest, initExpr))
      return failure();
  }

  // Run signature decorators, if any.
  auto processDecorator = [&](ExprNode *decorator) -> LogicalResult {
    if (auto ref = dyn_cast<DeclRefNode>(decorator);
        ref && ref->spelling == "export") {
      applyExport(ref->getLoc(), shared, decl, name, name, op);
      return success();
    }
    if (auto call = dyn_cast<CallNode>(decorator)) {
      if (auto ref = dyn_cast<DeclRefNode>(call->callee);
          ref && ref->spelling == "export") {
        applyExport(call->getLoc(), shared, decl, name, *call, op);
        return success();
      }
    }
    return failure();
  };
  Decorators(decl, shared)
      .applySignatureDecorators(decoratorExprs, processDecorator);

  shared.notifyListenerOnVariableDecl(decl, identifierLoc);
  return success();
}

ParseResult DeclResolver::resolveBody(GlobalVarDeclOp op, Lexer &lexer,
                                      ASTDecl &decl) {
  Decorators(decl, shared).applyBodyDecorators([](ExprNode *decorator) {
    return failure();
  });
  return success();
}

//===----------------------------------------------------------------------===//
// Alias Decl implementation
//===----------------------------------------------------------------------===//

/// alias_decl_stmt ::= "alias" identifier ":" expression ["=" expression]
///                   | "alias" identifier "=" expression
///
LogicalResult DeclResolver::resolveSignature(AliasDeclOp aliasDeclOp,
                                             Lexer &lexer, ASTDecl &decl) {
  ParserBase p(shared, lexer);
  auto decoratorExprs = p.parseDecorators(decl);
  SMLoc identifierLoc;

  // Parse the type if present.
  if (p.parseToken(Token::kw_alias, "internal error: checked by stmt parser") ||
      p.parseIdentifier("internal error: checked by stmt parser",
                        &identifierLoc))
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
      p.emitError(aliasDeclOp.getLoc(),
                  "declaration must have either a type or an initializer");
      return failure();
    }

    // `alias x: Int` is a forward declaration of a return parameter from a
    // function call, so it must occur in a function.
    if (!aliasDeclOp->getParentOfType<LIT::FuncOp>()) {
      p.emitError(aliasDeclOp.getLoc(),
                  "parameter results may only be declared in a function");
      return failure();
    }

    // Ok, things seem set up right, replace the ParamDeclOp with the right
    // operation that will allow us to track things.
    OpBuilder builder(aliasDeclOp);
    Operation *forwardDecl = builder.create<AliasForwardDeclOp>(
        aliasDeclOp.getLoc(), aliasDeclOp.getName(), TypeAttr::get(type),
        mlir::LocationAttr(), DocStringAttr());
    decl.setIRValue(forwardDecl);

    // Remove the paramDeclOp from the IR, since we ended up changing our mind
    // about how to represent this.
    aliasDeclOp->erase();

    // The check that the alias was specified is handled when the function body
    // has been fully resolved.
    rejectDecorators(decoratorExprs, decl, shared);

    // Process the doc string of the alias.
    p.parseDocString(decl);

    shared.notifyListenerOnAliasDecl(decl, identifierLoc);
    return success();
  }

  // Otherwise this is a normal `alias` declaration with an initializer.
  ExprNode *initExpr = nullptr;
  if (p.parseExpression(initExpr, decl.getIndentation()))
    return failure();

  ASTDecl &parentDecl = *decl.getParentDecl();
  ExprEmitter emitter(shared, parentDecl, EC_AliasValue);

  // Emit the value and convert to the expected type if we know it.
  auto rhsValue = emitter.emitExprPValue(initExpr, EC_AliasValue, type);
  if (!rhsValue)
    return failure();

  // If we had no declared type (`alias x = 42`), infer the type from the
  // initializer.
  if (!type)
    type = rhsValue.getType();

  // Remember the value, and update the type from UnresolvedType.
  NamedAttrList attrs = aliasDeclOp->getAttrDictionary();
  attrs.set(aliasDeclOp.getValueAttrName(), rhsValue.get());
  attrs.set(aliasDeclOp.getParamDeclAttrName(),
            ParamDeclAttr::get(aliasDeclOp.getName(), type));
  aliasDeclOp->setAttrs(attrs.getDictionary(decl.getContext()));
  rejectDecorators(decoratorExprs, decl, shared);

  // Process the doc string of the alias.
  p.parseDocString(decl);

  shared.notifyListenerOnAliasDecl(decl, identifierLoc);
  return success();
}

ParseResult DeclResolver::resolveBody(AliasDeclOp op, Lexer &lexer,
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
/// and return success, otherwise failure if it is handled later.
static LogicalResult processStructSignatureDecorator(ExprNode *decorator,
                                                     StructDeclOp structOp,
                                                     SharedState &shared,
                                                     ASTDecl &structDecl) {
  if (auto declRef = dyn_cast<DeclRefNode>(decorator)) {
    if (declRef->spelling == "register_passable") {
      structOp.setConvention(TypeConvention::RegisterPassable);
      return success();
    }
  }

  // `x()` forms.
  if (auto callNode = dyn_cast<CallNode>(decorator)) {
    if (auto declRef = dyn_cast<DeclRefNode>(callNode->callee)) {
      // @register_passable("trivial")
      if (declRef->spelling == "register_passable" &&
          callNode->operands.size() == 1 &&
          callNode->operands[0].isPositionalStringLiteral("trivial")) {
        structOp.setConvention(TypeConvention::RegisterPassableTrivial);
        return success();
      }

      // @nonmaterializable(TargetType)
      if (declRef->spelling == "nonmaterializable" &&
          callNode->operands.size() == 1)
        if (auto drn = dyn_cast<DeclRefNode>(callNode->operands[0].value))
          if (auto parentDecl = structDecl.getParentDecl()) {
            ExprEmitter emitter(shared, *parentDecl, EC_Type);
            if (ASTType t = emitter.emitExprType(drn)) {
              structOp.setNonmaterializableTargetAttr(
                  TypeAttr::get(t.mlirType));
              return success();
            }
          }
    }
  }
  // Not handled in signature phase.
  return failure();
}

static ParseResult
parseOptionalParentList(ParserBase &p, ASTDecl &declScope,
                        SmallVectorImpl<SymbolRefAttr> &traits,
                        SharedState &shared) {
  if (!p.consumeIf(Token::l_paren) || p.consumeIf(Token::r_paren))
    return success();

  SmallVector<ParsedArgument> names;
  if (ParsedArgument::parseAndResolvePresentArgumentList(
          p, names, ParsedArgument::ArgListKind::kArgList))
    return failure();

  // Resolve traits.
  // TODO: use `emitExprType` when we have types for traits.
  for (ParsedArgument name : names) {
    auto lookupResult = shared.lookupAndResolveDecl(
        name.name, declScope.getLoc(), *declScope.getParentDecl(),
        /*searchParentScopes*/ true);
    ArrayRef<ASTDecl *> decls = lookupResult.getIfSuccess();
    if (!decls.empty()) {
      traits.push_back(decls.front()->getSymbolRef());
      continue;
    }
    p.emitError(declScope.getLoc(), "expected to find a trait decl of ")
        << name.name << " for struct";
    declScope.hasReferenceError = true;
  }

  return p.parseToken(Token::r_paren, "expected ')' for parameter list");
}

/// structdef ::=
///   [decorators] "struct" identifier [param_signature] ":" suite
///
LogicalResult DeclResolver::resolveSignature(StructDeclOp structOp,
                                             Lexer &lexer, ASTDecl &decl) {
  ParserBase p(shared, lexer);
  auto decoratorExprs = p.parseDecorators(decl);

  // The signature of a struct is a self-contained decl where types can
  // reference the struct parameters.
  ASTDecl &sigDecl = addFullyResolvedDecl(nullptr, StringAttr(), decl.getLoc(),
                                          decl.getParentDecl());

  SmallVector<ParamDeclAttr> inputParamDecls, resultParamDecls;
  SmallVector<StringAttr> paramNames;
  SmallVector<PassingKind> paramPassingKinds;
  SmallVector<TypedAttr> paramDefaults;
  SmallVector<SymbolRefAttr> traits;

  bool paramVarArgs = false;
  SMLoc identifierLoc;
  if (p.parseToken(Token::kw_struct,
                   "internal error: checked by stmt parser") ||
      p.parseIdentifier("internal error: checked by stmt parser",
                        &identifierLoc) ||
      parseOptionalParameterSignature(
          p, sigDecl, inputParamDecls, resultParamDecls, paramNames,
          paramPassingKinds, paramDefaults, paramVarArgs) ||
      parseOptionalParentList(p, sigDecl, traits, shared) ||
      p.parseToken(Token::colon, "expected ':' in struct definition") ||
      decl.hasReferenceError)
    return failure();

  // Propagate signature errors and decls.
  moveDecls(decl, sigDecl);

  auto inputParams = ParamDeclArrayAttr::get(getContext(), inputParamDecls);
  structOp.setInputParamsAttr(inputParams);
  auto sig = TypeSignatureType::remapToSignature(
      silenceErrors(getContext()), inputParams, paramNames, paramPassingKinds,
      paramDefaults, paramVarArgs);
  if (!sig)
    return failure();
  structOp.setSignature(sig);

  if (!traits.empty())
    structOp.setTraitsAttr(
        M::SymbolRefArrayAttr::get(decl.getContext(), traits));

  // Reject result parameters.
  if (!resultParamDecls.empty())
    emitError(decl.getLoc(),
              "struct declarations do not support result parameters");

  // This is a struct, so we can use 'computeSelfTypeForStruct' to figure out
  // the self type.
  decl.setSelfType(ASTDecl::computeSelfTypeForStruct(structOp));

  // Structs are memory-only unless they opt-in to being passed in registers.
  structOp.setConvention(TypeConvention::MemoryOnly);

  // Now that we have the basic struct set up, process signature decorators.
  Decorators(decl, shared)
      .applySignatureDecorators(decoratorExprs, [&](ExprNode *decorator) {
        return processStructSignatureDecorator(decorator, structOp, shared,
                                               decl);
      });

  shared.notifyListenerOnStructDecl(decl, identifierLoc);
  return success();
}

//===----------------------------------------------------------------------===//
// Trait Decl implementation
//===----------------------------------------------------------------------===//

LogicalResult DeclResolver::resolveSignature(TraitDeclOp traitOp, Lexer &lexer,
                                             ASTDecl &decl) {
  ParserBase p(shared, lexer);
  auto decoratorExprs = p.parseDecorators(decl);

  SMLoc identifierLoc;
  if (p.parseToken(Token::kw_trait, "internal error: checked by stmt parser") ||
      p.parseIdentifier("internal error: checked by trait parser",
                        &identifierLoc))
    return failure();

  if (p.consumeIf(Token::l_square)) {
    // If the current token is on a new line, report the error on the end of
    // the previous line, this is probably where the punctuation was omitted.
    auto diagLoc = p.getTokenLocOrEndOfPreviousLineIfOnNewLine();
    // Report the error.
    emitError(diagLoc,
              "TODO: trait declarations do not support parameters yet");
    return failure();
  }

  if (p.parseToken(Token::colon, "expected ':' in trait definition"))
    return failure();

  // Insert the implicit trait parameters:
  // - MT: an AnyRegTypeType which points to the struct that implements this
  // trait.
  // - T: a ParamRef to MT which is the type of MT.
  // TODO: build AnyType instead
  auto mt = ParamDeclAttr::get("MT", AnyRegTypeType::get(decl.getContext()));
  auto mtRef = ParamDeclAttr::get(
      "T", KGEN::ParamRefType::get(KGEN::ParamDeclRefAttr::get(mt)));

  auto inputParams = ParamDeclArrayAttr::get(getContext(), {mt, mtRef});
  traitOp.setInputParams(inputParams);
  SmallVector<StringAttr> paramNames{StringAttr::get(decl.getContext(), ""),
                                     StringAttr::get(decl.getContext(), "")};
  SmallVector<PassingKind> paramPassingKinds{PassingKind::Implicit,
                                             PassingKind::Implicit};
  SmallVector<TypedAttr> paramDefaults;
  auto sig = TypeSignatureType::remapToSignature(
      silenceErrors(getContext()), inputParams, paramNames, paramPassingKinds,
      paramDefaults, false);
  if (!sig)
    return failure();
  traitOp.setSignature(sig);

  decl.setSelfType(ASTDecl::computeSelfTypeForTrait(traitOp));

  shared.notifyListenerOnTraitDecl(decl, identifierLoc);

  return success();
}

/// Look up the __del__ destructor for the specified `type` which is needed
/// for the specified declaration (typically a var or argument declaration).
/// This returns the destructor if successful, diagnoses an error if not, and
/// returns null if there is no defined destructor.
static SymbolConstantAttr lookupDestructor(ASTDecl &structDecl,
                                           SharedState &shared) {
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
  return func.getBoundSymbolRef();
}

/// Look up a __copyinit__/__moveinit__/__takeinit__  impl for the specified
/// `type`.  This returns the method if successful, and returns null if there is
/// none.
static SymbolConstantAttr
lookupCopyMoveTakeInit(ASTDecl &structDecl, SharedState &shared,
                       SpecialFunctionKind specialKind) {
  const char *name = SpecialFunctionInfo::get(specialKind).name;
  LookupResult inits = shared.lookupAndResolveDecl(
      name, structDecl.getLoc(), structDecl, /*searchParentScopes=*/false);
  ArrayRef<ASTDecl *> entries = inits.getIfSuccess();
  for (ASTDecl *candidate : entries) {
    LIT::FuncOp func = dyn_cast<LIT::FuncOp>(candidate);
    if (func && func.getSpecialFunctionKind() == specialKind)
      return func.getBoundSymbolRef();
  }
  return {};
}

/// Given a struct that has no explicitly defined __del__ member, define a new
/// one with an empty body.  This allows the CheckLifetimes pass to insert field
/// dels as needed, and makes sure that anything that refers to this struct
/// properly runs its destructor.
static SymbolConstantAttr synthesizeEmptyDtor(SharedState &shared,
                                              StructDeclOp structOp,
                                              ASTDecl &structDecl,
                                              DeclResolver &resolver) {
  auto builder = ImplicitLocOpBuilder::atBlockEnd(
      structOp.getLoc(), &structOp.getFields().front());

  // Figure out the type of the 'self' argument.  It is the struct's `Self`
  // type for register passable things, or indirect for a memory-only type.
  ASTType selfType = structDecl.getSelfType();
  // The argument is always owned.
  ValueInputConvention convention = ValueInputConvention::OwnedInReg;
  if (!selfType.isRegisterPassable(structDecl.getLoc(), resolver.shared)) {
    selfType = PointerType::get(selfType);
    convention = ValueInputConvention::OwnedInMem;
  }

  StringAttr selfName = builder.getStringAttr("self");

  // Create the FuncOp and ASTDecl for the method.
  StructEmitter emitter(shared);
  auto [funcOp, funcDecl] = emitter.synthesizeMethodInStruct(
      "__del__", /*inputParameters=*/{}, /*paramPassingKinds=*/{},
      selfType.mlirType, convention, selfName, PassingKind::PosOnly,
      shared.getNoneType(), structDecl, SpecialFunctionKind::kDel);

  // Set up the body.
  Block *body = funcOp.getBody();
  BlockArgument arg = body->getArgument(0);

  // We need to make a var box + store for register_passable values since that
  // is what lifetime tracking expects.
  if (convention == ValueInputConvention::OwnedInReg) {
    builder.setInsertionPointToStart(body);
    (void)makeArgLValueVarSlot(SRValue(arg), selfName, funcDecl, builder,
                               structDecl.getLoc(), resolver.shared);
  }

  // Finish off the function with a return + lit.endfunc.
  appendDefaultReturnAndEndOp(funcOp, funcDecl, resolver.shared);

  return funcOp.getBoundSymbolRef();
}

struct StructBodyDecorators : public SharedStateUser {
  StructBodyDecorators(
      StructDeclOp structOp, ASTDecl &structDecl, DeclResolver &resolver,
      ArrayRef<std::pair<StructFieldOp, ASTDecl *>> structFields)
      : SharedStateUser(resolver.shared), structOp(structOp),
        structDecl(structDecl), resolver(resolver), structFields(structFields) {
  }

  LogicalResult processDecorator(ExprNode *decorator);

private:
  void processValueDecorator(SMLoc decoratorLoc);
  void processRegisterPassableDecorator(bool isTrivial);

  StructDeclOp structOp;
  ASTDecl &structDecl;
  DeclResolver &resolver;
  ArrayRef<std::pair<StructFieldOp, ASTDecl *>> structFields;
};

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

  StructEmitter structEmitter(shared);
  StructDeclOp declOp = dyn_cast<StructDeclOp>(structDecl);
  std::optional<GeneratedStubs> stubs =
      structEmitter.addMissingValueMemberStubsToStruct(
          structDecl, /*generateFieldwiseInit=*/true);
  if (!stubs) {
    emitError(decoratorLoc, "'@value' cannot synthesize members of struct '")
        << declOp.getSymName() << "'";
    return;
  }
  if (LIT::FuncOp copyCtr = stubs->getCopyConstructor()) {
    SymbolConstantAttr ref = copyCtr.getBoundSymbolRef();
    ASTDecl *copyCtrDecl =
        getDeclResolver().getDeclForFuncSymbol(ref.getSymbol());
    if (failed(structEmitter.populateMoveCopy(*copyCtrDecl, /*isMove=*/false)))
      copyCtr.erase();
    else
      declOp.setCopyInitAttr(ref);
  }
  if (LIT::FuncOp moveCtr = stubs->getMoveConstructor()) {
    SymbolConstantAttr ref = moveCtr.getBoundSymbolRef();
    ASTDecl *moveCtrDecl =
        getDeclResolver().getDeclForFuncSymbol(ref.getSymbol());
    if (failed(structEmitter.populateMoveCopy(*moveCtrDecl, /*isMove=*/true)))
      moveCtr.erase();
    else
      declOp.setMoveInitAttr(ref);
  }
}

LogicalResult StructBodyDecorators::processDecorator(ExprNode *decorator) {
  if (auto declRef = dyn_cast<DeclRefNode>(decorator)) {
    if (declRef->spelling == "value") {
      processValueDecorator(decorator->getRangeStart());
      return success();
    }
    return failure();
  }
  return failure();
}

/// Process the @register_passable decorator on structs.  This finalizes
/// semantic checks.
static void processRegisterPassableDecorator(
    StructDeclOp structOp, ASTDecl &structDecl,
    ArrayRef<std::pair<StructFieldOp, ASTDecl *>> structFields,
    DeclResolver &resolver, TypeConvention structPassability) {

  bool isTrivial = structPassability == TypeConvention::RegisterPassableTrivial;
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

//===----------------------------------------------------------------------===//
// Conformance Check
//===----------------------------------------------------------------------===//

/// Get specialized signature of a trait function with a struct (who implements
/// the trait) type.
static SignatureType getSpecializedSignature(LIT::FuncOp traitFn,
                                             Type structSelfType,
                                             Type structSelfMetaType) {
  auto signature = traitFn.getFullSignature();
  SmallVector<TypedAttr> newInputParamValues;
  SmallVector<Type> newInputParamTypes;

  ArrayRef<Type> inputParamTypes = signature.getInputParamTypes();

  // Add trait's MT replacement.
  // FIXME(generics): We aren't propagating metatypes into pointer types, so
  // just pass a generic metatype here.
  newInputParamValues.push_back(
      TypeConstantAttr::get(AnyRegTypeType::get(traitFn.getContext())));
  // Add trait's T replacement.
  newInputParamValues.push_back(TypeConstantAttr::get(
      structSelfType, AnyRegTypeType::get(traitFn.getContext())));

  for (Type type : inputParamTypes.drop_front(2))
    newInputParamValues.push_back(UnboundAttr::get(type));

  return signature.getSpecializedSignature(newInputParamValues);
}

/// Check conformance for struct that implements traits.
static LogicalResult verifyConformance(LIT::StructDeclOp structDeclOp,
                                       ASTDecl &structDecl, Type structSelfType,
                                       Type structSelfMetaType,
                                       SharedState &shared) {
  InflightDiag diag =
      shared.emitError(structDeclOp.getLoc(), "conformance check failed");

  llvm::SmallVector<StringRef> failedTraits;
  for (SymbolRefAttr attr : structDeclOp.getTraitsAttr()) {
    ASTDecl &traitDecl = shared.declResolver->getDeclForTypeSymbol(attr);
    bool allMatch = true;
    for (auto &[name, decls] : traitDecl.getDeclsInScope()) {
      for (ASTDecl *decl : decls) {
        auto traitFn = cast<LIT::FuncOp>(*decl);
        SignatureType newSignature = getSpecializedSignature(
            traitFn, structSelfType, structSelfMetaType);

        ArrayRef<ASTDecl *> structFnDecls =
            structDecl.lookupInCurrentScope(name);
        bool foundMatch = false;
        for (ASTDecl *structFnDecl : structFnDecls)
          if (auto structFn = dyn_cast<LIT::FuncOp>(*structFnDecl))
            foundMatch |= newSignature == structFn.getSignature();

        if (foundMatch)
          continue;

        allMatch &= foundMatch;
        diag.attachNote(traitFn.getLoc())
            << "required function '" + name.str() + "' is not implemented";
      }
    }
    if (!allMatch)
      failedTraits.push_back(traitDecl.getNameIfOperation().value());
  }

  if (failedTraits.empty()) {
    diag.abandon();
    return success();
  }

  std::string errMsg;
  llvm::raw_string_ostream os(errMsg);
  os << "struct '" << structDeclOp.getNameAttr().str()
     << "' does not implement all requirements for ";
  for (auto [idx, failedTrait] : llvm::enumerate(failedTraits)) {
    os << "'" << failedTrait << "'";
    if (idx < failedTraits.size() - 1)
      os << ", ";
  }

  diag.attachNote(structDeclOp.getLoc()) << os.str();
  return failure();
}

ParseResult DeclResolver::resolveBody(StructDeclOp structOp, Lexer &lexer,
                                      ASTDecl &structDecl) {
  // Push the debug scope for this struct if necessary so that nested operations
  // have proper debug info.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (DebugInfo::DIScopeAttr spAttr = structOp.getLocScope())
    diScopeGuard = shared.diBuilder->pushScopeGuard(spAttr);

  if (ParserBase(shared, lexer).parseSuite(structDecl))
    return failure();

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
  if (structOp.isRegisterPassable()) {
    // TODO: Split trivial and register_passable apart.
    processRegisterPassableDecorator(structOp, structDecl, structFields, *this,
                                     structOp.getConvention());
  }

  // If there are any body decorators, resolve them now.
  StructBodyDecorators structDecorators(structOp, structDecl, *this,
                                        structFields);
  Decorators(structDecl, shared).applyBodyDecorators([&](ExprNode *decorator) {
    return structDecorators.processDecorator(decorator);
  });

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
        synthesizeEmptyDtor(shared, structOp, structDecl, *this));
  }
  // Look up move and copy constructors.
  if (!structOp.isRegisterPassable()) {
    if (auto copyInitAttr = lookupCopyMoveTakeInit(
            structDecl, shared, SpecialFunctionKind::kCopyInit))
      structOp.setCopyInitAttr(copyInitAttr);

    // We prefer a __moveinit__ over __takeinit__ because that allows copy
    // elision to completely eliminate the destructor calls.  We'll use
    // __takeinit__ if that's all we can get though.
    if (auto moveInitAttr = lookupCopyMoveTakeInit(
            structDecl, shared, SpecialFunctionKind::kMoveInit))
      structOp.setMoveInitAttr(moveInitAttr);
    else if (auto takeInitAttr = lookupCopyMoveTakeInit(
                 structDecl, shared, SpecialFunctionKind::kTakeInit))
      structOp.setMoveInitAttr(takeInitAttr);
  }

  if (!structOp.getTraitsAttr())
    return success();

  //// Resolve struct member functions' signature if they are candidates to
  /// implement trait functions so that we do conformance check next.
  for (SymbolRefAttr attr : structOp.getTraitsAttr()) {
    ASTDecl &traitDecl = shared.declResolver->getDeclForTypeSymbol(attr);
    for (auto &[name, _] : traitDecl.getDeclsInScope()) {
      ArrayRef<ASTDecl *> decls = structDecl.lookupInCurrentScope(name);
      for (ASTDecl *decl : decls) {
        if (isa<LIT::FuncOp>(*decl))
          if (failed(
                  resolve(*decl, DeclResolvedness::signature, decl->getLoc())))
            return failure();
      }
    }
  }

  return verifyConformance(structOp, structDecl, structDecl.getSelfType(),
                           structDecl.getSelfType().getMetaType(), shared);
}

//===----------------------------------------------------------------------===//
// StructFieldDecl implementation
//===----------------------------------------------------------------------===//

/// struct_field_decl_stmt ::= "var" identifier ":" expression
/// TODO: Support default values?
///
LogicalResult DeclResolver::resolveSignature(StructFieldOp fieldOp,
                                             Lexer &lexer, ASTDecl &decl) {
  ParserBase p(shared, lexer);
  auto decoratorExprs = p.parseDecorators(decl);

  ASTType type;
  SMLoc identifierLoc;
  // Parse the type if present.
  p.consumeToken(); // let or var.
  if (p.parseIdentifier("internal error: checked by stmt parser",
                        &identifierLoc) ||
      p.parseToken(Token::colon, "struct field declaration must have a type") ||
      parseType(p, type, *decl.getParentDecl(), decl.getIndentation()))
    return failure();

  fieldOp.setType(type);
  rejectDecorators(decoratorExprs, decl, shared);
  shared.notifyListenerOnStructFieldDecl(decl, identifierLoc);
  return success();
}

ParseResult DeclResolver::resolveBody(StructFieldOp op, Lexer &lexer,
                                      ASTDecl &decl) {
  return success();
}

//===----------------------------------------------------------------------===//
// TraitFieldDecl implementation
//===----------------------------------------------------------------------===//

ParseResult DeclResolver::resolveBody(TraitDeclOp traitOp, Lexer &lexer,
                                      ASTDecl &traitDecl) {
  // Push the debug scope for this trait if necessary so that nested operations
  // have proper debug info.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (DebugInfo::DIScopeAttr spAttr = traitOp.getLocScope())
    diScopeGuard = shared.diBuilder->pushScopeGuard(spAttr);

  if (ParserBase(shared, lexer).parseSuite(traitDecl))
    return failure();

  // Resolve TraitDeclOp's body here so that we get more information about its
  // functions right away.
  for (auto &decls : llvm::make_second_range(traitDecl.declsInScope)) {
    for (ASTDecl *decl : decls)
      // Only fully resolve children of LIT::FuncOp type.
      if (decl->getParentDecl() == &traitDecl && isa<LIT::FuncOp>(*decl))
        if (failed(resolveFully(*decl, decl->getLoc())))
          return failure();
  }

  for (auto fn : traitOp.getBodyRegion().getOps<LIT::FuncOp>()) {
    if (!fn.getBody()->empty())
      shared.emitError(fn.getLoc(),
                       "unexpected function body in trait function "
                       "declaration, use `...` or `pass`");

    auto b = ImplicitLocOpBuilder::atBlockEnd(fn.getLoc(), fn.getBody());
    b.create<TraitFuncOp>();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// UnresolvedImport Decl implementation
//===----------------------------------------------------------------------===//

ParseResult DeclResolver::resolveSignature(LIT::UnresolvedImportOp op,
                                           ASTDecl &decl) {
  PackageOp packageOp = op->getParentOfType<PackageOp>();

  // Grab the location of the import name if present.
  SMLoc importNameLoc =
      shared.diags.convertLocToSMLoc(op.getImportNameLocAttr());
  if (!importNameLoc.isValid())
    importNameLoc = decl.getLoc();

  // Check if we are importing a specific decl within the module, or the
  // module itself.
  if (auto declName = op.getDeclNameAttr()) {
    SMLoc declNameLoc = shared.diags.convertLocToSMLoc(op.getDeclNameLocAttr());
    if (!declNameLoc.isValid())
      declNameLoc = decl.getLoc();

    return getDeclResolver().importDeclFromModule(
        *decl.getParentDecl(), packageOp, op.getModuleNameAttr(), declName,
        op.getImportNameAttr(), decl.getLoc(), declNameLoc, importNameLoc);
  }
  return getDeclResolver().importModule(
      *decl.getParentDecl(), packageOp, op.getModuleNameAttr(),
      op.getImportNameAttr(), decl.getLoc(), importNameLoc);
}
