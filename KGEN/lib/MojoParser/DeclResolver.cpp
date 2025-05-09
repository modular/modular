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
#include "ExprEmitter.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/DocString.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "MojoUtils.h"
#include "ParserBase.h"
#include "Traits.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"

#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/TypeSwitch.h"

#include <deque>

using namespace M;
using namespace KGEN;
using namespace LIT;

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

//===----------------------------------------------------------------------===//
// Decl Constructors

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

ASTDecl &DeclResolver::addBytecodeDecl(Operation *op, StringAttr baseName,
                                       ASTDecl *parentDecl,
                                       DeclResolvedness resolvedness) {
  ASTDecl &decl =
      addDecl(op, shared.diags.convertLocToSMLoc(op->getLoc()), baseName,
              parentDecl, LexerCursor(), LexerCursor(), /*indentation=*/-1);
  decl.loadedFromBytecode = true;
  decl.resolvedness = resolvedness;
  return decl;
}

ASTDecl &DeclResolver::addFullyResolvedDecl(DeclIRValue declVal,
                                            StringAttr name, SMLoc loc,
                                            ASTDecl *parentDecl) {
  auto &decl =
      addDecl(declVal, loc, name, parentDecl, LexerCursor(), LexerCursor(), 0);
  decl.resolvedness = DeclResolvedness::body;
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
  errDecl.setErroneous();
  return errDecl;
}

ASTDecl &DeclResolver::createUnlistedDecl(DeclIRValue irValue, SMLoc loc,
                                          ASTDecl *parentDecl,
                                          LexerCursor cursor,
                                          LexerCursor endCursor,
                                          ssize_t indentation) {
  ASTDecl *decl = shared.allocPersistent<ASTDecl>(
      shared, irValue, loc, parentDecl, cursor, endCursor, indentation);
  parsedDeclList.push_back(decl);

  // If this is a declaration which has a TypeCheckErrorType, then all
  // references to it are invalid.
  if (auto cv = decl->getIfIRValue())
    if (cv.getRValueType().isTypeCheckErrorType())
      decl->setErroneous();

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

  // Lazy allocate declsInScope.
  if (!parentDecl->declsInScope)
    parentDecl->declsInScope.reset(new ASTDecl::DeclInScopeType());

  // Remember the named decl in the symbol table so it can be looked up.
  TinyPtrVector<ASTDecl *> &entries = (*parentDecl->declsInScope)[name];
  if (entries.empty()) {
    entries.push_back(decl);

    // If the decl is a type or alias that has a symbol, remember it.  This
    // allows us to look up decls by symbol when referenced as types. Functions
    // don't have symbols until they are fully resolved, but decls inside
    // functions cannot be accessed anyways.
    if (auto symbolDecl = dyn_cast<mlir::SymbolOpInterface>(decl);
        symbolDecl && !isa<FnOp>(*decl)) {
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
  if (isa<FnOp>(*decl)) {
    // Verify that all previous entries are also functions.  Note that we can't
    // check the overload set is compatible with each other because the
    // signatures aren't all resolved.
    for (ASTDecl *previous : entries) {
      if (!isa<FnOp>(*previous)) {
        auto diag = emitError(decl->getLoc(), "invalid redefinition of ")
                    << name;
        diag.attachNote(previous->getLoc())
            << "cannot overload with this non-function definition";
        decl->setErroneous();
        previous->setErroneous();
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
  decl->setErroneous();
  for (ASTDecl *previous : entries)
    previous->setErroneous();
}

TraitType DeclResolver::getCanonicalTrait(TraitType trait) {
  if (TraitType canonical = traitCanonicalizationCache.lookup(trait))
    return canonical;
  SmallVector<SymbolRefAttr> symbols(trait.getSymbols());
  return traitCanonicalizationCache[trait] = getCanonicalTrait(symbols);
}

TraitType
DeclResolver::getCanonicalTrait(SmallVectorImpl<SymbolRefAttr> &symbols) {
  if (!symbols.empty())
    canonicalizeTraitCompositionSymbols(shared, symbols);
  return TraitType::get(getContext(), symbols);
}

void DeclResolver::attachDeclToTraitCompositionDecl(ASTDecl *traitDecl,
                                                    ASTDecl *childDecl,
                                                    StringAttr name) {
  // Lazy allocate declsInScope.
  if (!traitDecl->declsInScope)
    traitDecl->declsInScope.reset(new ASTDecl::DeclInScopeType());
  (*traitDecl->declsInScope)[name].push_back(childDecl);
}

//===----------------------------------------------------------------------===//
// Import Resolution

void DeclResolver::aliasDecls(ArrayRef<ASTDecl *> decls, StringAttr name,
                              llvm::SMLoc aliasLoc, ASTDecl &context) {
  (void)aliasDeclsImpl(decls, name, aliasLoc, context);
}

LogicalResult DeclResolver::tryAliasDecls(ArrayRef<ASTDecl *> decls,
                                          StringAttr name, llvm::SMLoc aliasLoc,
                                          ASTDecl &context) {
  return aliasDeclsImpl(decls, name, aliasLoc, context,
                        /*emitDiagnostics=*/false);
}

LogicalResult
DeclResolver::aliasImportDecls(ArrayRef<ASTDecl *> decls, StringAttr name,
                               StringAttr declName, StringAttr moduleName,
                               llvm::SMLoc aliasLoc, ASTDecl &context) {
  return aliasDeclsImpl(decls, name, aliasLoc, context,
                        /*emitDiagnostics=*/true, moduleName, declName);
}

LogicalResult
DeclResolver::aliasDeclsImpl(ArrayRef<ASTDecl *> decls, StringAttr name,
                             llvm::SMLoc aliasLoc, ASTDecl &context,
                             bool emitDiagnostics, StringAttr moduleName,
                             StringAttr declNameInModule) {
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
    return success(!importDecl.isErroneous());
  }

  // Lazy allocate declsInScope.
  if (!context.declsInScope)
    context.declsInScope.reset(new ASTDecl::DeclInScopeType());

  auto [it, inserted] =
      context.declsInScope->insert({name, TinyPtrVector<ASTDecl *>(decls)});
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
        decl->resolvedness = DeclResolvedness::body;
      entries = TinyPtrVector<ASTDecl *>(decls);
    }
    return success();
  }
  ASTDecl *existing = it->second.back();

  // If the decls are functions, try to merge them into the existing set.
  if (isa<FnOp>(*frontDecl) && isa<FnOp>(*existing)) {
    // Check that none of the decls are already in the set.
    auto canMergeDecl = [&](ASTDecl *decl) {
      FnOp declOp = cast<FnOp>(decl->getIfOperation());
      return llvm::all_of(entries, [&](ASTDecl *existing) {
        if (failed(resolve(*existing, DeclResolvedness::signature, aliasLoc)))
          return false;
        FnOp existingOp = cast<FnOp>(existing->getIfOperation());

        FnTypeGeneratorType declSignature = declOp.getFullSignature();
        FnTypeGeneratorType existingSignature = existingOp.getFullSignature();
        // If the argument types match exactly *and* the parameter
        // types match exactly, then we don't want to merge this decl into the
        // set. We also need to remove the by-ref result type from the
        // input types, so that aliasing is strictly based on the actual
        // inputs.
        auto getActualArgs =
            [](FnTypeGeneratorType signature) -> ArrayRef<Type> {
          ArrayRef<Type> inputTypes = signature.getArguments();
          // Drop the trailing result slots. Memory-only functions and throwing
          // functions each add a result slot.
          inputTypes = inputTypes.drop_back(signature.hasMemoryOnlyResult() +
                                            signature.isThrows());
          return inputTypes;
        };

        if (getActualArgs(declSignature) == getActualArgs(existingSignature) &&
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
      previous->setErroneous();
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

  return aliasImportDecls(&module, importName,
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

  // Check to see if the module has the construct we are importing.
  auto result = shared.lookupAndResolveDecl(sourceName, sourceNameLoc, module,
                                            /*searchParentScopes=*/false);
  if (result.isErroneous())
    return failure();
  if (result.isFailure()) {
    StringRef name = cast<mlir::SymbolOpInterface>(module).getName();
    StringRef declType = isa<PackageOp>(module) ? "package" : "module";
    emitError(sourceNameLoc, declType + " '" + name + "' does not contain '" +
                                 sourceName.getValue() + "'");
    return failure();
  }
  ArrayRef<ASTDecl *> results = result.getIfSuccess();
  assert(!results.empty() && "other cases handled above");
  shared.notifyListenerOnRef(results, sourceName, sourceNameLoc);
  shared.notifyListenerOnRef(results, destName, destNameLoc);

  return aliasImportDecls(results, destName, sourceName, moduleName,
                          destNameLoc, dest);
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
  if (failed(resolveBody(module, loc)))
    return failure();

  // Resolve pending wildcard imports in this module.
  if (failed(resolveAllWildcardImports(module)))
    return failure();

  // Wildcard imports don't import decls with a leading '_'.
  LogicalResult result = success();
  for (const auto &[name, decls] : module.getDeclsInScope()) {
    // Ignore erroneous children, which have nothing in them.
    if (decls.empty())
      continue;
    if (!isFullImport && name.getValue()[0] == '_')
      continue;
    if (failed(aliasImportDecls(decls, name, name, moduleName, loc, context)))
      result = failure();
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Decl Resolution

LogicalResult DeclResolver::resolve(ASTDecl &decl, DeclResolvedness howResolved,
                                    SMLoc loc) {
  // If decl is already resolved enough, we're done.
  if (decl.resolvedness >= howResolved) {
    // If decl is busted, then return failure.
    return success(!decl.isErroneous());
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
    decl.setErroneous();
    return failure();
  }

  // Handle decls that are loaded from bytecode. These decls are not parsed like
  // decls originating from source files.
  if (decl.loadedFromBytecode) {
    if (failed(shared.resolveDeclFromBytecode(decl, howResolved)))
      decl.setErroneous();

    declsCurrentlyProcessing.erase(&decl);
    return success(!decl.isErroneous());
  }

  // If the signature hasn't been parsed, do so.
  if (decl.resolvedness < DeclResolvedness::signature) {
    // Handle each operation that can be name bound.  We handle this by
    // restoring the lexer to the position where parsing can continue, calling
    // the `resolveSignature` method for the op, and re-saving the new cursor
    // for the next stage of resolution.
    TypeSwitch<ASTDecl &>(decl)
        .Case<FnOp, StructDeclOp, StructFieldOp, TraitDeclOp, GlobalVarDeclOp,
              AliasDeclOp>([&](auto op) {
          Lexer lexer(shared.diags, decl.getCursor());

          // Generate pretty stack traces if a crash happens in this scope.
          LexerCrashReporter crashReporter(lexer, decl.getLoc(),
                                           "resolving decl signature");

          // Resolve the signature: on a parse error, we note that the decl
          // is malformed and should not be referenced to silence downstream
          // errors.
          if (failed(resolveSignature(op, lexer, decl)))
            decl.setErroneous();
          decl.getCursor() = lexer.getCursor();
        })
        .Case<UnresolvedImportOp>([&](auto op) {
          // Resolve the signature: on a parse error, we note that the decl
          // is malformed and should not be referenced to silence downstream
          // errors.
          if (failed(resolveSignature(op, decl)))
            decl.setErroneous();
        })
        .Case<LIT::FileModuleOp, ModuleOp, PackageOp,
              UnresolvedWildcardImportOp>([&](auto op) { /*Nothing*/ })
        .Default([&](auto &attr) {
          if (auto traitType =
                  dyn_cast_or_null<TraitType>(decl.getIfTypeValue())) {
            if (failed(resolveSignature(traitType, decl)))
              decl.setErroneous();
            return;
          }
          // Invalid function arguments will not be resolved to a value and will
          // have a null IR representation.
          if (!decl.isErroneous()) {
            emitError(decl.getLoc(),
                      "do not know how to resolve the signature of this decl!");
            decl.setErroneous();
          }
        });
    // Never regress resolvedness. In the case of non inlined nested functions,
    // the body is fully resolved when the signature is resolved in order
    // to identify the value of 'capturing'
    if (decl.resolvedness != DeclResolvedness::body)
      decl.resolvedness = DeclResolvedness::signature;
  }

  // If the declaration hasn't been fully parsed and we need to, do so.
  if (decl.resolvedness < DeclResolvedness::body &&
      howResolved == DeclResolvedness::body) {
    auto checkEndOfBodyCursor = [&](Lexer &lexer) {
      // If the final parse of the declaration didn't match the initial
      // parse, report an error about unrecognized tokens at end of
      // declaration.
      if (!decl.isMatchingEndCursor(lexer.getCursor()) && !decl.isErroneous()) {
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
    decl.resolvedness = DeclResolvedness::body;

    // If the decl is already erroneous, trying to process further may crash or
    // cause spurious error messages.
    if (decl.isErroneous())
      return failure();

    // Handle each operation that can be name bound.
    TypeSwitch<ASTDecl &>(decl)
        .Case<FileModuleOp, FnOp, StructDeclOp, StructFieldOp, TraitDeclOp,
              GlobalVarDeclOp, AliasDeclOp>([&](auto op) {
          // Parse the body of the declaration from the correct point.
          Lexer lexer(shared.diags, decl.getCursor());

          // Generate pretty stack traces if a crash happens in this scope.
          LexerCrashReporter crashReporter(lexer, decl.getLoc(),
                                           "resolving decl body");

          if (resolveBody(op, lexer, decl))
            return;

          checkEndOfBodyCursor(lexer);
        })
        .Case<ConformanceOp>([&](auto op) {
          if (failed(resolveBody(op, decl)))
            decl.setErroneous();
        })
        .Case<PackageOp>([&](auto op) { (void)resolveBody(op, decl); })
        .Case<ModuleOp, UnresolvedImportOp, UnresolvedWildcardImportOp>(
            [&](auto op) { /*Nothing*/ })
        .Default([&](auto &attr) {
          if (auto traitType =
                  dyn_cast_or_null<TraitType>(decl.getIfTypeValue())) {
            if (failed(resolveBody(traitType, decl)))
              decl.setErroneous();
            return;
          }

          if (!decl.isErroneous())
            emitError(decl.getLoc(),
                      "do not know how to resolve the body of this decl!");
        });
  }

  declsCurrentlyProcessing.erase(&decl);
  // If decl is busted, then return failure.
  return success(!decl.isErroneous());
}

//===----------------------------------------------------------------------===//
// Top-Level Decl Resolution

void DeclResolver::resolveAllReferencedFrom(ASTDecl &decl,
                                            bool eraseUnparsedDecls) {
  CompilerTimeTraceScope traceScope("resolveAllReferencedFrom", [&] {
    return decl.getNameIfOperation().value_or("").str();
  });

  // The first stage is to fully resolve all of the decls recursively defined
  // within the main container. These decls provide the anchor for resolution.
  std::deque<ASTDecl *> worklist({&decl});
  while (!worklist.empty()) {
    ASTDecl *declIt = worklist.back();
    worklist.pop_back();

    // Resolve the decl.
    (void)resolveBody(*declIt, declIt->getLoc());

    // When validating doc strings, we wish to only validate those defined on
    // decl in the main container. As this point the main container decl has
    // been fully resolved, so it's an opportune time to validate.
    validateDocString(*declIt);

    // If this is a package, resolve all of the modules within it as a pre-step.
    // Normally these get lazily resolved, but if we're forcing pulling them in,
    // we need to do it now.
    if (isa<PackageOp>(*declIt)) {
      for (auto &[_, decls] : declIt->getDeclsInScope())
        if (isa<UnresolvedImportOp>(*decls.front()))
          (void)resolveBody(*decls.front(), declIt->getLoc());
    }

    // Traverse the children. We don't resolve alias children, these will be
    // resolved separately if they actually got referenced.
    for (auto &[_, decls] : declIt->getDeclsInScope()) {
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
      if (decl.resolvedness == DeclResolvedness::unparsed) {
        // Some decls always need to be resolved if their parents were resolved,
        // allowlist the decls that we can safely ignore when unparsed.
        if (isa<FnOp, FileModuleOp, PackageOp, UnresolvedImportOp,
                UnresolvedWildcardImportOp, StructDeclOp, TraitDeclOp,
                AliasDeclOp, GlobalVarDeclOp>(decl)) {
          deferredDecls.insert(&decl);
          continue;
        }
      }

      (void)resolveBody(decl, decl.getLoc());
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
          (void)resolveBody(*decl, decl->getLoc());
          resolvedAnything = true;
        }
      }
    } while (resolvedAnything);
  } while (parsedDeclIt != parsedDeclList.size());

  // Erase unresolved operations from source.
  if (eraseUnparsedDecls) {
    for (ASTDecl *decl : parsedDeclList) {
      if (decl->resolvedness == DeclResolvedness::unparsed &&
          !decl->loadedFromBytecode)
        if (Operation *op = decl->getIfOperation()) {
          if (!isa<UnresolvedImportOp>(op)) {
            op->erase();
            decl->setIRValue(nullptr);
          }
        }
    }
  }
}

LogicalResult DeclResolver::resolveAllWildcardImports(ASTDecl &module) {
  if (!module.unresolvedWildcardImports)
    return success();

  while (!module.unresolvedWildcardImports->empty()) {
    auto it = module.unresolvedWildcardImports->begin();
    auto [moduleName, locAndIsFullImport] = *it;
    module.unresolvedWildcardImports->erase(it);

    if (failed(importWildCardDeclsFromModule(module, moduleName,
                                             locAndIsFullImport.second,
                                             locAndIsFullImport.first)))
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Symbol-ASTDecl Mapping

ASTDecl &DeclResolver::getDeclForTypeSymbol(SymbolRefAttr symbol) const {
  auto it = declForTypeSymbol.find(symbol);
#ifndef NDEBUG
  if (it == declForTypeSymbol.end())
    symbol.dump();
  assert(it != declForTypeSymbol.end() && "Unknown decl symbol!");
#endif
  return *it->second;
}

ASTDecl *DeclResolver::getDeclForFuncSymbol(SymbolRefAttr attr) const {
  auto it = declForFuncSymbol.find(attr);
  return it != declForFuncSymbol.end() ? it->second : nullptr;
}

Operation *DeclResolver::finalizeFuncSignature(FnOp funcOp, ASTDecl &decl) {
  // Remember the mapping from its fully mangled symbol so we can find its AST
  // representation and body from IR references.
  declForFuncSymbol[getFullyResolvedSymbolRef(funcOp)] = &decl;

  // Install it in the symbol table and check for redefinition while doing so.
  return shared.setResolvedDeclSymbol(funcOp);
}

ASTDecl *DeclResolver::getTraitDecl(TraitType trait) {
  ArrayRef<SymbolRefAttr> symbols = trait.getSymbols();
  if (symbols.size() == 1)
    return &getDeclForTypeSymbol(symbols.front());

  TraitType canonTraitType = getCanonicalTrait(trait);
  // Check if the canonicalized trait type has a hit.
  if (auto it = canonicalTraitCompositionDecls.find(canonTraitType);
      it != canonicalTraitCompositionDecls.end())
    return it->second;

  // Otherwise, create a new decl and register for the canonical trait type.
  // Trait compositions are anonymous declarations and do not have a source
  // location themselves. Conformance errors will be routed to its member decls.
  ASTDecl *decl = &createUnlistedDecl(DeclIRValue(canonTraitType), /*loc=*/{},
                                      /*parentDecl=*/nullptr, LexerCursor(),
                                      LexerCursor(), /*indentation=*/-1);

  // Initialize the decl to signature-resolved since we do not have anything to
  // do for the signature resolve phase.
  decl->resolvedness = DeclResolvedness::signature;
  canonicalTraitCompositionDecls[canonTraitType] = decl;
  return decl;
}

//===----------------------------------------------------------------------===//
// Export Handling

void DeclResolver::registerAndCheckExport(StringRef aliasName, SMLoc loc) {
  auto [it, inserted] = exportedSymbolNames.try_emplace(aliasName, loc);
  if (!inserted) {
    auto diag = emitError(loc, "invalid re-export of ") << aliasName;
    diag.attachNote(it->second) << "previous export here";
    return;
  }
}

void DeclResolver::exportMain(ASTDecl &funcDecl) {
  FnOp userMainFn = cast<FnOp>(funcDecl);
  FnTypeGeneratorType userMainSignature = userMainFn.getFuncTypeGenerator();
  ASTDecl *containingDecl = funcDecl.getParentDecl();
  SMLoc loc = funcDecl.getLoc();

  // The type of main function described by the given func decl.
  enum MainKind {
    // A non-raising function that returns None.
    kNonRaisingNoneMain,
    // A raising function that returns None.
    kRaisingNoneMain,
  };
  MainKind mainKind = kNonRaisingNoneMain;

  // Validate that main has the expected signature.
  if (!userMainSignature.getInputParamTypes().empty()) {
    shared.emitError(loc, "expected 'main' function to have no parameters");
    return;
  }
  ASTType userResultType(userMainFn.getUserResultType());
  ArrayRef<Type> argTypes = userMainSignature.getArguments();

  // Process a main returning none.
  if (userResultType.isNoneType()) {
    if (userMainSignature.isThrows()) {
      mainKind = kRaisingNoneMain;
      // Drop the error from the argument list.
      argTypes = argTypes.drop_front(2);
    }

    // Process a main returning object.
  } else {
    shared.emitError(loc, "expected 'main' function to return 'None'");
    return;
  }
  if (!argTypes.empty()) {
    shared.emitError(loc, "expected 'main' function to have no arguments");
    return;
  }

  // Validate that we aren't in a package, defining a `main` within a package
  // is not fully supported.
  if (userMainFn->getParentOfType<PackageOp>()) {
    shared.emitError(loc,
                     "defining 'main' within a package is not yet supported");
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
    if (failed(resolveBody(*decl, decl->getLoc())))
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
  FnOp mainShimProtoFn = cast<FnOp>(*mainShimProtoDecl);

  // Builder function.
  StringAttr mainAttr = StringAttr::get(getContext(), "main");
  auto shimMainFn = cast<FnOp>(builder.clone(*mainShimProtoFn));
  shimMainFn.setSymNameAttr(mainAttr);
  shimMainFn.setLinkageNameAttr(mainAttr);
  shimMainFn.setCExported();
  shimMainFn.getBody()->clear();

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
  }
  ASTDecl *mainWrapperDecl = resolveStartDecl(mainWrapperName);
  if (!mainWrapperDecl)
    return;
  FnOp mainWrapperFn = cast<FnOp>(*mainWrapperDecl);
  FnTypeGeneratorType mainWrapperSigGen = mainWrapperFn.getFuncTypeGenerator();

  // Generate a reference to the main wrapper function, which expects the user
  // main to be provided via an parameter.
  FnType mainWrapperSig = mainWrapperSigGen.getBody();
  FnMetadataAttr mainWrapperFnMeta = mainWrapperSig.getMetadata();
  auto strippedMainWrapperFnMeta = FnMetadataAttr::get(
      mainWrapperFnMeta.getArgListAttrs(),
      mainWrapperFnMeta.getNumImplicitOriginDecls(),
      mainWrapperFnMeta.getCaptureOrigins(),
      mainWrapperFnMeta.getIsNestedOriginExclusivityCheckingDisabled());
  auto strippedMainWrapperSig =
      FuncType::get(getContext(), mainWrapperSig.getValues(),
                    mainWrapperSig.getArgConventions(),
                    mainWrapperSig.getFnEffects(), strippedMainWrapperFnMeta);
  SymbolConstantAttr wrapperFnRef = SymbolConstantAttr::get(
      getFullyResolvedSymbolRef(mainWrapperFn),
      GeneratorType::get(/*inputParamTypes=*/{}, strippedMainWrapperSig,
                         /*metadata=*/PogListAttr::get(getContext())),
      {SymbolConstantAttr::get(getFullyResolvedSymbolRef(userMainFn),
                               userMainSignature)});

  auto shimBodyBuilder = ImplicitLocOpBuilder::atBlockBegin(
      shimMainFn->getLoc(), shimMainFn.getBody());
  auto wrappedCall = shimBodyBuilder.create<CallOp>(
      shimMainFn.getArgumentTypes()[0], wrapperFnRef,
      /*originParams=*/std::nullopt, shimMainFn.getArguments());
  ExprEmitter::emitNormalReturn(shimBodyBuilder, wrappedCall.getResult(0));

  exportedSymbolNames.insert({mainAttr, funcDecl.getLoc()});
}

//===----------------------------------------------------------------------===//
// Decl Helpers

StringAttr DeclResolver::getMangledName(StringAttr baseName, ASTDecl &container,
                                        FnTypeGeneratorType signatureGen) {
  // Compute the full signature of the decl to ensure dependent parameters from
  // a parent decl are name-erased in the mangled name.
  FnTypeGeneratorType fullSig =
      LIT::getFullSignature(container.getIfOperation(), signatureGen);

  SmallString<64> mangledName(baseName.getValue().begin(),
                              baseName.getValue().end());
  llvm::raw_svector_ostream os(mangledName);
  // Don't include parent parameters in the mangling.
  ArrayRef<Type> params = fullSig.getInputParamTypes().take_back(
      signatureGen.getInputParamTypes().size());
  if (!params.empty()) {
    size_t numSkipped = fullSig.getInputParamTypes().size() - params.size();
    os << '[';
    llvm::interleave(
        llvm::enumerate(params), os,
        [&](auto typeAndIdx) {
          auto [idx, implType] = typeAndIdx;
          ASTType type = implType;
          if (fullSig.getMetadata().isPosVarArg(idx + numSkipped)) {
            os << "*";
            type = type.getVariadicElementType();
          }
          os << type.getAsString(/*diags=*/nullptr, /*demangleParams=*/true);
        },
        ",");
    os << ']';
  }

  mangledName += '(';
  for (auto [argNo, conventionX, argTypeX] :
       llvm::enumerate(fullSig.getArgConventions(), fullSig.getArguments())) {
    auto convention = conventionX;
    ASTType argType = argTypeX;

    // We do not mangle results into the signature.
    if (isResultSlot(convention))
      continue;

    // Update the mangled name for this argument.
    if (argNo != 0)
      mangledName += ",";

    // If this had adjustments added to it because of its argument convention /
    // variadic state, strip them off.
    unsigned numStars = 0;
    if (fullSig.isPosVarArg(argNo)) {
      auto variadic = cast<VariadicType>(argType);
      argType = variadic.getElementType();
      convention = variadic.getConvention();
      numStars = 1;
    } else if (ASTType variadicPack = fullSig.getIfVariadicPack(argNo)) {
      TypedAttr packVariadic = variadicPack.getVariadicPackTypeList();
      mangledName += '*';
      ASTType::printParam(os, packVariadic, /*diags=*/nullptr,
                          /*demangleParams=*/true);
      continue;
    } else if (fullSig.isKwVarArg(argNo)) {
      // TODO: Propagate convention correctly.
      convention = ArgConvention::ReadReg;
      argType = argType.getKwargsDictRefValueType();
      numStars = 2;
    }

    if (hasAddress(convention))
      argType = argType.getReferenceElementType();

    mangledName +=
        argType.getAsString(/*forDiag=*/nullptr, /*demangleParams=*/true);

    // Add suffix to disambiguate overloadable conventions.
    switch (convention) {
    case ArgConvention::OwnedReg:
      llvm_unreachable("not used by the parser");
    case ArgConvention::OwnedMem:
    case ArgConvention::ReadReg:
    case ArgConvention::ReadMem:
      break;
    case ArgConvention::Mut:
      mangledName += '&';
      break;
    case ArgConvention::Ref:
    case ArgConvention::MutRef:
      mangledName += '%';
      break;
    case ArgConvention::ByRefResult:
    case ArgConvention::ByRefError:
      llvm_unreachable("byref_result should be skipped");
    }

    while (numStars--)
      mangledName += '*';
  }
  mangledName += ')';

  // Having "@" in mangled names confuses gnu ld and triggers error at linking
  // stage. See issue #6918. So replacing "@" with "_".
  std::replace(mangledName.begin(), mangledName.end(), '@', '_');
  return StringAttr::get(baseName.getContext(), mangledName);
}
