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
#include "IREmitter.h"
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
// DiagnosticDeclContextChanger
//===----------------------------------------------------------------------===//

DeclResolver::DiagnosticDeclContextChanger::DiagnosticDeclContextChanger(
    ASTDecl *declToUse) {
  if (!declToUse)
    return;
  auto &shared = declToUse->getShared();
  resolver = &*shared.declResolver;
  prevDiagnosticDeclContext = resolver->diagnosticDeclContext;
  resolver->diagnosticDeclContext = declToUse;
}
DeclResolver::DiagnosticDeclContextChanger::~DiagnosticDeclContextChanger() {
  if (!resolver)
    return;
  resolver->diagnosticDeclContext = prevDiagnosticDeclContext;
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

void DeclResolver::registerStructGeneratorDecl(StructGeneratorOp structGen,
                                               SymbolRefAttr symbol, SMLoc loc,
                                               ASTDecl &parentDecl) {
  // Create an unlisted decl for the struct generator. It's unlisted because
  // we don't want it in the parent's name table (it's not a normal named decl),
  // but we do want it in declForTypeSymbol for lookup.
  ASTDecl &decl = createUnlistedDecl(structGen.getOperation(), loc, &parentDecl,
                                     LexerCursor(), LexerCursor(), -1);
  decl.resolvedness = DeclResolvedness::body;
  declForTypeSymbol[symbol] = &decl;
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

// Check whether we're merging in a bunch of ASTDecls that could all contribute
// to a single struct's namespace.
// For example, say we have these imports:
//     from module_a import MyStruct  # imports a struct
//     from module_b import MyStruct  # imports an extension for it
//     from module_c import MyStruct  # imports an extension for it
// and the first two are resolved. That means that these three entries all
// coexist under the name "MyStruct":
// - struct ASTDecl for module_a's MyStruct struct
// - extension ASTDecl for module_b's MyStruct extension
// - unresolved import for module_c's MyStruct extension
// Basically, any entries for things that might contribute to a single
// struct's namespace is allowed to coexist.
// TODO(MOCO-522): Arcana doc mention on how multiple extensions and one
// struct and multiple imports can all coexist with the same name, because
// struct extensions are importable via their target struct's name.
static LogicalResult
canMergeSingleNamespaceDecls(ArrayRef<ASTDecl *> incoming,
                             ArrayRef<ASTDecl *> existing) {
  // Check if all declarations (both incoming and existing) could contribute
  // to a single struct's namespace.
  bool structFound = false;

  for (ASTDecl *decl : llvm::concat<ASTDecl *const>(existing, incoming)) {
    auto op = decl->getIfOperation();
    bool couldContribute = isa_and_nonnull<StructDeclOp>(op) ||
                           isa_and_nonnull<ExtensionDeclOp>(op) ||
                           isa_and_nonnull<UnresolvedImportOp>(op);
    if (!couldContribute)
      return failure();
    if (isa_and_nonnull<StructDeclOp>(op)) {
      if (structFound) {
        // User is trying to add a second struct with the same name, fail.
        return failure();
      }
      structFound = true;
    }
  }

  return success();
}

void DeclResolver::attachDeclToParentNameTable(ASTDecl *decl, StringAttr name) {
  ASTDecl *parentDecl = decl->getParentDecl();

  // Lazy allocate declsInScope.
  if (!parentDecl->declsInScope)
    parentDecl->declsInScope.reset(new ASTDecl::DeclInScopeType());

  // Remember the named decl in the symbol table so it can be looked up.
  TinyPtrVector<ASTDecl *> &entries = (*parentDecl->declsInScope)[name];

  // Function support method overloading on input arguments.  Variables and
  // types cannot be overloaded because they have no inputs.  Well, we could
  // actually allow type overloading on parameters theoretically to support
  // T[4] and T[1,7] as different things, but let's no proactively add
  // complexity.
  if (isa_and_nonnull<FnOp>(decl->getIfOperation())) {
    // Verify that all previous entries are also functions.  Note that we can't
    // check the overload set is compatible with each other because the
    // signatures aren't all resolved.
    for (ASTDecl *previous : entries) {
      if (!isa_and_nonnull<FnOp>(previous->getIfOperation())) {
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
    // We don't uniquifyNameAndAddToParentSymbolTable here, that's done
    // elsewhere for functions.
    return;
  }

  // Structs and extensions can both have the same name in the same scope.
  // TODO(MOCO-522): Reference some arcana docs on this
  bool addingStruct = isa_and_nonnull<StructDeclOp>(decl->getIfOperation());
  bool addingExtension =
      isa_and_nonnull<ExtensionDeclOp>(decl->getIfOperation());
  if (addingStruct || addingExtension) {
    // Verify that all previous entries are also structs or extensions.  Note
    // that we can't check the overload set is compatible with each other
    // because the signatures aren't all resolved.
    for (ASTDecl *previous : entries) {
      bool previousIsStruct =
          isa_and_nonnull<StructDeclOp>(previous->getIfOperation());
      bool previousIsExtension =
          isa_and_nonnull<ExtensionDeclOp>(previous->getIfOperation());
      bool previousIsImport =
          isa_and_nonnull<UnresolvedImportOp>(previous->getIfOperation());
      bool previousNotStructRelated =
          !previousIsStruct && !previousIsExtension && !previousIsImport;
      // This checks that we're not giving e.g. a function and a struct the same
      // name.
      if (previousNotStructRelated) {
        auto diag = emitError(decl->getLoc(), "cannot define ")
                    << (addingStruct ? "a struct" : "an extension")
                    << " here with name " << name;
        diag.attachNote(previous->getLoc())
            << "conflicts with this previous declaration";
        decl->setErroneous();
        previous->setErroneous();
        return;
      }
      // Check for import vs local struct conflicts
      // An imported declaration cannot coexist with a locally defined struct
      // because the import cannot be an extension of a struct we're currently
      // defining. Search "#12090" for an example.
      // TODO(MOCO-522): This deserves an arcana doc and a few references to it.
      if (addingStruct && previousIsImport) {
        auto diag =
            emitError(decl->getLoc(), "cannot define a struct here with name ")
            << name;
        diag.attachNote(previous->getLoc())
            << "conflicts with this previous declaration";
        decl->setErroneous();
        previous->setErroneous();
        return;
      }
      // This makes sure we're not adding two structs with the same name.
      if (addingStruct && previousIsStruct) {
        auto diag = emitError(decl->getLoc(), "invalid redefinition of ")
                    << name;
        diag.attachNote(previous->getLoc())
            << "conflicts with this previous struct declaration";
        decl->setErroneous();
        previous->setErroneous();
        return;
      }
    }

    // Otherwise, we're good, charge forwards.
    entries.push_back(decl);

    assert(dyn_cast_or_null<mlir::SymbolOpInterface>(decl->getIfOperation()));
    registerDeclSymbol(decl);
    return;
  }

  // For any other type of declaration, check for conflicts
  if (!entries.empty()) {
    // Check if we are adding an identical unresolved import.
    auto op = decl->getIfOperation();
    if (auto import = dyn_cast_or_null<UnresolvedImportOp>(op)) {
      // First check for duplicate imports
      for (ASTDecl *existing : entries) {
        if (auto prevImportOp = dyn_cast_or_null<UnresolvedImportOp>(
                existing->getIfOperation())) {
          if (import.getModuleNameAttr() == prevImportOp.getModuleNameAttr() &&
              import.getDeclNameAttr() == prevImportOp.getDeclNameAttr()) {
            // This is a duplicate UnresolvedImportOp, just ignore it.
            return;
          }
        }
      }
      // TODO(MOCO-522): Arcana docs mention for decls sharing a namespace.
      if (succeeded(canMergeSingleNamespaceDecls({decl}, entries))) {
        entries.push_back(decl);
        return;
      }
    }

    // This is a genuine redefinition error
    ASTDecl *existing = entries.back();
    auto diag = emitError(decl->getLoc(), "invalid redefinition of ") << name;
    diag.attachNote(existing->getLoc()) << "previous definition here";

    // Mark the existing decl and this one as erroneous so uses of either
    // don't create confusing errors.
    decl->setErroneous();
    for (ASTDecl *previous : entries)
      previous->setErroneous();
    return;
  }

  // This is the first declaration with this name
  entries.push_back(decl);

  // Register symbol with the parent symbol table.
  // Functions don't have symbols until they are fully resolved, but decls
  // inside functions cannot be accessed anyways.
  registerDeclSymbol(decl);
}

void DeclResolver::registerDeclSymbol(ASTDecl *decl) {
  if (auto symbolDecl =
          dyn_cast_or_null<mlir::SymbolOpInterface>(decl->getIfOperation())) {
    shared.uniquifyNameAndAddToParentSymbolTable(symbolDecl.getOperation());
    // This symbol may have been renamed by the above
    // uniquifyNameAndAddToParentSymbolTable call.
    SymbolRefAttr symbol = decl->getSymbolRef();
    // This shouldn't trip because we uniqued it in the above
    // uniquifyNameAndAddToParentSymbolTable call.
    assert(!declForTypeSymbol.count(symbol) && "Symbol redefinition/collision");
    declForTypeSymbol[symbol] = decl;
  }
}

void DeclResolver::aliasDeclInParent(ASTDecl *decl, StringAttr aliasName) {
  ASTDecl *parentDecl = decl->getParentDecl();

  // Lazy allocate declsInScope.
  if (!parentDecl->declsInScope)
    parentDecl->declsInScope.reset(new ASTDecl::DeclInScopeType());

  // Add the decl to the parent's name table under the name aliasName.
  TinyPtrVector<ASTDecl *> &entries = (*parentDecl->declsInScope)[aliasName];

  // TODO(MOCO-522): Linear, seems expensive, maybe we can change declsInScope
  // to something like a linked hash map?
  if (!llvm::is_contained(entries, decl))
    entries.push_back(decl);

  // Note: We intentionally do NOT call uniquifyNameAndAddToParentSymbolTable
  // because the extension is already in the symbol table under its primary name
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
                               llvm::SMLoc aliasLoc, ASTDecl &context,
                               bool allowMultipleWithSameName) {
  return aliasDeclsImpl(decls, name, aliasLoc, context,
                        /*emitDiagnostics=*/true, moduleName, declName,
                        allowMultipleWithSameName);
}

// Check whether the incoming decls conflict with existing decls under the same
// name, applying the same rules as `attachDeclToParentNameTable` does for local
// declarations: functions may freely overload each other, but a function and a
// non-function (struct, alias, MLIR type, …) under the same name is an error,
// as are two distinct non-functions.
LogicalResult DeclResolver::checkImportNamingConflict(
    ArrayRef<ASTDecl *> incoming, ArrayRef<ASTDecl *> existing, StringAttr name,
    llvm::SMLoc aliasLoc, bool emitDiagnostics) {
  // Single pass over each set to find a representative def and non-def decl
  // (skipping UnresolvedImportOps whose type is not yet known).
  //
  // By module naming rules, each set has at most one non-function element
  // (a module cannot declare e.g. both a struct and an alias under the same
  // name). This lets us classify both sets in O(N+M) and dispatch directly,
  // avoiding an O(N*M) nested loop over two potentially large overload sets.
  struct DeclKinds {
    ASTDecl *fn = nullptr, *nonFn = nullptr;
  };
  auto classify = [](ArrayRef<ASTDecl *> decls,
                     ASTDecl *skip = nullptr) -> DeclKinds {
    DeclKinds result;
    for (ASTDecl *d : decls) {
      if (isa_and_nonnull<UnresolvedImportOp>(d->getIfOperation()))
        continue;
      if (d == skip)
        continue;
      if (isa_and_nonnull<FnOp>(d->getIfOperation())) {
        if (!result
                 .fn) // any representative def suffices for conflict detection
          result.fn = d;
      } else {
        result.nonFn = d; // at most one non-def per set (module naming rules)
      }
    }
    return result;
  };

  auto [incomingFn, incomingNonFn] = classify(incoming);
  // Skip incomingNonFn when scanning the existing set: if the user wrote
  // `from mod_a import Foo` twice, the first resolution already placed the
  // struct into `existing`, so the same ASTDecl* appears in both arrays.
  // Without the skip, we would compare the decl against itself and
  // incorrectly diagnose a "struct vs. struct" conflict.
  auto [existingFn, existingNonFn] = classify(existing, incomingNonFn);

  // Determine the conflicting pair, if any.
  ASTDecl *conflictA = nullptr, *conflictB = nullptr;
  if (incomingNonFn && existingFn) {
    // Non-function being imported conflicts with an existing function.
    conflictA = existingFn;
    conflictB = incomingNonFn;
  } else if (incomingFn && existingNonFn) {
    // Function being imported conflicts with an existing non-function.
    conflictA = existingNonFn;
    conflictB = incomingFn;
  } else if (incomingNonFn && existingNonFn) {
    // Two non-functions: compatible only if they form a single struct namespace
    // (one struct + its extensions). Two structs, two aliases, etc. conflict.
    if (failed(
            canMergeSingleNamespaceDecls({incomingNonFn}, {existingNonFn}))) {
      conflictA = existingNonFn;
      conflictB = incomingNonFn;
    }
  }

  if (!conflictA)
    return success();

  if (emitDiagnostics) {
    auto diag = emitError(aliasLoc, "import of ") << name << " is ambiguous";
    diag.attachNote(conflictA->getLoc()) << name << " declared here";
    diag.attachNote(conflictB->getLoc()) << name << " also declared here";
  }
  return failure();
}

LogicalResult DeclResolver::aliasDeclsImpl(
    ArrayRef<ASTDecl *> decls, StringAttr name, llvm::SMLoc aliasLoc,
    ASTDecl &context, bool emitDiagnostics, StringAttr moduleName,
    StringAttr declNameInModule, bool allowMultipleWithSameName) {
  // Check to see if the decl is an import. We create new decls within the
  // context for these instead of aliasing, because import decls lazily replace
  // themselves with new decls (depending on what gets imported). That
  // replacement is only known when the import decl is referenced (and thus
  // resolved), so we can't alias the import directly.
  ASTDecl *frontDecl = decls.front();
  if (auto importOp =
          dyn_cast_or_null<UnresolvedImportOp>(frontDecl->getIfOperation())) {
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
  // It succeeded and there was nothing in this scope by that name already, so
  // we're done.
  if (inserted)
    return success();
  // If we got here, it failed, there's already entries here by that name.
  TinyPtrVector<ASTDecl *> &entries = it->second;

  // If we get here, then we've hit an overlap. This is likely because we're
  // seeing the import statement that's already here, and it's conflicting with
  // the new entries we're bringing in.
  // Check to see if that's the case, and if so, replace the unresolved import
  // with the real decls from the target module.
  //
  // The `moduleName` argument tells us which module we're importing from, and
  // is only present when we're resolving an import (not just creating an
  // alias).
  // Here, we look for that import.
  // TODO(MOCO-522): This seems weird. This function shouldn't be making
  // assumptions about what moduleName's existence means. Possibly rename
  // moduleName or find some better way to represent this, or last resort, make
  // it some arcana.
  if (moduleName) {
    // Find and remove all matching imports (in case of duplicate imports).
    // Keep in mind, the user may have imported the module twice, so we have to
    // remove all matching imports (see test MSWGHRI).
    bool foundMatchingImport = false;
    for (int i = entries.size() - 1; i >= 0; --i) {
      if (auto importOp = dyn_cast_or_null<UnresolvedImportOp>(
              entries[i]->getIfOperation());
          importOp && importOp.getModuleNameAttr() == moduleName &&
          importOp.getDeclNameAttr() == declNameInModule) {
        // Mark the import we're replacing as resolved in case anyone sees it
        // (which would be weird, since we're about to remove it, but just in
        // case).
        entries[i]->resolvedness = DeclResolvedness::body;
        // Remove this matching import. We'll replace it with the real decls
        // further below.
        entries.erase(entries.begin() + i);
        foundMatchingImport = true;
      }
    }

    // TODO(MOCO-522): It feels like this function is doing a few too many
    // things in too many odd cases, should split and revisit this abstraction.
    bool shouldAdd = false;
    if (foundMatchingImport) {
      // Sure enough, we found an importOp that matches the module and decl
      // name, let's replace the import with the real decls.
      if (failed(checkImportNamingConflict(decls, entries, name, aliasLoc,
                                           emitDiagnostics)))
        return failure();
      shouldAdd = true;
    } else {
      // No placeholder was removed, this can happen if someone is calling
      // aliasDeclsImpl for things that were already imported, for example if
      // we're importing a bunch of extensions when we've already imported them
      // in the past.
      // Now, check if the new decls can coexist with the existing ones.
      if (allowMultipleWithSameName)
        if (succeeded(canMergeSingleNamespaceDecls(decls, entries)))
          shouldAdd = true;
    }
    if (shouldAdd) {
      // Add new decls, avoiding duplicates.
      // TODO(MOCO-522): Quadratic loop, maybe we can change declsInScope to
      // something like a linked hash map?
      for (ASTDecl *decl : decls)
        if (!llvm::is_contained(entries, decl))
          entries.push_back(decl);
    }

    return success();
  }

  // TODO(MOCO-522): Arcana docs mention for decls sharing a namespace.
  if (succeeded(canMergeSingleNamespaceDecls(decls, entries))) {
    // Add new decls, avoiding duplicates.
    // TODO(MOCO-522): Quadratic loop, maybe we can change declsInScope to
    // something like a linked hash map?
    for (ASTDecl *decl : decls)
      if (!llvm::is_contained(entries, decl))
        entries.push_back(decl);
    return success();
  }

  ASTDecl *existing = entries.back();

  // If the decls are functions, try to merge them into the existing set.
  if (isa_and_nonnull<FnOp>(frontDecl->getIfOperation()) &&
      isa_and_nonnull<FnOp>(existing->getIfOperation())) {
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
      // We don't have to check for duplicates here because canMergeDecl
      // already detects duplicates.
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

ASTDecl &DeclResolver::createImportOp(ASTDecl &dest, mlir::OpBuilder &builder,
                                      StringAttr name,
                                      StringAttr realModuleName,
                                      mlir::Location loc, bool allowAll) {
  auto importOp = ImportOp::create(builder, loc,
                                   /*sym_name=*/name, realModuleName, allowAll);
  SMLoc smloc = shared.diags.convertLocToSMLoc(loc);
  ASTDecl &importDecl =
      addDecl(static_cast<Operation *>(importOp), smloc, name, &dest,
              LexerCursor(), LexerCursor(), /*indentation=*/-1);
  // ImportOp has no body to parse — mark as fully resolved so that name
  // lookup through parent scopes doesn't trip the resolvedness assertion.
  importDecl.resolvedness = DeclResolvedness::body;
  return importDecl;
}

LogicalResult DeclResolver::importModule(ASTDecl &dest, UnresolvedImportOp op,
                                         PackageOp currentPackage, SMLoc loc,
                                         SMLoc importNameLoc) {
  StringAttr moduleName = op.getModuleNameAttr();
  StringAttr importName = op.getImportNameAttr();
  ASTDecl &module = shared.importModule(moduleName, currentPackage, loc);
  shared.notifyListenerOnModuleImport(module, moduleName, loc);
  shared.notifyListenerOnRef(&module, importName, importNameLoc);

  if (failed(aliasImportDecls(&module, importName,
                              /*declName=*/StringAttr(), moduleName,
                              importNameLoc, dest, false)))
    return failure();

  // For dotted imports without an alias (e.g. "import pkg.a.b1"), build/merge
  // a tree of ImportOps: ImportOp("pkg") → ImportOp("a") → ImportOp("b1",
  // allowAll). This gates access at every level of the dotted path.
  StringRef moduleStr = moduleName.getValue();
  if (!moduleStr.contains('.') || moduleName != importName)
    return success();

  SmallVector<StringRef> segments;
  moduleStr.split(segments, '.');

  ASTDecl *currentScope = &dest;
  OpBuilder importOpBuilder = OpBuilder(op);
  std::string qualifiedName;

  for (size_t i = 0, e = segments.size(); i != e; ++i) {
    if (i > 0)
      qualifiedName += '.';
    qualifiedName += segments[i];
    bool isLeaf = (i == e - 1);
    StringAttr segName = StringAttr::get(getContext(), segments[i]);
    StringAttr qualName = StringAttr::get(getContext(), qualifiedName);

    // Look for existing ImportOp at this level.
    ASTDecl *existingDecl = nullptr;
    ImportOp existingImport;
    ArrayRef<ASTDecl *> existing = currentScope->lookupInCurrentScope(segName);
    for (ASTDecl *d : existing) {
      if (auto imp = dyn_cast_or_null<ImportOp>(d->getIfOperation())) {
        existingImport = imp;
        existingDecl = d;
        break;
      }
    }

    if (existingImport) {
      if (existingImport.getAllowAll()) {
        // Already unrestricted at this level — nothing more to do.
        return success();
      }
      if (isLeaf) {
        // Mark leaf as unrestricted.
        existingImport.setAllowAllAttr(UnitAttr::get(getContext()));
        return success();
      }
      // Navigate into existing ImportOp for next segment.
      currentScope = existingDecl;
      importOpBuilder = existingDecl->getDeclEndBuilder();
    } else {
      // Create new ImportOp at this level.  Use the UnresolvedImportOp's
      // location so that the ImportOp inherits the correct debug scope
      // (e.g. the enclosing function's subprogram scope).
      ASTDecl &newImport =
          createImportOp(*currentScope, importOpBuilder, segName, qualName,
                         op->getLoc(), /*allowAll=*/isLeaf);
      if (isLeaf)
        return success();
      currentScope = &newImport;
      importOpBuilder = newImport.getDeclEndBuilder();
    }
  }

  return success();
}

FailureOr<ASTDecl *> DeclResolver::bodyResolvePackageInit(ASTDecl &module,
                                                          SMLoc loc) {
  // Not a package — nothing to do.
  if (!isa_and_nonnull<PackageOp>(module.getIfOperation()))
    return FailureOr<ASTDecl *>(nullptr);
  StringAttr initName = StringAttr::get(getContext(), "__init__");
  auto initResult = shared.lookupAndResolveDecl(initName, loc, module,
                                                /*searchParentScopes=*/false,
                                                /*resolveTarget=*/false);
  // Package has no __init__.
  if (!initResult.isSuccess())
    return FailureOr<ASTDecl *>(nullptr);
  ASTDecl *initDecl = initResult.getIfSuccess().front();
  if (failed(resolveBody(*initDecl, loc)))
    return failure();
  return initDecl;
}

SmallVector<ASTDecl *> DeclResolver::lookupNonModuleDecls(ASTDecl &initDecl,
                                                          StringAttr name,
                                                          SMLoc loc,
                                                          bool resolveTarget) {
  SmallVector<ASTDecl *> nonModuleDecls;
  auto lookup =
      shared.lookupAndResolveDecl(name, loc, initDecl,
                                  /*searchParentScopes=*/false, resolveTarget);
  if (lookup.isSuccess()) {
    for (ASTDecl *d : lookup.getIfSuccess()) {
      auto *op = d->getIfOperation();
      if (!isa_and_nonnull<FileModuleOp, PackageOp>(op))
        nonModuleDecls.push_back(d);
    }
  }
  return nonModuleDecls;
}

LogicalResult DeclResolver::importDeclFromModule(
    ASTDecl &dest, PackageOp currentPackage, StringAttr moduleName,
    StringAttr sourceName, StringAttr destName, SMLoc loc, SMLoc sourceNameLoc,
    SMLoc destNameLoc, bool resolveTarget) {

  ASTDecl &module = shared.importModule(moduleName, currentPackage, loc);
  shared.notifyListenerOnModuleImport(module, moduleName, loc);

  // Check to see if the module has the construct we are importing.
  auto result = shared.lookupAndResolveDecl(sourceName, sourceNameLoc, module,
                                            /*searchParentScopes=*/false,
                                            /*resolveTarget=*/resolveTarget);
  if (result.isErroneous())
    return failure();
  if (result.isFailure()) {
    StringRef name =
        cast_or_null<mlir::SymbolOpInterface>(module.getIfOperation())
            .getName();
    StringRef declType = isa_and_nonnull<PackageOp>(module.getIfOperation())
                             ? "package"
                             : "module";
    emitError(sourceNameLoc, declType + " '" + name + "' does not contain '" +
                                 sourceName.getValue() + "'");
    return failure();
  }
  ArrayRef<ASTDecl *> results = result.getIfSuccess();
  assert(!results.empty() && "other cases handled above");

  // If the initial lookup only found submodule/package decls in a package,
  // the name might also refer to a re-exported symbol from __init__.mojo.
  // The directory-scan creates whole-module imports that shadow wildcard
  // imports from __init__, so look up the name directly in __init__'s scope.
  // Note: the lookup results here are already resolved, so we only need to
  // check for FileModuleOp/PackageOp (unlike the wildcard path which also
  // sees UnresolvedImportOp entries from raw getDeclsInScope()).
  SmallVector<ASTDecl *> reExported;
  if (llvm::all_of(results, [](ASTDecl *d) {
        return isa_and_nonnull<FileModuleOp, PackageOp>(d->getIfOperation());
      })) {
    auto initOrFailure = bodyResolvePackageInit(module, loc);
    if (failed(initOrFailure))
      return failure();
    if (ASTDecl *initDecl = *initOrFailure) {
      reExported = lookupNonModuleDecls(*initDecl, sourceName, sourceNameLoc,
                                        resolveTarget);
      if (!reExported.empty())
        results = ArrayRef(reExported);
    }
  }

  shared.notifyListenerOnRef(results, sourceName, sourceNameLoc);
  shared.notifyListenerOnRef(results, destName, destNameLoc);

  // Import the desired declaration (struct, function, etc.) that the user
  // specifically asked for.
  if (failed(aliasImportDecls(results, destName, sourceName, moduleName,
                              destNameLoc, dest, false)))
    return failure();

  // Also look for extensions in the source module.
  // When importing any declaration from a module, import all extensions from
  // that module so they're available in the destination scope.
  // All extensions known to their parents as e.g. `extension:MyStruct` but
  // also as `extension:` so asking for `extension:` will get all extensions.
  StringAttr extensionNameAttr = StringAttr::get(getContext(), "extension:");
  auto requestedModuleExts =
      shared.lookupAndResolveDecl(extensionNameAttr, sourceNameLoc, module,
                                  /*searchParentScopes=*/false,
                                  /*resolveTarget=*/false);
  if (requestedModuleExts.isSuccess()) {
    ArrayRef<ASTDecl *> allExtensions = requestedModuleExts.getIfSuccess();
    if (!allExtensions.empty()) {
      shared.notifyListenerOnRef(allExtensions, extensionNameAttr,
                                 sourceNameLoc);
      shared.notifyListenerOnRef(allExtensions, extensionNameAttr, destNameLoc);

      // Import under "extension:" for finding all extensions in a module
      if (failed(aliasImportDecls(allExtensions, extensionNameAttr,
                                  extensionNameAttr, moduleName, destNameLoc,
                                  dest, true))) {
        emitError(destNameLoc, "failed to import extensions from module '" +
                                   moduleName.getValue() + "'");
        return failure();
      }
      // Now that we have all the extensions, go through each one and register
      // it under its specific name (e.g. "extension:SIMD") so
      // collectTypeAndExtensions can find them.
      for (ASTDecl *extensionDecl : allExtensions) {
        auto extOp =
            dyn_cast_or_null<ExtensionDeclOp>(extensionDecl->getIfOperation());
        if (!extOp)
          continue;
        auto targetStructName = extOp.getTargetStructName().value();
        StringAttr specificExtensionName = StringAttr::get(
            getContext(), "extension:" + targetStructName.str());
        if (failed(aliasImportDecls({extensionDecl}, specificExtensionName,
                                    extensionNameAttr, moduleName, destNameLoc,
                                    dest, true))) {
          emitError(destNameLoc, "failed to import extension for '" +
                                     targetStructName + "' from module '" +
                                     moduleName.getValue() + "'");
          return failure();
        }
      }
    }
  }

  return success();
}

LogicalResult DeclResolver::importWildCardDeclsFromModule(ASTDecl &context,
                                                          StringAttr moduleName,
                                                          bool isFullImport,
                                                          llvm::SMLoc loc) {
  PackageOp currentPackage =
      dyn_cast_or_null<PackageOp>(context.getIfOperation());
  if (!currentPackage && context.getIfOperation())
    currentPackage = context.getIfOperation()->getParentOfType<PackageOp>();

  // Make sure the module has been resolved.
  ASTDecl &module = shared.importModule(moduleName, currentPackage, loc);
  if (failed(resolveBody(module, loc)))
    return failure();

  // Resolve pending wildcard imports in this module.
  if (failed(resolveAllWildcardImports(module)))
    return failure();

  // For packages, resolve __init__'s body so we can look up re-exported
  // symbols that may be shadowed by submodule names of the same name.
  FailureOr<ASTDecl *> initOrFailure = bodyResolvePackageInit(module, loc);
  if (failed(initOrFailure))
    return failure();
  ASTDecl *initDecl = *initOrFailure;

  // Wildcard imports don't import decls with a leading '_'.
  LogicalResult result = success();
  for (const auto &[name, decls] : module.getDeclsInScope()) {
    // Ignore erroneous children, which have nothing in them.
    if (decls.empty())
      continue;
    if (!isFullImport && isInternalName(name))
      continue;

    ArrayRef<ASTDecl *> importDecls = decls;

    // If this name only has whole-module imports (from directory scanning) or
    // resolved module/package decls, check __init__'s scope for re-exported
    // non-module decls that should take priority. Unlike the explicit import
    // path, we iterate raw getDeclsInScope() entries here, so we must also
    // check for UnresolvedImportOp (whole-module imports without a declName).
    SmallVector<ASTDecl *> reExported;
    if (initDecl && llvm::all_of(decls, [](ASTDecl *d) {
          auto *op = d->getIfOperation();
          if (isa_and_nonnull<FileModuleOp, PackageOp>(op))
            return true;
          // Whole-module imports (no declName) from directory scanning.
          if (auto importOp = dyn_cast_or_null<UnresolvedImportOp>(op))
            return !importOp.getDeclNameAttr();
          return false;
        })) {
      reExported = lookupNonModuleDecls(*initDecl, name, loc,
                                        /*resolveTarget=*/true);
      if (!reExported.empty())
        importDecls = ArrayRef(reExported);
    }

    if (failed(aliasImportDecls(importDecls, name, name, moduleName, loc,
                                context, false)))
      result = failure();
  }

  // Also import all extensions from the source module, similar to what
  // importDeclFromModule does. This ensures that when doing wildcard imports,
  // extensions are available in the destination scope.
  // Extensions are registered under "extension:" so we can find all of them.
  StringAttr extensionNameAttr = StringAttr::get(getContext(), "extension:");
  auto moduleExtensions =
      shared.lookupAndResolveDecl(extensionNameAttr, loc, module,
                                  /*searchParentScopes=*/false,
                                  /*resolveTarget=*/false);
  if (moduleExtensions.isSuccess()) {
    ArrayRef<ASTDecl *> allExtensions = moduleExtensions.getIfSuccess();
    if (!allExtensions.empty()) {
      shared.notifyListenerOnRef(allExtensions, extensionNameAttr, loc);

      // Import under "extension:" for finding all extensions in a module
      if (failed(aliasImportDecls(allExtensions, extensionNameAttr,
                                  extensionNameAttr, moduleName, loc, context,
                                  true))) {
        emitError(loc, "failed to import extensions from module '" +
                           moduleName.getValue() + "'");
        return failure();
      }

      // Now register each extension under its specific name (e.g.
      // "extension:SIMD") so collectTypeAndExtensions can find them.
      for (ASTDecl *extensionDecl : allExtensions) {
        auto extOp =
            dyn_cast_or_null<ExtensionDeclOp>(extensionDecl->getIfOperation());
        if (!extOp)
          continue;
        auto targetStructName = extOp.getTargetStructName();
        if (!targetStructName)
          continue;
        StringAttr specificExtensionName = StringAttr::get(
            getContext(), "extension:" + targetStructName.value().str());
        if (failed(aliasImportDecls({extensionDecl}, specificExtensionName,
                                    extensionNameAttr, moduleName, loc, context,
                                    true))) {
          emitError(loc, "failed to import extension for '" +
                             targetStructName.value() + "' from module '" +
                             moduleName.getValue() + "'");
          return failure();
        }
      }
    }
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

  // If we are currently name binding this operation, we found a cycle, reject
  // it with an error.
  if (failed(declsCurrentlyProcessing.insert(&decl, loc))) {
    auto diag =
        emitError(decl.getLoc(),
                  "attempt to resolve a recursive reference to declaration");

    auto addDeclName = [&](ASTDecl *decl) {
      std::optional<StringRef> name = decl->getUserNameIfOperation();
      if (!name)
        return;
      diag << " '";
      if (auto structOp = dyn_cast_or_null<StructDeclOp>(
              decl->getParentDecl()->getIfOperation()))
        diag << structOp.getDeclName().getValue() << ".";
      diag << *name << "'";
    };

    addDeclName(&decl);
    diag.attachNote(loc) << "referenced from here";

    // Include a stack trace of notes showing why this is being cyclicly
    // resolved.
    for (ASTDecl *prev : llvm::reverse(declsCurrentlyProcessing.stack)) {
      // Bottom out when we find the declaration in question.
      diag.attachNote(prev->getLoc()) << "by declaration";
      addDeclName(prev);

      diag.attachNote(declsCurrentlyProcessing.map[prev])
          << "referenced through this use";
      if (prev == &decl)
        break;
    }
    decl.setErroneous();
    return failure();
  }

  // Handle decls that are loaded from bytecode. These decls are not parsed like
  // decls originating from source files.
  if (decl.loadedFromBytecode) {
    if (failed(shared.resolveDeclFromBytecode(decl, howResolved)))
      decl.setErroneous();

    declsCurrentlyProcessing.pop();
    return success(!decl.isErroneous());
  }

  // If the signature hasn't been parsed, do so.
  if (decl.resolvedness < DeclResolvedness::signature) {
    // Handle each operation that can be name bound.  We handle this by
    // restoring the lexer to the position where parsing can continue, calling
    // the `resolveSignature` method for the op, and re-saving the new cursor
    // for the next stage of resolution.
    if (auto declOp = decl.getIfOperation()) {
      TypeSwitch<Operation &>(*declOp)
          .Case<FnOp, StructDeclOp, StructFieldOp, TraitDeclOp, ExtensionDeclOp,
                AliasDeclOp>([&](auto op) {
            // If this is a synthetic decl, resolve it specially.
            if (decl.getCursor().isInvalid()) {
              if constexpr (std::is_same_v<FnOp, decltype(op)>) {
                if (failed(resolveSyntheticSignature(op, decl)))
                  decl.setErroneous();
                return;
              }
              if constexpr (std::is_same_v<AliasDeclOp, decltype(op)>) {
                if (failed(resolveSyntheticSignature(op, decl)))
                  decl.setErroneous();
                return;
              }
            }

            Lexer lexer(shared.diags, decl.getCursor());

            // Generate pretty stack traces if a crash happens in this
            // scope.
            LexerCrashReporter crashReporter(lexer, decl.getLoc(),
                                             "resolving decl signature");

            // Resolve the signature: on a parse error, we note that the
            // decl is malformed and should not be referenced to silence
            // downstream errors.
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
          .Case<LIT::FileModuleOp, ModuleOp, PackageOp, ImportOp,
                UnresolvedWildcardImportOp>([&](auto op) { /*Nothing*/ })
          .Default([&](Operation &attr) {
            llvm_unreachable(
                "do not know how to resolve the signature of this decl!");
          });
    } else if (auto typeValue = decl.getIfTypeValue()) {
      auto traitType = dyn_cast_or_null<TraitType>(decl.getIfTypeValue());
      assert(traitType && "do not know how to resolve the signature of this "
                          "decl!");
      if (failed(resolveSignature(traitType, decl)))
        decl.setErroneous();
    } else {
      llvm_unreachable(
          "do not know how to resolve the signature of this decl!");
    }
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
    //    decl.resolvedness = DeclResolvedness::body;

    // Handle each operation that can be name bound.
    if (decl.isErroneous()) {
      // If the decl is already erroneous, trying to process further may crash
      // or cause spurious error messages.
    } else if (auto declOp = decl.getIfOperation()) {
      TypeSwitch<Operation &>(*declOp)
          .Case<FileModuleOp, FnOp, StructDeclOp, StructFieldOp,
                ExtensionDeclOp, TraitDeclOp, AliasDeclOp>([&](auto op) {
            // If this is a synthetic decl, complete it specially.
            if (decl.getCursor().isInvalid()) {
              if constexpr (std::is_same_v<FnOp, decltype(op)>) {
                if (op.isSynthetic() && failed(resolveSyntheticBody(op, decl)))
                  decl.setErroneous();
                return;
              }
            }

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
          .Case<ModuleOp, ImportOp, UnresolvedImportOp,
                UnresolvedWildcardImportOp>([&](auto op) { /*Nothing*/ })
          .Default([&](Operation &attr) {
            llvm_unreachable(
                "do not know how to resolve the body of this decl!");
          });
    } else if (auto typeVal = decl.getIfTypeValue()) {
      auto traitType = dyn_cast_or_null<TraitType>(decl.getIfTypeValue());
      assert(traitType && "do not know how to resolve the body of this decl!");
      if (failed(resolveBody(traitType, decl)))
        decl.setErroneous();
    } else {
      llvm_unreachable("do not know how to resolve the body of this decl!");
    }

    if (decl.resolvedness == DeclResolvedness::signature)
      decl.resolvedness = DeclResolvedness::body;
  }

  declsCurrentlyProcessing.pop();
  // If decl is busted, then return failure.
  return success(!decl.isErroneous());
}

void DeclResolver::resolveAllWithin(ASTDecl &decl) {
  std::deque<ASTDecl *> worklist{&decl};
  while (!worklist.empty()) {
    ASTDecl *declIt = worklist.back();
    worklist.pop_back();

    if (declIt->isDisabled())
      continue;

    (void)resolveBody(*declIt, declIt->getLoc());

    for (auto &[name, decls] : declIt->getDeclsInScope()) {
      for (ASTDecl *child : decls) {
        if (child->getParentDecl() == declIt)
          worklist.push_front(child);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Top-Level Decl Resolution

void DeclResolver::resolveReferencedDecls() {
  // Iteratively resolve all of the parsed decls that got referenced outside
  // the main container (typically stdlib/library declarations).
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
        if (isa_and_nonnull<FnOp, FileModuleOp, PackageOp, ImportOp,
                            UnresolvedImportOp, UnresolvedWildcardImportOp,
                            StructDeclOp, TraitDeclOp, AliasDeclOp>(
                decl.getIfOperation())) {
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
}

void DeclResolver::resolveAllReferencedFrom(ASTDecl &decl,
                                            bool eraseUnparsedDecls) {
  CompilerTimeTraceScope traceScope("resolveAllReferencedFrom", [&] {
    return decl.getUserNameIfOperation().value_or("").str();
  });

  // The first stage is to fully resolve all of the decls recursively defined
  // within the main container. These decls provide the anchor for resolution.
  std::deque<ASTDecl *> worklist({&decl});
  while (!worklist.empty()) {
    ASTDecl *declIt = worklist.back();
    worklist.pop_back();

    // Resolve the decl.
    (void)resolveBody(*declIt, declIt->getLoc());

    if (declIt->isDisabled())
      continue;

    // When validating doc strings, we wish to only validate those defined on
    // decl in the main container. As this point the main container decl has
    // been fully resolved, so it's an opportune time to validate.
    validateDocString(*declIt);

    // If this is a package, resolve all of the modules within it as a pre-step.
    // Normally these get lazily resolved, but if we're forcing pulling them in,
    // we need to do it now.
    if (isa_and_nonnull<PackageOp>(declIt->getIfOperation())) {
      for (auto &[_, decls] : declIt->getDeclsInScope())
        if (isa_and_nonnull<UnresolvedImportOp>(
                decls.front()->getIfOperation()))
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

  // After all of the children within `decl` have been fully resolved,
  // iteratively resolve all of the outside decls that got referenced.
  // Skip when errors have already been emitted: resolving library/stdlib
  // declarations is expensive (potentially the entire stdlib) and
  // unnecessary when compilation will fail anyway.
  if (!shared.diags.isErrorEmitted())
    resolveReferencedDecls();

  // Erase unresolved operations from source.
  if (eraseUnparsedDecls) {
    for (ASTDecl *decl : parsedDeclList) {
      // During trait body resolution we create decls that point to parent
      // trait decl's FnOps. In order to avoid double frees later on in this
      // loop bail early if we come across such a decl.
      if (decl->getCursor().isInvalid())
        continue;
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

ASTDecl *
DeclResolver::getDeclForTypeSymbolIfExists(SymbolRefAttr symbol) const {
  auto it = declForTypeSymbol.find(symbol);
  return it != declForTypeSymbol.end() ? it->second : nullptr;
}

ASTDecl *DeclResolver::getDeclForFuncSymbol(SymbolRefAttr attr) const {
  auto it = declForFuncSymbol.find(attr);
  return it != declForFuncSymbol.end() ? it->second : nullptr;
}

Operation *DeclResolver::finalizeFuncSignature(FnOp funcOp, ASTDecl &decl) {
  // Install it in the symbol table and check for redefinition while doing so.
  Operation *existing = shared.uniquifyNameAndAddToParentSymbolTable(funcOp);
  // Remember the mapping from its fully mangled symbol so we can find its AST
  // representation and body from IR references.
  // NOTE: this has to run after `uniquifyNameAndAddToParentSymbolTable` as the
  // call above might update the symbol name when there is a name collision.
  declForFuncSymbol[getFullyResolvedSymbolRef(funcOp)] = &decl;
  return existing;
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
  FnOp userMainFn = cast_or_null<FnOp>(funcDecl.getIfOperation());
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
      "std.builtin._startup", /*currentPackage=*/nullptr, funcDecl.getLoc());
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
  FnOp mainShimProtoFn =
      cast_or_null<FnOp>(mainShimProtoDecl->getIfOperation());

  // Builder function.
  StringAttr mainAttr = StringAttr::get(getContext(), "main");
  auto shimMainFn = cast<FnOp>(builder.clone(*mainShimProtoFn));
  shimMainFn.setSymNameAttr(mainAttr);
  shimMainFn.setLinkageNameAttr(
      LinkageNameAttr::get(shimMainFn->getContext(), "main"));
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
  FnOp mainWrapperFn = cast_or_null<FnOp>(mainWrapperDecl->getIfOperation());
  FnTypeGeneratorType mainWrapperSigGen = mainWrapperFn.getFuncTypeGenerator();

  // Generate a reference to the main wrapper function, which expects the user
  // main to be provided via an parameter.
  FnType mainWrapperSig = mainWrapperSigGen.getBody();
  FnMetadataAttr mainWrapperFnMeta = mainWrapperSig.getMetadata();
  auto strippedMainWrapperFnMeta = FnMetadataAttr::get(
      mainWrapperFnMeta.getArgListAttrs(),
      mainWrapperFnMeta.getNumImplicitOriginDecls(),
      mainWrapperFnMeta.getCaptureOrigins(),
      mainWrapperFnMeta.getIsNestedOriginExclusivityCheckingDisabled(),
      mainWrapperFnMeta.getConstraints());
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
  Value wrappedCallResult =
      CallOp::create(
          shimBodyBuilder, mainWrapperSigGen.getUserResultType(), wrapperFnRef,
          /*originParams=*/ArrayRef<TypedAttr>(), shimMainFn.getArguments())
          .getResult(0);

  // Align sugar if needed.
  if (wrappedCallResult.getType() != shimMainFn.getArgumentTypes()[0]) {
    assert(
        isEqualCanon(wrappedCallResult.getType(),
                     shimMainFn.getArgumentTypes()[0]) &&
        "wrapped call result type does not match shim main fn argument type");
    wrappedCallResult = RebindOp::create(
        shimBodyBuilder, shimMainFn.getArgumentTypes()[0], wrappedCallResult);
  }

  IREmitter::emitNormalReturn(shimBodyBuilder, wrappedCallResult);

  exportedSymbolNames.insert({mainAttr, funcDecl.getLoc()});
}

//===----------------------------------------------------------------------===//
// Decl Helpers

static void printConstraints(llvm::raw_ostream &os,
                             ArrayRef<ConstraintAttr> constraints) {
  if (constraints.empty())
    return;
  os << '{';
  llvm::interleave(
      constraints, os,
      [&](ConstraintAttr constraint) {
        ASTType::printParam(os, constraint.getProposition(), /*diags=*/nullptr);
      },
      ",");
  os << '}';
}

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
  ArrayRef<PogMetadataAttr> pogs = fullSig.getParamListAttrs().getPogs();
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
            type = type.getParameterListInfo().elementType;
          }
          os << type.getAsString(/*diags=*/nullptr);
          printConstraints(os, pogs[idx].getConstraints());
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

    // Required keyword arguments can be overloaded on.
    if (fullSig.getArgListAttrs().getPassingKind(argNo) == PassingKind::KwOnly)
      mangledName += fullSig.getArgName(argNo).str() + ":";

    // If this had adjustments added to it because of its argument convention /
    // variadic state, strip them off.
    unsigned numStars = 0;
    if (fullSig.isPosVarArg(argNo)) {
      argType = ASTType(fullSig.getIfVariadicListOrPack(argNo))
                    .getVariadicListInfo()
                    .elementType;
      convention = fullSig.getVariadicConvention(argNo);
      numStars = 1;
    } else if (fullSig.isPack(argNo)) {
      TypedAttr packVariadic = ASTType(fullSig.getIfVariadicListOrPack(argNo))
                                   .getVariadicPackInfo()
                                   .typeList;
      mangledName += '*';
      ASTType::printParam(os, packVariadic, /*diags=*/nullptr);
      continue;
    } else if (fullSig.isKwVarArg(argNo)) {
      // TODO: Propagate convention correctly.
      convention = ArgConvention::ReadReg;
      argType = argType.getKwargsDictRefValueType();
      numStars = 2;
    } else {
      argType = RefType::stripRefConvention(argType, convention);
    }
    mangledName += argType.getAsString(/*forDiag=*/nullptr);

    // Add suffix to disambiguate overloadable conventions.
    switch (convention) {
    case ArgConvention::OwnedReg:
      llvm_unreachable("not used by the parser");
    case ArgConvention::OwnedMem:
    case ArgConvention::DeinitMem:
      mangledName += '$';
      break;
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

  // Add def constraints to the mangled name.
  printConstraints(os, fullSig.getFnMetadata().getConstraints());

  // Having "@" in mangled names confuses gnu ld and triggers error at linking
  // stage. See issue #6918. So replacing "@" with "_".
  std::replace(mangledName.begin(), mangledName.end(), '@', '_');
  return StringAttr::get(baseName.getContext(), mangledName);
}
