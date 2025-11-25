//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITInterfaces.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "KGEN/MojoParser/DocString.h"
#include "llvm/ADT/StringExtras.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

ASTDecl::ASTDecl(SharedState &shared, DeclIRValue irValue, llvm::SMLoc loc,
                 ASTDecl *parentDecl, LexerCursor cursor, LexerCursor endCursor,
                 ssize_t indentation)
    : shared(shared), irValue(irValue), loc(loc), parentDecl(parentDecl),
      cursor(cursor), endCursorState(endCursor.getState()),
      indentation(indentation) {
  resolvedness = DeclResolvedness::unparsed;
  referencedFromBytecode = false;
  hasReferenceError = false;
  hasBodyDecorators = false;
  loadedFromBytecode = false;
}

DocStringAttr ASTDecl::getDocString() const {
  if (auto astDeclOp = dyn_cast_or_null<ASTDeclInterface>(getIfOperation()))
    return astDeclOp.getDocStringAttr();
  return {};
}

void ASTDecl::setErroneous() { hasReferenceError = true; }

std::optional<DocString> ASTDecl::getParsedDocString() const {
  if (auto rawDocStr = getDocString())
    return DocString(rawDocStr);
  return {};
}

ArrayRef<ASTDecl *> ASTDecl::lookupInCurrentScope(StringRef name) const {
  return lookupInCurrentScope(StringAttr::get(getContext(), name));
}

ArrayRef<ASTDecl *> ASTDecl::lookupInCurrentScope(StringAttr name) const {
  assert(resolvedness == DeclResolvedness::body &&
         "cannot perform lookup in a decl that isn't fully resolved");
  if (!declsInScope)
    return {};

  auto it = declsInScope->find(name);
  if (it != declsInScope->end() && !it->second.empty())
    return it->second;
  return {};
}

/// If this is a method of a struct or trait, return the decl for the struct
/// or trait.
ASTDecl *ASTDecl::tryGetMethodParentDecl() const {
  // Methods are always FuncOps.
  if (!isa_and_nonnull<FnOp>(getIfOperation()))
    return nullptr;

  // Don't return non-null for nested functions or module-level functions.
  ASTDecl *parent = getParentDecl();
  return isa_and_nonnull<StructDeclOp, TraitDeclOp, ExtensionDeclOp>(
             parent->getIfOperation())
             ? parent
             : nullptr;
}

/// Collect the struct/trait declaration and all visible extension declarations
/// for the given type from this use-site context.
llvm::SmallVector<ASTDecl *, 4>
ASTDecl::collectTypeAndExtensions(ASTType type, llvm::SMLoc callLoc) {
  SharedState &shared = getShared();
  auto astDecl = type.getDecl(shared);

  SmallVector<ASTDecl *, 4> result;
  if (astDecl)
    result.push_back(astDecl);

  // Handle both struct and trait types for extension lookup.
  if (!astDecl || !astDecl->getNameIfOperation() ||
      !isa<StructDeclOp, TraitDeclOp>(astDecl->getIfOperation()))
    return result;

  // Now find all extensions that target this struct/trait.
  // Extensions are registered with the name of their target type, prefixed
  // with "extension:" (e.g., "extension:Spaceship") so that we can do this
  // lookup here.
  StringRef typeName = astDecl->getNameIfOperation().value();
  std::string extensionName = "extension:" + typeName.str();
  LookupAllResult lookupResult =
      shared.lookupAllDeclsWithName(extensionName, callLoc, *this, true);

  // Only consider results from successful lookups. Lookups with isErroneous()
  // means the error was already diagnosed. Lookups with isFailure() should
  // be rare since we expect to find at least the original type declaration,
  // but we gracefully handle it by skipping extension lookup.
  if (!lookupResult.isSuccess())
    return result;

  ArrayRef<ASTDecl *> foundAstDecls = lookupResult.getIfSuccess();
  for (ASTDecl *foundAstDecl : foundAstDecls) {
    if (failed(shared.declResolver->resolveBody(*foundAstDecl, callLoc))) {
      // Do nothing, skip it. Errors were already printed out, and we don't
      // mind missing call candidates from erroneous extensions.
      continue;
    }

    if (isa_and_nonnull<ExtensionDeclOp>(foundAstDecl->getIfOperation()))
      result.push_back(foundAstDecl);
  }

  return result;
}

void ASTDecl::takeDecls(ASTDecl &src) {
  if (src.isErroneous())
    setErroneous();
  for (auto &[name, children] : src.getDeclsInScope())
    for (ASTDecl *child : children)
      child->parentDecl = this;
  declsInScope = std::move(src.declsInScope);
  counter = src.counter;
  knownAssumptions = std::move(src.knownAssumptions);
}

DenseMap<SymbolRefAttr, std::pair<SymbolRefAttr, SMLoc>> *
ASTDecl::getTraitConformanceLineage(bool createIfMissing) {
  if (!traitConformanceLineage && createIfMissing)
    traitConformanceLineage.reset(new TraitConformanceLineageType());
  return traitConformanceLineage.get();
}

void ASTDecl::getKnownAssumptionsIncludingParents(
    SmallVectorImpl<ConstraintAttr> &assumptions) const {
  const ASTDecl *decl = this;
  while (decl) {
    if (decl->knownAssumptions)
      assumptions.append(decl->knownAssumptions->begin(),
                         decl->knownAssumptions->end());
    decl = decl->getParentDecl();
  }
}

void ASTDecl::insertKnownAssumptions(ArrayRef<ConstraintAttr> assumptions) {
  if (!knownAssumptions)
    knownAssumptions.reset(new llvm::SetVector<ConstraintAttr>());
  knownAssumptions->insert(assumptions.begin(), assumptions.end());
}

/// Return the nearest parameter scope (i.e. DeclInterface) for the given decl,
/// as well as the total depth from the nearest file module.
static std::pair<ASTDecl *, size_t> getNearestParamScopeAndDepth(
    ASTDecl *decl, function_ref<void(const ASTDecl *)> checkForCollision) {
  ASTDecl *paramScope = nullptr;
  size_t depth = 0;
  while (decl) {
    checkForCollision(decl);

    if (isa_and_nonnull<DeclInterface>(decl->getIfOperation())) {
      ++depth;
      if (!paramScope)
        paramScope = decl;
      if (isa_and_nonnull<FileModuleOp>(decl->getIfOperation()))
        break;
    }

    decl = decl->getParentDecl();
  }

  return {paramScope, --depth}; // Adjust so depth starts at 0.
}

/// Add an unresolved wild card import into this scope.
void ASTDecl::addUnresolvedWildCardImport(StringAttr importedModule,
                                          bool isFullImport, SMLoc loc) {
  // Lazy allocate the MapVector.
  if (!unresolvedWildcardImports)
    unresolvedWildcardImports.reset(new UnresolvedWildcardImportsType());
  unresolvedWildcardImports->insert({importedModule, {loc, isFullImport}});
}

/// Mangle a parameter name for the given parameter scope and scope depth. Due
/// to the use of depth, the mangling doesn't change when the order of function
/// declarations change, so we have hash stability.
static StringAttr mangleParamNameImpl(const Twine &name, size_t depth,
                                      ASTDecl *paramScope) {
  MLIRContext *ctx = paramScope->getContext();

  // Top level funcs/structs are the most common, so we want to simplify the
  // mangling for that case. Many tests (and real world code too) has a single
  // parameter in a scope, so we also try to make that case nicer.
  std::string suffix = "`";
  if (depth != 1)
    suffix.append(llvm::utostr(depth) + 'x');
  if (size_t id = paramScope->getNextUniqueID(); id != 0)
    suffix.append(llvm::utostr(id));

  return StringAttr::get(ctx, name + suffix);
}

StringAttr ASTDecl::mangleUserDefinedParamName(StringAttr name) {
  bool hasCollision = false;
  auto [paramScope, depth] =
      getNearestParamScopeAndDepth(this, [&](const ASTDecl *curScope) {
        hasCollision =
            hasCollision || !curScope->lookupInCurrentScope(name).empty();
      });
  if (!hasCollision)
    return name;

  return mangleParamNameImpl(name.strref(), depth, paramScope);
}

StringAttr ASTDecl::mangleParamName(const Twine &name) {
  // This function always mangles, so no need to check for collisions.
  auto [paramScope, depth] =
      getNearestParamScopeAndDepth(this, [&](const ASTDecl *) {});
  return mangleParamNameImpl(name, depth, paramScope);
}

void ASTDecl::dump() const {
  // The value is either an operation or a type of MLIR `Value`.
  if (auto *op = getIfOperation()) {
    // Print without verifying, since IR could be in an invalid state.
    op->print(llvm::errs(), mlir::OpPrintingFlags().printGenericOpForm());
    llvm::errs() << "\n";
  } else if (auto cv = getIfIRValue()) {
    cv.dump();
  } else {
    llvm::errs() << "<null decl>\n";
  }
}

ASTType ASTDecl::getIfTypeValue() const {
  if (auto cv = getIfIRValue().getIfPValue())
    return cv.getIfTypeValue();
  return {};
}

std::optional<StringRef> ASTDecl::getNameIfOperation() const {
  if (Operation *op = getIfOperation())
    if (auto decl = dyn_cast<ASTDeclInterface>(op))
      return decl.getDeclName().getValue();
  return {};
}

PValue ASTDecl::getFuncAsPValue() const {
  return SymbolConstantAttr::get(
      getSymbolRef(), cast<FnOp>(getIfOperation()).getFuncTypeGenerator());
}

/// Return the SymbolRefAttr for a declaration, including all scoping that may
/// be needed, making it unique for every declaration.  This returns null for
/// named values that do not have a declaration.
SymbolRefAttr ASTDecl::getSymbolRef() const {
  auto op = dyn_cast_if_present<mlir::SymbolOpInterface>(getIfOperation());
  if (!op)
    return {};
  assert((!isa<FnOp>(op) || resolvedness >= DeclResolvedness::signature) &&
         "Functions don't have a symbol until their signatures are resolved");
  return getFullyResolvedSymbolRef(op);
}

/// Given an MLIR op for a struct declaration, return the self type.
Type ASTDecl::computeSelfTypeForStruct(StructDeclOp structOp) {
  SmallVector<TypedAttr> parameters;
  for (auto decl : structOp.getParams()) {
    // We're using the parameter from the type declaration scope in the
    // parameter binding list.
    parameters.push_back(ParamDeclRefAttr::get(decl));
  }

  // Methods on structs (but not classes) take the struct implicitly by
  // pointer so they can use and mutate it.
  return structOp.bindReference(parameters);
}

Type ASTDecl::computeSelfTypeForTrait(TraitDeclOp traitOp) {
  // The last parameter to the trait is the 'T' parameter which (when everything
  // gets instantiated) resolves to the final type the trait is instantiated on.
  return ASTType(ParamDeclRefAttr::get(traitOp.getParamsAttr().back()));
}

void ASTDecl::findExtensionsInScopeForStruct(
    SymbolRefAttr targetStruct, llvm::SmallPtrSetImpl<ASTDecl *> &results,
    std::optional<SymbolRefAttr> filterTrait) {
  for (auto &[name, decls] : getDeclsInScope()) {
    for (ASTDecl *decl : decls) {
      if (auto extOp =
              dyn_cast_or_null<ExtensionDeclOp>(decl->getIfOperation())) {
        // Check if this extension targets our struct
        if (extOp.getTargetStructAttr() &&
            extOp.getTargetStructAttr() == targetStruct) {
          // If no trait filter specified, add this extension
          if (!filterTrait.has_value()) {
            results.insert(decl);
            continue;
          }

          // Check if this extension implements our trait or any trait that
          // inherits from it
          if (!extOp.getCanonicalTrait()) {
            // Extension doesn't have canonicalTrait computed yet - skip it
            // This happens during error conditions or early parsing phases
            continue;
          }

          // Use the extension's canonicalTrait (flattened hierarchy) to check
          TraitType extCanonicalTrait = extOp.getCanonicalTrait().value();
          for (SymbolRefAttr symbol : extCanonicalTrait.getSymbols()) {
            if (symbol == filterTrait.value()) {
              results.insert(decl);
              break; // Found it, no need to check more symbols
            }
          }
        }
      }
    }
  }
}
