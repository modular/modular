//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITInterfaces.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser/DocString.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

DocStringAttr ASTDecl::getDocString() const {
  if (auto astDeclOp = dyn_cast<ASTDeclInterface>(this))
    return astDeclOp.getDocStringAttr();
  return {};
}

std::optional<DocString> ASTDecl::getParsedDocString() const {
  if (auto rawDocStr = getDocString())
    return DocString(rawDocStr);
  return {};
}

ArrayRef<ASTDecl *> ASTDecl::lookupInCurrentScope(StringRef name) const {
  return lookupInCurrentScope(StringAttr::get(getContext(), name));
}

ArrayRef<ASTDecl *> ASTDecl::lookupInCurrentScope(StringAttr name) const {
  assert(resolvedness == DeclResolvedness::fully &&
         "cannot perform lookup in a decl that isn't fully resolved");
  auto it = declsInScope.find(name);
  if (it != declsInScope.end() && !it->second.empty())
    return it->second;
  return {};
}

void ASTDecl::takeDecls(ASTDecl &src) {
  hasReferenceError |= src.hasReferenceError;
  for (auto &[name, children] : src.declsInScope)
    for (ASTDecl *child : children)
      child->parentDecl = this;
  declsInScope = std::move(src.declsInScope);
}

StringAttr ASTDecl::getUniqueParamNameNew(StringAttr name,
                                          bool isUserDefinedDecl) {
  // First, calculate depths and check if we need to mangle due to collisions.
  const ASTDecl *curScope = this;
  bool hasCollision = false;
  size_t depth = 0;
  while (curScope) {
    // We only check for collisions if this is a user defined decl. Implicit
    // declarations always get mangled.
    if (isUserDefinedDecl && !hasCollision) {
      ArrayRef<ASTDecl *> result = curScope->lookupInCurrentScope(name);
      hasCollision = !result.empty();
    }
    depth++;

    if (isa<FileModuleOp>(*curScope))
      break;
    curScope = curScope->parentDecl;
  }
  depth--; // Adjust so depth starts at 0.

  // User visible declarations only get mangled if there is a collision.
  if (isUserDefinedDecl && !hasCollision)
    return name;

  // This mangling guarantees that whatever name we generate is unique,
  // independently of whether the name we are mangling comes from an explicit
  // declaration by the Mojo user. Due to the use of depth, the mangling doesn't
  // change when the order of function declarations change, so we have hash
  // stability as well.
  return StringAttr::get(name.getContext(), name.getValue() + "`" +
                                                Twine(depth) + "x" +
                                                Twine(getNextUniqueID()));
}

StringAttr ASTDecl::getUniqueParamName(const Twine &name, bool isLifetime,
                                       bool dontRenameOutermost) {
  // Find the enclosing isolated from above decl that will scope parameter
  // names.
  ASTDecl *outermostFuncScope = nullptr;
  ASTDecl *innermostFuncScope = nullptr;
  ASTDecl *scope = this;
  while (scope) {
    // If we see a function scope, remember it but see if we are nested in some
    // other function.
    if (isa<LIT::FuncOp>(*scope) ||
        (!isLifetime && isa<StructDeclOp>(*scope))) {
      if (!innermostFuncScope)
        innermostFuncScope = scope;
      outermostFuncScope = scope;
    }

    // If we found the file module, then we're at the top level.
    if (isa<FileModuleOp>(*scope)) {
      // If we haven't found the either the inner- or outermost scope yet, we
      // need to set them. This can happen, for example, with top level aliases.
      if (!outermostFuncScope)
        outermostFuncScope = scope;
      if (!innermostFuncScope)
        innermostFuncScope = scope;
      break;
    }

    scope = scope->getParentDecl();
  }
  assert(outermostFuncScope && "couldn't find an enclosing function");

  // If this is a declaration in the outermost function, then we don't need to
  // unique it - there are no other names it could conflict with.
  MLIRContext *ctx = outermostFuncScope->getContext();
  if (innermostFuncScope == outermostFuncScope && dontRenameOutermost)
    return StringAttr::get(ctx, name + (isLifetime ? "`" : ""));

  return StringAttr::get(ctx, name + "`" +
                                  Twine(outermostFuncScope->getNextUniqueID()));
}

StringAttr ASTDecl::getAnonymousLifetimeFor(const Twine &valueName,
                                            bool dontRenameOutermost) {
  return getUniqueParamName(valueName, /*isLifetime=*/true,
                            dontRenameOutermost);
}

void ASTDecl::dump() const {
  // The value is either an operation or a type of MLIR `Value`.
  TypeSwitch<DeclIRValue>(getIRValue())
      .Case<Operation *>([](Operation *op) {
        // Print without verifying, since IR could be in an invalid state.
        op->print(llvm::errs(), mlir::OpPrintingFlags().printGenericOpForm());
        llvm::errs() << "\n";
      })
      .Case<PValue, SRValue, MRValue, SBValue, MBValue, MLValue>(
          [](auto v) { v.dump(); })
      .Default([](DeclIRValue v) { llvm::errs() << "<null decl>\n"; });
}

MLIRContext *ASTDecl::getContext() const {
  if (auto *op = getIfOperation())
    return op->getContext();
  if (auto mv = dyn_cast<PValue>(irValue))
    return mv.get().getContext();
  if (auto dr = dyn_cast<SRValue>(irValue))
    return dr.getContext();
  if (auto value = dyn_cast_or_null<MLValue>(irValue))
    return value.getContext();
  return cast<MRValue>(irValue).getContext();
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

std::optional<StringRef> ASTDecl::getNameIfOperation() const {
  if (Operation *op = getIfOperation())
    if (auto decl = dyn_cast<ASTDeclInterface>(op))
      return decl.getDeclName().getValue();
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
  return ParamRefType::get(
      ParamDeclRefAttr::get(traitOp.getParamsAttr().back()));
}
