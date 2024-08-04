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
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

DocStringAttr ASTDecl::getDocString() const {
  if (auto astDeclOp = dyn_cast<ASTDeclInterface>(this))
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
  assert(resolvedness == DeclResolvedness::fully &&
         "cannot perform lookup in a decl that isn't fully resolved");
  auto it = declsInScope.find(name);
  if (it != declsInScope.end() && !it->second.empty())
    return it->second;
  return {};
}

void ASTDecl::takeDecls(ASTDecl &src) {
  if (src.isErroneous())
    setErroneous();
  for (auto &[name, children] : src.declsInScope)
    for (ASTDecl *child : children)
      child->parentDecl = this;
  declsInScope = std::move(src.declsInScope);
  counter = src.counter;
}

/// Return the nearest parameter scope (i.e. DeclInterface) for the given decl,
/// as well as the total depth from the nearest file module.
static std::pair<ASTDecl *, size_t> getNearestParamScopeAndDepth(
    ASTDecl *decl, function_ref<void(const ASTDecl *)> checkForCollision) {
  ASTDecl *paramScope = nullptr;
  size_t depth = 0;
  while (decl) {
    checkForCollision(decl);

    if (isa<DeclInterface>(*decl)) {
      ++depth;
      if (!paramScope)
        paramScope = decl;
      if (isa<FileModuleOp>(*decl))
        break;
    }

    decl = decl->getParentDecl();
  }

  return {paramScope, --depth}; // Adjust so depth starts at 0.
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
  TypeSwitch<DeclIRValue>(getIRValue())
      .Case<Operation *>([](Operation *op) {
        // Print without verifying, since IR could be in an invalid state.
        op->print(llvm::errs(), mlir::OpPrintingFlags().printGenericOpForm());
        llvm::errs() << "\n";
      })
      .Case<PValue, SRValue, MRValue, SBValue, MBValue, MLValue, MBPValue>(
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
RValue ASTDecl::getIfRValue() const {
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
  if (auto value = dyn_cast_or_null<MBPValue>(irValue))
    return value;
  return {};
}

/// If this is a LValue, return it, otherwise return null.
LValue ASTDecl::getIfLValue() const {
  if (auto mlValue = dyn_cast_or_null<MLValue>(irValue))
    return mlValue;

  if (auto storage = dyn_cast_or_null<RCRef<BaseDLValue>>(irValue))
    return DLValue(storage);
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
  // The last parameter to the trait is the 'T' parameter which (when everything
  // gets instantiated) resolves to the final type the trait is instantiated on.
  return ASTType(ParamDeclRefAttr::get(traitOp.getParamsAttr().back()));
}
