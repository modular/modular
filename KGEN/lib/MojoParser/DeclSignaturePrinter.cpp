//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Renders Mojo-syntax signatures directly from MLIR ops. This is the canonical
// implementation used by both the compiler diagnostic path and the mojo-doc
// tool (which delegates here via `PublicASTDecl`). The data model and rendering
// primitives live in `SignatureModel.h`; this file just wires the
// per-op-kind entry points together.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser/DeclSignaturePrinter.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/LITDialect/SpecialFunctions.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/ASTType.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "KGEN/MojoParser/MojoDiags.h"
#include "KGEN/MojoParser/SignatureModel.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// Public entry points
//===----------------------------------------------------------------------===//

void M::KGEN::printFunctionSignature(LIT::FnOp fnOp, LIT::SharedState &shared,
                                     llvm::raw_string_ostream &os,
                                     const LIT::ASTDecl *contextDecl,
                                     const SignatureOffsets &offsets) {
  // The DeclResolver context-changer only records the decl; it does not
  // mutate it. Const-cast lets us thread it through from `const`-qualified
  // doc-tooling callers.
  DeclResolver::DiagnosticDeclContextChanger scope(
      const_cast<LIT::ASTDecl *>(contextDecl));

  bool isStatic = fnOp.getIsStatic();
  bool isMethod = !isStatic && isa<StructDeclOp>(fnOp->getParentOp());
  bool isInit = fnOp.getSpecialFunctionInfo().isInitializer();
  FnTypeGeneratorType signature = fnOp.getFuncTypeGenerator();

  // Self-type substitution for `Self` keyword rendering. `Self` can be uttered
  // by static methods too (e.g. in a return type), so the substitution is
  // gated on the enclosing decl being a struct - not on `isMethod`.
  std::optional<ASTType> selfType;
  if (auto parentStruct = dyn_cast<StructDeclOp>(fnOp->getParentOp()))
    selfType = ASTType(ASTDecl::computeSelfTypeForStruct(parentStruct));

  SmallVector<ParameterInfo, 2> params;
  ParameterEvaluator evaluator =
      populateParameterInfos(shared, signature.getInputParamTypes(),
                             signature.getParamListAttrs(), params, selfType);

  // Function-level constraints (the trailing "where ...").
  std::string fnConstraints;
  if (auto cs = signature.getMetadata().getBodyConstraints(); !cs.empty())
    fnConstraints = mergeConformsToConstraints(cs, &evaluator, shared, params);

  SmallVector<ArgumentInfo, 2> args;
  populateArgumentInfos(
      shared, signature, fnOp.getArgumentTypes(), selfType, evaluator,
      [&] { return fnOp.getSpecialFunctionInfo().hasSelfResult(); }, args);

  // Pre-render the return type for the shared printer; suppressed entirely
  // for `__init__`-style functions whose out-arg gets hoisted to the front.
  bool hasOutArgument =
      !args.empty() && args.back().convention == ArgumentConvention::kOut;
  std::string returnTypeStr;
  ASTType resultType = signature.getUserResultType();
  if (!hasOutArgument && resultType && !resultType.isNoneType()) {
    std::optional<ArgConvention> convention;
    if (signature.isRefResult()) {
      convention = ArgConvention::Ref;
      returnTypeStr =
          "ref" + getRefPrefixAsString(shared, cast<RefType>(resultType),
                                       signature, /*isRefResult=*/true);
    }
    Type reboundUserResultType =
        evaluator.getReboundType(fnOp.getUserResultType());
    returnTypeStr +=
        generateTypeString(shared, reboundUserResultType, VariadicKind::None,
                           selfType, convention);
  }

  // Strip the "(...)" mangle suffix from a function source name, leaving just
  // the bare identifier.
  auto stripFunctionMangle = [](StringRef name) {
    return name.split('(').first;
  };

  printFunctionSignatureFromInfos(
      stripFunctionMangle(fnOp.getSourceName().value_or(StringRef())), args,
      params, returnTypeStr, fnConstraints, isInit, isMethod, shared, os,
      offsets);
}

void M::KGEN::printStructSignature(LIT::StructDeclOp structOp,
                                   LIT::SharedState &shared,
                                   llvm::raw_string_ostream &os,
                                   const LIT::ASTDecl *contextDecl,
                                   const SignatureOffsets &offsets) {
  DeclResolver::DiagnosticDeclContextChanger scope(
      const_cast<LIT::ASTDecl *>(contextDecl));
  TypeSignatureType signature = structOp.getSignature();
  SmallVector<ParameterInfo, 2> params;
  ParameterEvaluator evaluator =
      populateParameterInfos(shared, signature.getInputParamTypes(),
                             signature.getParamListAttrs(), params);

  // Struct-level constraints (the trailing "where ...").
  std::string constraints;
  if (auto cs = signature.getParamListAttrs().getBodyConstraints(); !cs.empty())
    constraints = mergeConformsToConstraints(cs, &evaluator, shared, params);

  printStructSignatureFromInfos(structOp.getName(), params, constraints, shared,
                                os, offsets);
}

void M::KGEN::printAliasSignature(LIT::AliasDeclOp aliasOp,
                                  LIT::SharedState &shared,
                                  llvm::raw_string_ostream &os,
                                  const LIT::ASTDecl *contextDecl,
                                  const SignatureOffsets &offsets) {
  DeclResolver::DiagnosticDeclContextChanger scope(
      const_cast<LIT::ASTDecl *>(contextDecl));
  auto maybeValue = aliasOp.getValue();
  if (!maybeValue)
    return;
  auto generator = dyn_cast<GeneratorAttr>(*maybeValue);
  if (!generator)
    return;
  auto generatorType = dyn_cast<GeneratorType>(generator.getType());
  if (!generatorType)
    return;

  SmallVector<ParameterInfo, 2> params;
  populateParameterInfos(shared, generatorType.getInputParamTypes(),
                         generatorType.getParamListAttrs(), params);

  auto name = demangleParameterName(aliasOp.getName(), /*forUser=*/true);
  printAliasSignatureFromInfos(name, /*type=*/"", params, shared, os, offsets);
}

//===----------------------------------------------------------------------===//
// Diagnostic-oriented helpers
//===----------------------------------------------------------------------===//

std::string M::KGEN::synthesizeDeclSignature(Operation *op, SharedState &shared,
                                             const ASTDecl *contextDecl) {
  if (!op)
    return {};
  std::string out;
  llvm::raw_string_ostream os(out);
  llvm::TypeSwitch<Operation *>(op)
      .Case<FnOp>([&](FnOp fnOp) {
        printFunctionSignature(fnOp, shared, os, contextDecl);
      })
      .Case<StructDeclOp>([&](StructDeclOp structOp) {
        printStructSignature(structOp, shared, os, contextDecl);
      })
      .Case<AliasDeclOp>([&](AliasDeclOp aliasOp) {
        printAliasSignature(aliasOp, shared, os, contextDecl);
      })
      .Default([](Operation *) {});
  return out;
}

bool M::KGEN::hasReadableSourceLocation(Location loc, SharedState &shared) {
  auto &diags = shared.diags;
  auto &sourceMgr = diags.sourceMgr;
  return sourceMgr.FindBufferContainingLoc(diags.convertLocToSMLoc(loc)) != 0;
}

MojoInflightDiag &
M::KGEN::synthesizeToDiagIfLocUnreadable(MojoInflightDiag &diag, Operation *op,
                                         StringRef preamble,
                                         const LIT::ASTDecl *contextDecl) {
  auto *shared = diag.getSharedIfActive();
  if (!shared)
    return diag;
  if (hasReadableSourceLocation(diag.getLastLoc(), *shared))
    return diag;

  std::string sig = synthesizeDeclSignature(op, *shared, contextDecl);

  if (!sig.empty())
    diag << preamble << sig;

  return diag;
}

MojoInflightDiag &
M::KGEN::synthesizeToDiagIfLocUnreadable(MojoInflightDiag &diag, TypedAttr attr,
                                         StringRef preamble) {
  auto *shared = diag.getSharedIfActive();
  if (!shared)
    return diag;
  if (hasReadableSourceLocation(diag.getLastLoc(), *shared))
    return diag;

  std::string out;
  llvm::raw_string_ostream os(out);
  ASTType::printParam(os, attr, /*forDiag=*/shared);

  if (!out.empty())
    diag << preamble << "'" << out << "'";

  return diag;
}
