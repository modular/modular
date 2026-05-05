//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements common utilities shared by the parser implementation.
//
//===----------------------------------------------------------------------===//

#include "MojoUtils.h"

#include "IREmitter.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/ExprNode.h"
#include "KGEN/MojoParser/MojoDiags.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "ParamInf.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

TypedAttr LIT::getOriginsAccessibleByParams(PogListAttr paramList,
                                            ArrayRef<ParamDeclAttr> params,
                                            SharedState &shared,
                                            TypedAttr captureOrigins) {
  // Implicit parameters are not accessible on the callee side, so we don't
  // consider their origin accesses.
  params = params.drop_back(countNumImplicitKinds(paramList));

  // We also need to find all accessible origin sets, even if they are
  // parametric, and union them with the found origins. We don't need to
  // recurse into any nested parameter origins. Even if they contain origin
  // set references, they may not be within the top-level parameter scope, and
  // also we know they can't be accessed by the current function. For example,
  //
  //   def foo[f: def[g: def() capturing [_] -> None] -> None]():
  //       pass
  //
  // `foo` doesn't access the inner origin set of `g` through `f`, because
  // `foo` cannot call `f` without constructing and passing a closure.
  //
  // We can union the sets together by wrapping them in a origin set union.
  // The mutability doesn't matter since it will get flattened.
  SmallVector<TypedAttr> origins;
  auto addOriginSet = [&](TypedAttr param) {
    origins.push_back(OriginSetUnionAttr::get(
        param, OriginType::get(shared.getContext(), /*isMutable=*/true)));
  };

  for (ParamDeclAttr param : params)
    if (sugarIsa<OriginSetType>(param.getType()))
      addOriginSet(ParamDeclRefAttr::get(param));
  if (captureOrigins)
    addOriginSet(captureOrigins);

  return OriginSetAttr::get(shared.getContext(), origins);
}

ASTType LIT::getBoundCoroutineType(ASTDecl &declScope, const ExprNode *expr,
                                   FnTypeGeneratorType sig, TypedAttr origin) {
  auto &shared = declScope.getShared();
  SMLoc loc = expr->getLoc();
  ASTDecl *decl = sig.isThrows() ? shared.getBuiltinRaisingCoroutineType(loc)
                                 : shared.getBuiltinCoroutineType(loc);
  if (!decl) {
    shared.emitError(loc,
                     "internal error: could not find builtin 'Coroutine' type");
    return {};
  }
  ASTType resultType = ASTType(sig.getUserResultType());
  ParamBindings paramBinds(declScope, expr);
  paramBinds.add(expr, PValue(resultType));
  paramBinds.add(expr, origin);

  auto structOp = cast<StructDeclOp>(decl->getIfOperation());
  TypeSignatureType structSig = structOp.getSignature();
  ParamInf inference(paramBinds, structSig.getParamTypes(),
                     structSig.getParamListAttrs(),
                     /*allowImplicitConversions=*/true, decl,
                     /*discardError=*/false);
  ParameterExprArrayAttr bindings = inference.inferForStruct();

  if (!bindings)
    return {};
  return structOp.bindReference(bindings);
}

TypedAttr LIT::computeArgumentsOrigin(AsyncCallOp call,
                                      CachedOriginFinder &originFinder) {
  SmallVector<std::pair<Value, OperandEffect>> operands;
  SmallVector<ResultEffect> results;
  SmallVector<TypedAttr> origins;
  LIT::getOperationEffects(*call, operands, results, origins, originFinder);
  for (Value value : call.getOperands())
    if (auto ref = dyn_cast<RefType>(value.getType()))
      origins.push_back(ref.getOrigin());
  return OriginSetAttr::get(call.getContext(), origins);
}

CValue LIT::materializeAsyncCallAsCoroutine(IREmitter &emitter,
                                            AsyncCallOp call,
                                            const ExprNode *expr,
                                            FnTypeGeneratorType sig,
                                            ExprDest &dest) {
  ASTType coroutineType = getBoundCoroutineType(
      emitter.getDeclScope(), expr, sig,
      computeArgumentsOrigin(call, emitter.shared.cachedOriginFinder));
  if (!coroutineType) {
    dest.resetForError(emitter);
    return {};
  }

  return emitter.emitConstructorCall(
      coroutineType, CallOperands(CallSyntax::kImplicitConvert, expr,
                                  std::move(dest), {{SRValue(call), expr}}));
}

void LIT::markRegionUnreachable(Region *deadRegion, Location unreachableLoc) {
  // Erase bottom up to avoid deleting an op while something uses its results.
  for (Operation &op :
       llvm::make_early_inc_range(llvm::reverse(deadRegion->front()))) {
    // Avoid erasing ops that correspond to lazily resolved decls.
    if (isa<ImportOp, UnresolvedImportOp, UnresolvedWildcardImportOp>(op))
      continue;
    op.erase();
  }

  auto builder = OpBuilder::atBlockEnd(&deadRegion->front());
  UnreachableOp::create(builder, unreachableLoc);
}

//===----------------------------------------------------------------------===//
// Diagnostic utilities
//===----------------------------------------------------------------------===//

void LIT::emitWrongArgOrParamCount(MojoInflightDiag &diag, size_t minRequired,
                                   size_t maxAllowed, size_t numActual,
                                   const Twine &argOrParam) {
  diag << " expects ";

  // Tailor the diagnostic if the exact number of expected args is known.
  if (minRequired == maxAllowed && numActual != minRequired) {
    diag << minRequired << " " << argOrParam << plural(minRequired);
  } else if (numActual < minRequired) {
    diag << "at least " << minRequired << " " << argOrParam
         << plural(minRequired);
  } else {
    assert(numActual > maxAllowed);
    diag << "at most " << maxAllowed << " " << argOrParam << plural(maxAllowed);
  }

  diag << ", but " << numActual << plural(numActual, " was", " were")
       << " specified";
}

/// Emit a comma separated list of names, each in '...'.
static void emitNames(MojoInflightDiag &diag, ArrayRef<StringAttr> names) {
  llvm::interleave(
      names, [&](StringAttr str) { diag << str; }, [&]() { diag << ", "; });
}

void LIT::emitUnknownKeywords(MojoInflightDiag &diag,
                              ArrayRef<StringAttr> unknownKeywords,
                              StringRef argOrParam) {
  diag << "unknown keyword " << argOrParam << plural(unknownKeywords.size())
       << ": ";
  emitNames(diag, unknownKeywords);
}

void LIT::emitPosOnlyPassedByKw(MojoInflightDiag &diag,
                                ArrayRef<StringAttr> names,
                                StringRef argOrParam) {
  size_t numNames = names.size();
  diag << "positional-only " << argOrParam << plural(numNames)
       << " passed as keyword operand" << plural(numNames) << ": ";
  emitNames(diag, names);
}

void LIT::emitMissing(MojoInflightDiag &diag, ArrayRef<StringAttr> names,
                      const Twine &kindStr) {
  size_t numNames = names.size();
  diag << "missing " << numNames << " required " << kindStr << plural(numNames)
       << ": ";
  emitNames(diag, names);
}

void LIT::emitByPosAndKw(MojoInflightDiag &diag, ArrayRef<StringAttr> names,
                         const Twine &kindStr) {
  size_t numNames = names.size();
  diag << kindStr << plural(numNames)
       << " passed both as positional and keyword operand: ";
  emitNames(diag, names);
}

bool LIT::isInternalName(StringRef name) { return name.starts_with('_'); }
