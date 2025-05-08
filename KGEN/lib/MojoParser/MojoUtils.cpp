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

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/POPDialect/POPTypes.h"

#include "Support/Compiler/Diags.h"

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

  SmallVector<Type> types;
  for (ParamDeclAttr param : params)
    types.push_back(param.getType());
  SmallVector<TypedAttr> origins =
      shared.cachedOriginFinder.findOriginsIn(types);

  // We also need to find all accessible origin sets, even if they are
  // parametric, and union them with the found origins. We don't need to
  // recurse into any nested parameter origins. Even if they contain origin
  // set references, they may not be within the top-level parameter scope, and
  // also we know they can't be accessed by the current function. For example,
  //
  //   fn foo[f: fn[g: fn() capturing [_] -> None] -> None]():
  //       pass
  //
  // `foo` doesn't access the inner origin set of `g` through `f`, because
  // `foo` cannot call `f` without constructing and passing a closure.
  //
  // We can union the sets together by wrapping them in a origin set union.
  // The mutability doesn't matter since it will get flattened.
  auto addOriginSet = [&](TypedAttr param) {
    origins.push_back(OriginSetUnionAttr::get(
        param, OriginType::get(shared.getContext(), /*isMutable=*/true)));
  };

  for (ParamDeclAttr param : params)
    if (isa<OriginSetType>(param.getType()))
      addOriginSet(ParamDeclRefAttr::get(param));
  if (captureOrigins)
    addOriginSet(captureOrigins);

  return OriginSetAttr::get(shared.getContext(), origins);
}

void LIT::markRegionUnreachable(Region *deadRegion, Location unreachableLoc) {
  // Erase bottom up to avoid deleting an op while something uses its results.
  for (Operation &op :
       llvm::make_early_inc_range(llvm::reverse(deadRegion->front()))) {
    // Avoid erasing ops that correspond to lazily resolved decls.
    if (isa<UnresolvedImportOp, UnresolvedWildcardImportOp>(op))
      continue;
    op.erase();
  }

  OpBuilder::atBlockEnd(&deadRegion->front())
      .create<UnreachableOp>(unreachableLoc);
}

ASTType LIT::getFunctionArgumentRValueType(ASTType type, ArgConvention conv) {
  if (conv == ArgConvention::ReadReg)
    return type;
  else // Everything else is passed through memory.
    return type.getReferenceElementType();
}

//===----------------------------------------------------------------------===//
// Diagnostic utilities
//===----------------------------------------------------------------------===//

void LIT::emitWrongArgOrParamCount(InflightDiag &diag, size_t minRequired,
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
static void emitNames(InflightDiag &diag, ArrayRef<StringAttr> names) {
  llvm::interleave(
      names, [&](StringAttr str) { diag << str; }, [&]() { diag << ", "; });
}

void LIT::emitUnknownKeywords(InflightDiag &diag,
                              ArrayRef<StringAttr> unknownKeywords,
                              StringRef argOrParam) {
  diag << "unknown keyword " << argOrParam << plural(unknownKeywords.size())
       << ": ";
  emitNames(diag, unknownKeywords);
}

void LIT::emitPosOnlyPassedByKw(InflightDiag &diag, ArrayRef<StringAttr> names,
                                StringRef argOrParam) {
  size_t numNames = names.size();
  diag << "positional-only " << argOrParam << plural(numNames)
       << " passed as keyword operand" << plural(numNames) << ": ";
  emitNames(diag, names);
}

void LIT::emitOutOfOrderInferredKw(InflightDiag &diag,
                                   ArrayRef<StringAttr> names) {
  size_t numNames = names.size();
  diag << "inferred parameter" << plural(numNames) << " passed out of order: ";
  emitNames(diag, names);
}

void LIT::emitMissing(InflightDiag &diag, ArrayRef<StringAttr> names,
                      const Twine &kindStr) {
  size_t numNames = names.size();
  diag << "missing " << numNames << " required " << kindStr << plural(numNames)
       << ": ";
  emitNames(diag, names);
}

void LIT::emitByPosAndKw(InflightDiag &diag, ArrayRef<StringAttr> names,
                         const Twine &kindStr) {
  size_t numNames = names.size();
  diag << kindStr << plural(numNames)
       << " passed both as positional and keyword operand: ";
  emitNames(diag, names);
}

void LIT::emitTooManyPositional(InflightDiag &diag, size_t numMaxAllowed,
                                size_t numActual, const Twine &kindStr) {
  diag << "expected at most " << numMaxAllowed << " positional " << kindStr
       << plural(numMaxAllowed) << ", got " << numActual;
}

std::string LIT::nameForPosOnly(size_t idx, const Twine &argOrParam) {
  return ("positional-only " + argOrParam + " #" + Twine(idx)).str();
}

void LIT::printNameOrIdx(StringAttr name, size_t idx, InflightDiag &diag) {
  if (!name.empty())
    diag << "'" << name.getValue() << "'";
  else
    diag << "#" << idx;
}

void LIT::emitModuleCallSubscriptDiag(InflightDiag &diag,
                                      StructMetaType metaType,
                                      const Twine &callOrSubscript, SMLoc loc,
                                      SharedState &shared) {
  StringAttr name = metaType.getSymbol().getLeafReference();
  diag << "module " << name << " is not " << callOrSubscript << "able";

  LookupResult lookupResult = shared.lookupAndResolveDecl(
      name, loc, metaType.getType(), /*searchParentScopes=*/false);
  if (ArrayRef<ASTDecl *> resDecls = lookupResult.getIfSuccess();
      !resDecls.empty()) {
    diag << "; did you mean to " << callOrSubscript << ' ' << name.strref()
         << '.' << name.strref() << '?';
  }
}
