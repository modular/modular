//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CODialect/COTypes.h"
#include "KGEN/CODialect/CODialect.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
using namespace CO;

//===----------------------------------------------------------------------===//
// CoroutineType
//===----------------------------------------------------------------------===//

std::optional<int64_t> CoroutineType::getTypeSize(TargetInfoAttr target) const {
  return target.getDataLayout().getPointerSize();
}

std::optional<int64_t>
CoroutineType::getTypeAlign(TargetInfoAttr target) const {
  return target.getDataLayout().getPointerABIAlign();
}

CoroutineType CoroutineType::get(MLIRContext *ctx, TypeRange resultTypes,
                                 bool raises) {
  auto coroSig = SignatureType::get(FunctionType::get(ctx, {}, resultTypes), {},
                                    {}, {}, FnEffects().setThrows(raises));
  return CoroutineType::get(ctx, coroSig);
}

//===----------------------------------------------------------------------===//
// CODialect
//===----------------------------------------------------------------------===//

void CODialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "KGEN/CODialect/COTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "KGEN/CODialect/COTypes.cpp.inc"
