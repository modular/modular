//===- KGENTypeInterfaces.cpp ---------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENTypeInterfaces.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "Support/DType.h"

using namespace M;
using namespace KGEN;

DType DataTypeInterface::resolveDType() {
  if (auto constant = getDType().dyn_cast_or_null<DTypeConstantAttr>())
    return constant.getDType();
  return DType::invalid;
}

#include "KGEN/KGENDialect/KGENTypeInterfaces.cpp.inc"
