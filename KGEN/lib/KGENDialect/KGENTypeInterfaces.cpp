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
  TypedAttr expr = getDType();
  if (!expr)
    return DType::invalid;
  auto constant = expr.dyn_cast<DTypeConstantAttr>();
  if (!constant)
    return DType::invalid;
  return constant.getDType();
}

#include "KGEN/KGENDialect/KGENTypeInterfaces.cpp.inc"
