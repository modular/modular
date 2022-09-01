//===- IndexOps.cpp -------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/IndexDialect/IndexOps.h"
#include "Support/IndexDialect/IndexAttrs.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace M;
using namespace M::index;

//===----------------------------------------------------------------------===//
// IndexDialect
//===----------------------------------------------------------------------===//

void IndexDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "Support/IndexDialect/Index.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// CastSOp
//===----------------------------------------------------------------------===//

bool CastSOp::areCastCompatible(TypeRange lhsTypes, TypeRange rhsTypes) {
  return lhsTypes.front().isa<IndexType>() != rhsTypes.front().isa<IndexType>();
}

//===----------------------------------------------------------------------===//
// CastUOp
//===----------------------------------------------------------------------===//

bool CastUOp::areCastCompatible(TypeRange lhsTypes, TypeRange rhsTypes) {
  return lhsTypes.front().isa<IndexType>() != rhsTypes.front().isa<IndexType>();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Support/IndexDialect/Index.cpp.inc"
