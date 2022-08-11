//===- POPAttrs.cpp -------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "KGEN/POPDialect/POPEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "KGEN/POPDialect/POPAttrs.cpp.inc"

using namespace M;
using namespace KGEN;

void POPDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "KGEN/POPDialect/POPAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// CmpPredicateAttr
//===----------------------------------------------------------------------===//

void CmpPredicateAttr::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrs,
    function_ref<void(Type)> walkTypes) const {}

Attribute
CmpPredicateAttr::replaceImmediateSubElements(ArrayRef<Attribute> replAttrs,
                                              ArrayRef<Type> replTypes) const {
  return *this;
}
