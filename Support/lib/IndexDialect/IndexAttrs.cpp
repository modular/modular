//===- IndexAttrs.cpp -----------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/IndexDialect/IndexAttrs.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M::index;

//===----------------------------------------------------------------------===//
// IndexDialect
//===----------------------------------------------------------------------===//

void IndexDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Support/IndexDialect/IndexAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "Support/IndexDialect/IndexEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Support/IndexDialect/IndexAttrs.cpp.inc"
