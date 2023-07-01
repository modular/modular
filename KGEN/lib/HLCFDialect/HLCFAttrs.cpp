//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/HLCFDialect/HLCFAttrs.h"
#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Error.h"

using namespace M;
using namespace HLCF;

//===----------------------------------------------------------------------===//
// POPDialect
//===----------------------------------------------------------------------===//

void HLCFDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "KGEN/HLCFDialect/HLCFAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "KGEN/HLCFDialect/HLCFEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "KGEN/HLCFDialect/HLCFAttrs.cpp.inc"
