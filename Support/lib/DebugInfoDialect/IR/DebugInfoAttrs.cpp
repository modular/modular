//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/BinaryFormat/Dwarf.h"

using namespace M;
using namespace M::DebugInfo;

//===----------------------------------------------------------------------===//
// DebugInfoDialect
//===----------------------------------------------------------------------===//

void DebugInfoDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "Support/DebugInfoDialect/IR/DebugInfoEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.cpp.inc"

//===----------------------------------------------------------------------===//
// DIAttr
//===----------------------------------------------------------------------===//

bool DIAttr::classof(Attribute attr) {
  return llvm::isa<DebugInfoDialect>(attr.getDialect());
}

//===----------------------------------------------------------------------===//
// DIScopeAttr
//===----------------------------------------------------------------------===//

bool DIScopeAttr::classof(Attribute attr) {
  return llvm::isa<DICompileUnitAttr, DIFileAttr, DILocalScopeAttr>(attr);
}

//===----------------------------------------------------------------------===//
// DILocalScopeAttr
//===----------------------------------------------------------------------===//

bool DILocalScopeAttr::classof(Attribute attr) {
  return llvm::isa<DILexicalBlockAttr, DISubprogramAttr>(attr);
}
