//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/DebugInfoDialect/IR/DebugInfoTypes.h"
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/BinaryFormat/Dwarf.h"

using namespace M;
using namespace M::DebugInfo;

//===----------------------------------------------------------------------===//
// DebugInfoDialect
//===----------------------------------------------------------------------===//

void DebugInfoDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Support/DebugInfoDialect/IR/DebugInfoTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// DIBasicType
//===----------------------------------------------------------------------===//

DIBasicType DIBasicBoolType::get(MLIRContext *ctx, const Twine &name,
                                 uint64_t sizeInBits, uint32_t alignInBits) {
  return DIBasicType::get(ctx, StringAttr::get(ctx, name), sizeInBits,
                          alignInBits, llvm::dwarf::DW_ATE_boolean);
}

DIBasicType DIBasicUIntType::get(MLIRContext *ctx, const Twine &name,
                                 uint64_t sizeInBits, uint32_t alignInBits) {
  return DIBasicType::get(ctx, StringAttr::get(ctx, name), sizeInBits,
                          alignInBits, llvm::dwarf::DW_ATE_unsigned);
}

DIBasicType DIBasicSIntType::get(MLIRContext *ctx, const Twine &name,
                                 uint64_t sizeInBits, uint32_t alignInBits) {
  return DIBasicType::get(ctx, StringAttr::get(ctx, name), sizeInBits,
                          alignInBits, llvm::dwarf::DW_ATE_signed);
}

DIBasicType DIBasicFloatType::get(MLIRContext *ctx, const Twine &name,
                                  uint64_t sizeInBits, uint32_t alignInBits) {
  return DIBasicType::get(ctx, StringAttr::get(ctx, name), sizeInBits,
                          alignInBits, llvm::dwarf::DW_ATE_float);
}

//===----------------------------------------------------------------------===//
// AsmFormat Utilities
//===----------------------------------------------------------------------===//

static LogicalResult parseName(AsmParser &p, FailureOr<StringAttr> &result) {
  std::string name;
  if (failed(p.parseKeywordOrString(&name)))
    return failure();
  result = p.getBuilder().getStringAttr(name);
  return success();
}

static void printName(AsmPrinter &p, StringAttr name) {
  p.printKeywordOrString(name);
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "Support/DebugInfoDialect/IR/DebugInfoTypes.cpp.inc"
