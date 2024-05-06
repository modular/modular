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
// custom<DIType>
//===----------------------------------------------------------------------===//

static void printDIType(AsmPrinter &printer, DIType type) {
  // Strip unresolved types during printing.
  if (auto diMLIRTy = dyn_cast<DIUnresolvedMLIRType>(type))
    printer.printType(diMLIRTy.getType());
  else
    printer.printType(type);
}

static ParseResult parseDIType(AsmParser &parser, DIType &type) {
  Type rawType;
  if (failed(parser.parseType(rawType)))
    return failure();

  // If the type is not a DI type, wrap it as an unresolved type.
  if (!(type = dyn_cast<DIType>(rawType)))
    type = DIUnresolvedMLIRType::get(rawType);
  return success();
}

static void printDITypes(AsmPrinter &printer, ArrayRef<DIType> types) {
  printer << "(";
  llvm::interleaveComma(types, printer,
                        [&](DIType type) { printDIType(printer, type); });
  printer << ")";
}

static ParseResult parseDITypes(AsmParser &parser,
                                SmallVectorImpl<DIType> &types) {
  return parser.parseCommaSeparatedList(AsmParser::Delimiter::Paren, [&]() {
    return parseDIType(parser, types.emplace_back());
  });
}

//===----------------------------------------------------------------------===//
// DIType
//===----------------------------------------------------------------------===//

uint64_t DIType::getSizeInBits() const {
  if (auto arrayType = mlir::dyn_cast<DIArrayType>(*this)) {
    return arrayType.getElementCount() *
           arrayType.getElementType().getSizeInBits();
  }
  if (auto basicType = mlir::dyn_cast<DIBasicType>(*this))
    return basicType.getSizeInBits();
  if (auto memberType = mlir::dyn_cast<DIMemberType>(*this))
    return memberType.getType().getSizeInBits();
  if (auto pointerType = mlir::dyn_cast<DIPointerType>(*this))
    return pointerType.getSizeInBits();
  if (auto structType = mlir::dyn_cast<DIStructType>(*this))
    return structType.getSizeInBits();
  if (auto variantType = mlir::dyn_cast<DIVariantType>(*this))
    return variantType.getSizeInBits();
  if (auto vectorType = mlir::dyn_cast<DIVectorType>(*this)) {
    return vectorType.getElementCount() *
           vectorType.getElementType().getSizeInBits();
  }
  return 0;
}

uint32_t DIType::getAlignInBits() const {
  if (auto arrayType = mlir::dyn_cast<DIArrayType>(*this))
    return arrayType.getElementType().getAlignInBits();
  if (auto basicType = mlir::dyn_cast<DIBasicType>(*this))
    return basicType.getAlignInBits();
  if (auto memberType = mlir::dyn_cast<DIMemberType>(*this))
    return memberType.getType().getAlignInBits();
  if (auto pointerType = mlir::dyn_cast<DIPointerType>(*this))
    return pointerType.getAlignInBits();
  if (auto structType = mlir::dyn_cast<DIStructType>(*this))
    return structType.getAlignInBits();
  if (auto variantType = mlir::dyn_cast<DIVariantType>(*this))
    return variantType.getAlignInBits();
  if (auto vectorType = mlir::dyn_cast<DIVectorType>(*this))
    return vectorType.getElementType().getAlignInBits();
  return 0;
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
// DIStructType
//===----------------------------------------------------------------------===//

uint64_t DIStructType::getSizeInBits() const {
  uint64_t structSize = 0;
  uint32_t structAlign = 1;
  for (DIMemberType member : getMembers()) {
    uint32_t memberAlign = member.getAlignInBits();
    if (!memberAlign)
      return 0;

    structSize =
        llvm::alignTo(structSize + member.getSizeInBits(), memberAlign);
    structAlign = std::max(structAlign, memberAlign);
  }

  // Pad the struct size to the largest element alignment.
  if (structAlign)
    structSize = llvm::alignTo(structSize, structAlign);
  return structSize;
}

uint32_t DIStructType::getAlignInBits() const {
  // The alignment is the max alignment of any of the members.
  uint32_t align = 1;
  for (DIMemberType member : getMembers()) {
    uint32_t memberAlign = member.getAlignInBits();
    if (!memberAlign)
      return 0;
    align = std::max(align, memberAlign);
  }
  return align;
}

//===----------------------------------------------------------------------===//
// DIUnresolvedMLIRType
//===----------------------------------------------------------------------===//

LogicalResult
DIUnresolvedMLIRType::verify(function_ref<InFlightDiagnostic()> emitError,
                             Type type) {
  if (::isa<DIUnresolvedMLIRType>(type))
    return emitError() << "should not wrap unresolved type in another: "
                       << type;
  return success();
}

//===----------------------------------------------------------------------===//
// ASMFormat Utilities
//===----------------------------------------------------------------------===//

static LogicalResult parseName(AsmParser &p, StringAttr &result) {
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
