//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Compiler/DenseArray.h"
#include "mlir/IR/DialectImplementation.h"

using namespace M;

// Pack the integer values into a byte array;
static std::vector<char> packIntegerValues(unsigned width,
                                           ArrayRef<APInt> values) {
  unsigned byteSize = llvm::divideCeil(width, CHAR_BIT);
  std::vector<char> rawData(values.size() * byteSize, 0);
  auto *ptr = (uint8_t *)rawData.data();
  for (auto &it : llvm::enumerate(values))
    llvm::StoreIntToMemory(it.value(), ptr + (it.index() * byteSize), byteSize);
  return rawData;
}

//===----------------------------------------------------------------------===//
// DenseIntArrayAttr
//===----------------------------------------------------------------------===//

DenseIntArrayAttr DenseIntArrayAttr::get(IntegerType type,
                                         ArrayRef<APInt> values) {
  std::vector<char> rawData = packIntegerValues(type.getWidth(), values);
  return DenseArrayAttr::get(RankedTensorType::get(values.size(), type),
                             rawData)
      .cast<DenseIntArrayAttr>();
}

APInt DenseIntArrayAttr::Iterator::operator*() const {
  unsigned byteWidth = llvm::divideCeil(type.getWidth(), CHAR_BIT);
  APInt value(type.getWidth(), 0, !type.isUnsigned());
  llvm::LoadIntFromMemory(value, (const uint8_t *)base + index * byteWidth,
                          byteWidth);
  return value;
}

auto DenseIntArrayAttr::begin() const -> Iterator {
  return Iterator(getElementType().cast<IntegerType>(), getRawData().data(), 0);
}

auto DenseIntArrayAttr::end() const -> Iterator {
  return Iterator(getElementType().cast<IntegerType>(), getRawData().data(),
                  size());
}

bool DenseIntArrayAttr::classof(Attribute attr) {
  if (auto arr = attr.dyn_cast<DenseArrayAttr>())
    return arr.getElementType().isa<IntegerType>();
  return false;
}

//===----------------------------------------------------------------------===//
// custom<DenseIntArray>
//===----------------------------------------------------------------------===//

ParseResult M::parseDenseIntArray(AsmParser &p, DenseIntArrayAttr &result,
                                  unsigned width,
                                  IntegerType::SignednessSemantics signedness) {
  auto elementType = IntegerType::get(p.getContext(), width, signedness);
  APInt value;
  mlir::OptionalParseResult maybeEmpty = p.parseOptionalInteger(value);
  // Check for an empty array.
  if (!maybeEmpty.has_value()) {
    result = DenseIntArrayAttr::get(elementType, {});
    return success();
  }
  if (maybeEmpty.value())
    return failure();

  SmallVector<APInt> values;
  auto addValue = [&](const APInt &value) {
    if (elementType.isUnsigned())
      values.push_back(value.zextOrTrunc(elementType.getWidth()));
    else
      values.push_back(value.sextOrTrunc(elementType.getWidth()));
  };
  addValue(value);

  while (succeeded(p.parseOptionalComma())) {
    if (p.parseInteger(value))
      return failure();
    addValue(value);
  }
  result = DenseIntArrayAttr::get(elementType, values);
  return success();
}

void M::printDenseIntArray(AsmPrinter &p, Operation *op,
                           DenseIntArrayAttr result, unsigned width,
                           IntegerType::SignednessSemantics) {
  llvm::interleaveComma(result, p);
}

//===----------------------------------------------------------------------===//
// DenseFloatArrayAttr
//===----------------------------------------------------------------------===//

DenseFloatArrayAttr DenseFloatArrayAttr::get(FloatType type,
                                             ArrayRef<APFloat> values) {
  SmallVector<APInt> intVals;
  intVals.reserve(values.size());
  for (const APFloat &value : values)
    intVals.push_back(value.bitcastToAPInt());
  std::vector<char> rawData = packIntegerValues(type.getWidth(), intVals);
  return DenseArrayAttr::get(RankedTensorType::get(values.size(), type),
                             rawData)
      .cast<DenseFloatArrayAttr>();
}

APFloat DenseFloatArrayAttr::Iterator::operator*() const {
  FloatType type = this->type;
  unsigned byteWidth = llvm::divideCeil(type.getWidth(), CHAR_BIT);
  APInt intVal(type.getWidth(), 0);
  llvm::LoadIntFromMemory(intVal, (const uint8_t *)base + index * byteWidth,
                          byteWidth);
  return APFloat(type.getFloatSemantics(), intVal);
}

auto DenseFloatArrayAttr::begin() const -> Iterator {
  return Iterator(getElementType().cast<FloatType>(), getRawData().data(), 0);
}

auto DenseFloatArrayAttr::end() const -> Iterator {
  return Iterator(getElementType().cast<FloatType>(), getRawData().data(),
                  size());
}

bool DenseFloatArrayAttr::classof(Attribute attr) {
  if (auto arr = attr.dyn_cast<DenseArrayAttr>())
    return arr.getElementType().isa<FloatType>();
  return false;
}
