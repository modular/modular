//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/STLExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Error.h"

using namespace M;
using namespace KGEN;
using namespace POP;

//===----------------------------------------------------------------------===//
// POPDialect
//===----------------------------------------------------------------------===//

void POPDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "KGEN/POPDialect/POPAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// UnionAttr
//===----------------------------------------------------------------------===//

LogicalResult UnionAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                TypedAttr value, UnionType type) {
  auto it = llvm::find(type.getTypes(), value.getType());
  if (it != type.getTypes().end())
    return success();
  return emitError() << "value type " << value.getType()
                     << " is not a union element type of " << type;
}

//===----------------------------------------------------------------------===//
// DTypeValue
//===----------------------------------------------------------------------===//

bool DTypeValue::isValidFloatDType(KGENDType dtype) {
  return dtype.isFloat() &&
         !(dtype == DType::f8 || dtype == DType::f24 || dtype == DType::tf32);
}

const llvm::fltSemantics &DTypeValue::getFloatSemantics(KGENDType dtype) {
  switch (dtype.getValue()) {
  case DType::f16:
    return APFloat::IEEEhalf();
  case DType::tf32:
    return APFloat::FloatTF32();
  case DType::f32:
    return APFloat::IEEEsingle();
  case DType::f64:
    return APFloat::IEEEdouble();
  case DType::f128:
    return APFloat::IEEEquad();
  case DType::bf16:
    return APFloat::BFloat();
  case DType::f80:
    return APFloat::x87DoubleExtended();

  case DType::f8:
  case DType::f24:
  default:
    llvm_unreachable("unknown float dtype");
  }
}

DTypeValue::DTypeValue(APInt data, KGENDType dtype)
    : data(std::move(data)), dtype(dtype) {
  assert(dtype.isAddress() || dtype.isIndex() ||
         this->data.getBitWidth() == dtype.getWidthInBits());
}

DTypeValue::DTypeValue(APSInt value, KGENDType dtype)
    : DTypeValue(APInt(std::move(value)), dtype) {
  assert(dtype.isInt());
}

DTypeValue::DTypeValue(APFloat value, KGENDType dtype)
    : DTypeValue(value.bitcastToAPInt(), dtype) {
  assert(isValidFloatDType(dtype));
}

DTypeValue::DTypeValue(bool value, KGENDType dtype)
    : DTypeValue(APInt(8, value), dtype) {
  assert(dtype.isBool());
}

DTypeValue::DTypeValue(int64_t value, KGENDType dtype)
    : DTypeValue(APInt(64, value), dtype) {
  assert(dtype.isIndex() || dtype.isAddress());
}

APSInt DTypeValue::getIntVal() const {
  assert(dtype.isInt());
  return APSInt(data, /*isUnsigned=*/dtype.isUInt());
}

APFloat DTypeValue::getFloatVal() const {
  assert(isValidFloatDType(dtype));
  return APFloat(getFloatSemantics(dtype), data);
}

bool DTypeValue::getBoolVal() const {
  assert(dtype.isBool());
  return data.isOne();
}

int64_t DTypeValue::getIndexVal() const {
  assert(dtype.isIndex() || dtype.isAddress());
  return data.getSExtValue();
}

namespace M::KGEN::POP {
/// Provide the ability to hash values for attribute uniquing.
inline llvm::hash_code hash_value(const DTypeValue &value) {
  return hash_combine(value.getData(), value.getDType().getValue());
}
} // namespace M::KGEN::POP

/// Parse a value of a particular DType. If `optionalParse` is set and no valid
/// token was found, then `std::nullopt` is returned. Otherwise, the parser
/// emits and error and returns `failure`.
template <bool optionalParse>
static std::conditional_t<optionalParse, std::optional<FailureOr<DTypeValue>>,
                          FailureOr<DTypeValue>>
parseDTypeValue(AsmParser &p, KGENDType dtype) {
  llvm::SMLoc loc = p.getCurrentLocation();

  // Handle integers.
  if (dtype.isInt()) {
    APInt apInt;
    if constexpr (optionalParse) {
      OptionalParseResult result = p.parseOptionalInteger(apInt);
      if (!result.has_value())
        return std::nullopt;
      if (failed(*result))
        return failure();
    } else {
      if (p.parseInteger(apInt))
        return failure();
    }
    APSInt apsInt(std::move(apInt), /*isUnsigned=*/dtype.isUInt());
    APSInt fitted = apsInt.extOrTrunc(dtype.getIntegerWidthInBits());
    if (fitted.extOrTrunc(apsInt.getBitWidth()) != apsInt) {
      SmallVector<char, 256> strVal;
      apsInt.toString(strVal);
      p.emitError(loc, "integer value doesn't fit into ")
          << dtype.getIntegerWidthInBits()
          << " bits: " << StringRef(strVal.data(), strVal.size());
      return failure();
    }
    return DTypeValue(std::move(fitted), dtype);
  }

  // Handle floats.
  if (dtype.isFloat()) {
    // Parse the float value as a string. MLIR doesn't expose access to raw
    // float literals. Make sure we can parse and print arbitrary precision
    // floats losslessly.
    std::string strVal;
    if constexpr (optionalParse) {
      if (p.parseOptionalString(&strVal))
        return std::nullopt;
    } else {
      if (p.parseString(&strVal))
        return failure();
    }
    APFloat apFp(DTypeValue::getFloatSemantics(dtype));
    llvm::Expected<APFloat::opStatus> status =
        apFp.convertFromString(strVal, APFloat::rmNearestTiesToEven);
    if (llvm::errorToBool(status.takeError())) {
      p.emitError(loc, "failed to parse floating point value");
      return failure();
    }
    if (*status != APFloat::opOK && !(*status & APFloat::opInexact)) {
      SmallVector<char> c;
      apFp.toString(c);
      p.emitError(loc, "cannot convert ")
          << strVal << " to " << dtype.getAsString() << ": got "
          << StringRef(c.data(), c.size());
      return failure();
    }
    return DTypeValue(apFp, dtype);
  }

  // Handle bools.
  if (dtype.isBool()) {
    if (succeeded(p.parseOptionalKeyword("true")))
      return DTypeValue(true, dtype);
    if (succeeded(p.parseOptionalKeyword("false")))
      return DTypeValue(false, dtype);
    if constexpr (optionalParse)
      return std::nullopt;
    else
      return p.emitError(loc, "expected 'true' or 'false' for bool literal");
  }

  // Handle indices.
  assert(dtype.isIndex() || dtype.isAddress());
  int64_t indexVal;
  if constexpr (optionalParse) {
    OptionalParseResult result = p.parseOptionalInteger(indexVal);
    if (!result.has_value())
      return std::nullopt;
    if (failed(*result))
      return failure();
  } else {
    if (p.parseInteger(indexVal))
      return failure();
  }
  return DTypeValue(indexVal, dtype);
}

//===----------------------------------------------------------------------===//
// SIMDAttrStorage / ODS Boilerplate
//===----------------------------------------------------------------------===//

namespace M::KGEN::POP::detail {
/// Custom storage class that allocates and owns the `DTypeValue` instances in
/// an `OwningArrayRef`, because they are not POD.
struct SIMDAttrStorage : public mlir::AttributeStorage {
  using KeyTy = std::tuple<ArrayRef<DTypeValue>, SIMDType>;
  SIMDAttrStorage(llvm::OwningArrayRef<DTypeValue> values, SIMDType type)
      : values(std::move(values)), type(type) {}

  KeyTy getAsKey() const { return KeyTy(values, type); }
  bool operator==(const KeyTy &key) const {
    return std::tie(values, type) == key;
  }
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(std::get<0>(key), std::get<1>(key));
  }
  static SIMDAttrStorage *construct(mlir::AttributeStorageAllocator &allocator,
                                    KeyTy &&key) {
    return new (allocator.allocate<SIMDAttrStorage>())
        SIMDAttrStorage(std::get<0>(key), std::get<1>(key));
  }

  llvm::OwningArrayRef<DTypeValue> values;
  SIMDType type;
};
} // namespace M::KGEN::POP::detail

ArrayRef<DTypeValue> SIMDAttr::getValues() const { return getImpl()->values; }
SIMDType SIMDAttr::getType() const { return getImpl()->type; }

//===----------------------------------------------------------------------===//
// SIMDAttr
//===----------------------------------------------------------------------===//

LogicalResult SIMDAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                               ArrayRef<DTypeValue> values, SIMDType type) {
  std::optional<KGENDType> dtype = type.getResolvedDType();
  std::optional<int64_t> size = type.getResolvedSize();
  if (!dtype || !size)
    return emitError() << "SIMD attribute requires fully-resolved SIMD type";
  if (static_cast<int64_t>(values.size()) != *size)
    return emitError() << "wrong number of elements, got " << values.size()
                       << " but expected " << *size;
  if (!llvm::all_of(values, [&](const DTypeValue &value) {
        return value.getDType() == *dtype;
      }))
    return emitError() << "all elements must have dtype "
                       << dtype->getAsString() << " but the first element is "
                       << values[0].getDType().getAsString();
  return success();
}

/// Value is a constant by definition.
bool SIMDAttr::isConstant() const { return true; }

SIMDAttr SIMDAttr::get(uint64_t intVal, SIMDType type) {
  DType dtype = *type.getResolvedDType();
  APInt apVal(dtype.getIntegerWidthInBits(), intVal);
  APSInt apsVal(std::move(apVal), /*isUnsigned=*/dtype.isUInt());
  POP::DTypeValue scalarVal(std::move(apsVal), dtype);
  return SIMDAttr::get(scalarVal, type);
}

//===----------------------------------------------------------------------===//
// custom<DTypeValues>
//===----------------------------------------------------------------------===//

/// Parse a list of dtype values.
static ParseResult parseDTypeValues(AsmParser &p,
                                    SmallVector<DTypeValue> &values,
                                    KGENDType dtype, int64_t size) {
  auto parseElt = [&](int64_t) -> ParseResult {
    FailureOr<DTypeValue> value =
        parseDTypeValue</*optionalParse=*/false>(p, dtype);
    if (failed(value))
      return failure();
    values.push_back(*value);
    return success();
  };
  return failableInterleave(llvm::seq<int64_t>(0, size), parseElt,
                            [&] { return p.parseComma(); });
}

/// Print a single DType value.
static void printDTypeValue(AsmPrinter &p, const DTypeValue &value,
                            KGENDType dtype) {
  if (dtype.isInt()) {
    p << value.getIntVal();
  } else if (dtype.isFloat()) {
    SmallString<256> strVal;
    value.getFloatVal().toString(strVal);
    p << '"' << StringRef(strVal.data(), strVal.size()) << '"';
  } else if (dtype.isBool()) {
    p << (value.getBoolVal() ? "true" : "false");
  } else {
    assert(dtype.isIndex() || dtype.isAddress());
    p << value.getIndexVal();
  }
}

static void printDTypeValues(AsmPrinter &p, ArrayRef<DTypeValue> values,
                             SIMDType type) {
  KGENDType dtype = *type.getResolvedDType();
  llvm::interleaveComma(values, p, [&](const DTypeValue &value) {
    printDTypeValue(p, value, dtype);
  });
}

/// Parse a splat or a list of values deliminated as by `<` and `>`. If
/// `optionalParse` is set and no valid tokens were found, then `std::nullopt`
/// is returned, otherwise failure.
template <bool optionalParse>
static std::conditional_t<optionalParse, OptionalParseResult, ParseResult>
parseSplatOrVector(AsmParser &p, SmallVector<DTypeValue> &values,
                   SIMDType type) {
  std::optional<KGENDType> dtype = type.getResolvedDType();
  std::optional<int64_t> size = type.getResolvedSize();
  auto checkDType = [&]() -> ParseResult {
    if (dtype->isInt() || DTypeValue::isValidFloatDType(*dtype) ||
        dtype->isBool() || dtype->isIndex() || dtype->isAddress())
      return success();
    return p.emitError(
        p.getCurrentLocation(),
        "only integer, float, bool, and index dtype constants can be parsed");
  };

  // If the size and dtype are both known, try to parse a splat value.
  if (dtype && size) {
    if (checkDType())
      return failure();
    std::optional<FailureOr<DTypeValue>> splat =
        parseDTypeValue</*optionalParse=*/true>(p, *dtype);
    if (splat.has_value()) {
      if (failed(*splat))
        return failure();
      values.assign(*size, **splat);
      return mlir::success();
    }
  }

  // Otherwise, look for the opening brace.
  if constexpr (optionalParse) {
    if (p.parseOptionalLess())
      return std::nullopt;
  } else {
    if (p.parseLess())
      return failure();
  }

  // Make sure the SIMD type is concrete.
  if (!dtype || !size)
    return p.emitError(p.getCurrentLocation(),
                       "SIMD constant requires a concrete type");
  if (checkDType())
    return failure();

  if (failed(parseDTypeValues(p, values, *dtype, *size)))
    return failure();
  return p.parseGreater();
}

/// Print the values of a SIMD vector as either a splat if they're all the same
/// or a list of values deliminated by `<` and `>`.
template <bool inAttr>
static void printSplatOrVector(AsmPrinter &p, ArrayRef<DTypeValue> values,
                               SIMDType type) {
  // Check if all values are equal, in which case we print as a splat.
  if (!values.empty() && llvm::all_equal(values)) {
    // Make sure to add a space after the attribute mnemonic.
    if constexpr (inAttr)
      p << ' ';
    printDTypeValue(p, values.front(), values.front().getDType());
    return;
  }

  p << '<';
  printDTypeValues(p, values, type);
  p << '>';
}

/// Custom directive parse hook for SIMDAttr assembly format.
static ParseResult
parseSIMDValues(AsmParser &p, SmallVector<DTypeValue> &values, SIMDType type) {
  return parseSplatOrVector</*optionalParse=*/false>(p, values, type);
}

/// Custom directive print hook for SIMDAttr assembly format.
static void printSIMDValues(AsmPrinter &p, ArrayRef<DTypeValue> values,
                            SIMDType type) {
  printSplatOrVector</*inAttr=*/true>(p, values, type);
}

OptionalParseResult SIMDType::parseValue(AsmParser &p, TypedAttr &value) const {
  SmallVector<DTypeValue> values;
  OptionalParseResult result =
      parseSplatOrVector</*optionalParse=*/true>(p, values, *this);
  if (result.has_value() && succeeded(*result))
    value = SIMDAttr::get(values, *this);
  return result;
}

LogicalResult SIMDType::printValue(AsmPrinter &p, TypedAttr value) const {
  auto simd = ::dyn_cast<SIMDAttr>(value);
  if (!simd)
    return failure();

  printSplatOrVector</*inAttr=*/false>(p, simd.getValues(), *this);
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ArrayAttr
//===----------------------------------------------------------------------===//
OptionalParseResult POP::ArrayType::parseValue(AsmParser &p,
                                               TypedAttr &value) const {
  if (failed(p.parseOptionalLSquare()))
    return std::nullopt;
  if (!getResolvedSize())
    return p.emitError(p.getCurrentLocation(),
                       "array attribute expected a concrete size");
  if (succeeded(p.parseOptionalRSquare())) {
    value = POP::ArrayAttr::get({}, *this);
    return mlir::success();
  }
  SmallVector<TypedAttr> values;
  if (failed(parseSequenceElements(p, values, *this)))
    return failure();
  value = POP::ArrayAttr::get(values, *this);
  return p.parseRSquare();
}

LogicalResult POP::ArrayType::printValue(AsmPrinter &p, TypedAttr value) const {
  auto array = ::dyn_cast<POP::ArrayAttr>(value);
  if (!array)
    return failure();
  p << '[';
  llvm::interleaveComma(array.getValues(), p,
                        [&](TypedAttr value) { printParamValue(p, value); });
  p << ']';
  return mlir::success();
}

/// The array attribute is a constant if all element values are constants.
bool POP::ArrayAttr::isConstant() const {
  return llvm::all_of(getValues(), ParameterAttr::isSimpleConstant);
}

LogicalResult
POP::ArrayAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                       ArrayRef<TypedAttr> values, ArrayType type) {
  std::optional<int64_t> size = type.getResolvedSize();
  if (!size)
    return emitError() << "array attribute expected a concrete size";
  Type elementType = type.getElementType();
  if (*size != static_cast<int64_t>(values.size()))
    return emitError() << "array attribute type requires " << *size
                       << " elements but value has " << values.size();
  for (auto [idx, value] : llvm::enumerate(values))
    if (value.getType() != elementType)
      return emitError() << "array element #" << idx << " has type "
                         << value.getType() << " but expected " << elementType;
  return success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "KGEN/POPDialect/POPEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "KGEN/POPDialect/POPAttrs.cpp.inc"
