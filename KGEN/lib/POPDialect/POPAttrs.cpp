//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/STLExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
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
// DTypeValue
//===----------------------------------------------------------------------===//

/// Returns true if this is a supported float dtype.
static bool isValidFloatDType(DType dtype) {
  return dtype.isFloat() &&
         !(dtype == DType::f8 || dtype == DType::f24 || dtype == DType::tf32);
}

/// Returns the floating semantics for the given dtype.
static const llvm::fltSemantics &getFloatSemantics(DType dtype) {
  switch (dtype.getValue()) {
  case DType::f16:
    return APFloat::IEEEhalf();
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
  case DType::tf32:
  default:
    llvm_unreachable("unknown float dtype");
  }
}

DTypeValue::DTypeValue(APSInt value, DType dtype)
    : DTypeValue(APInt(value), dtype) {
  assert(dtype.isInt() && value.getBitWidth() == dtype.getIntegerWidthInBits());
}

DTypeValue::DTypeValue(APFloat value, DType dtype)
    : DTypeValue(value.bitcastToAPInt(), dtype) {
  assert(isValidFloatDType(dtype));
}

DTypeValue::DTypeValue(bool value, DType dtype)
    : DTypeValue(APInt(1, value), dtype) {
  assert(dtype.isBool());
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

namespace M::KGEN::POP {
/// Provide the ability to hash values for attribute uniquing.
inline llvm::hash_code hash_value(const DTypeValue &value) {
  return hash_combine(value.getData(), value.getDType().getValue());
}
} // namespace M::KGEN::POP

/// Parse a value of a particular DType.
static FailureOr<DTypeValue> parseDTypeValue(AsmParser &p, DType dtype) {
  llvm::SMLoc loc = p.getCurrentLocation();

  // Handle integers.
  if (dtype.isInt()) {
    APInt apInt;
    if (p.parseInteger(apInt))
      return failure();
    APSInt apsInt(apInt, /*isUnsigned=*/dtype.isUInt());
    APSInt fitted = apsInt.extOrTrunc(dtype.getIntegerWidthInBits());
    if (fitted.extOrTrunc(apsInt.getBitWidth()) != apsInt) {
      SmallVector<char, 256> strVal;
      apsInt.toString(strVal);
      return p.emitError(loc, "integer value doesn't fit into ")
             << dtype.getIntegerWidthInBits()
             << " bits: " << StringRef(strVal.data(), strVal.size());
    }
    return DTypeValue(fitted, dtype);
  }

  // Handle floats.
  if (dtype.isFloat()) {
    // Parse the float value as a string. MLIR doesn't expose access to raw
    // float literals. Make sure we can parse and print arbitrary precision
    // floats losslessly.
    std::string strVal;
    if (p.parseString(&strVal))
      return failure();
    APFloat apFp(getFloatSemantics(dtype));
    llvm::Expected<APFloat::opStatus> status =
        apFp.convertFromString(strVal, APFloat::rmNearestTiesToEven);
    if (llvm::errorToBool(status.takeError()))
      return p.emitError(loc, "failed to parse floating point value");
    if (*status != APFloat::opOK)
      return p.emitError(loc, "cannot convert ")
             << strVal << " to " << dtype.getAsString();
    return DTypeValue(apFp, dtype);
  }

  // Handle bools.
  assert(dtype.isBool());
  if (succeeded(p.parseOptionalKeyword("true")))
    return {{true, dtype}};
  if (succeeded(p.parseOptionalKeyword("false")))
    return {{false, dtype}};
  return p.emitError(loc, "expected 'true' or 'false' for bool literal");
}

LogicalResult SIMDAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                               ArrayRef<DTypeValue> values, SIMDType type) {
  Optional<KGENDType> dtype = type.getResolvedDType();
  Optional<int64_t> size = type.getResolvedSize();
  if (!dtype || !size)
    return emitError() << "SIMD attribute requires fully-resolved SIMD type";
  if (static_cast<int64_t>(values.size()) != *size)
    return emitError() << "wrong number of elements, got " << values.size()
                       << " but expected " << *size;
  if (!llvm::all_of(values, [&](const DTypeValue &value) {
        return value.getDType() == *dtype;
      }))
    return emitError() << "all elements must have dtype "
                       << dtype->getAsString();
  return success();
}

/// Value is a constant by definition.
bool SIMDAttr::isConstant() const { return true; }

//===----------------------------------------------------------------------===//
// custom<DTypeValues>
//===----------------------------------------------------------------------===//

static ParseResult parseDTypeValues(AsmParser &p,
                                    FailureOr<SmallVector<DTypeValue>> &values,
                                    SIMDType type) {
  Optional<KGENDType> dtype = type.getResolvedDType();
  Optional<int64_t> size = type.getResolvedSize();
  if (!dtype || !size) {
    return p.emitError(p.getCurrentLocation(),
                       "SIMD constant requires a concrete type");
  }
  if (!dtype->isInt() && !isValidFloatDType(*dtype) && !dtype->isBool()) {
    return p.emitError(
        p.getCurrentLocation(),
        "only integer, float, and bool dtype constants can be parsed");
  }

  values.emplace();
  auto parseElt = [&](int64_t) -> ParseResult {
    FailureOr<DTypeValue> value = parseDTypeValue(p, *dtype);
    if (failed(value))
      return failure();
    values->push_back(*value);
    return success();
  };
  return failableInterleave(llvm::seq<int64_t>(0, *size), parseElt,
                            [&] { return p.parseComma(); });
}

static void printDTypeValues(AsmPrinter &p, ArrayRef<DTypeValue> values,
                             SIMDType type) {
  DType dtype = *type.getResolvedDType();
  auto printElt = [&](const DTypeValue &value) {
    if (dtype.isInt()) {
      p << value.getIntVal();
    } else if (dtype.isFloat()) {
      SmallVector<char, 256> strVal;
      value.getFloatVal().toString(strVal);
      p << '"' << StringRef(strVal.data(), strVal.size()) << '"';
    } else {
      p << (value.getBoolVal() ? "true" : "false");
    }
  };
  llvm::interleaveComma(values, p, printElt);
}

//===----------------------------------------------------------------------===//
// ArrayAttr
//===----------------------------------------------------------------------===//

static ParseResult parseArrayElements(AsmParser &p,
                                      FailureOr<SmallVector<TypedAttr>> &values,
                                      POP::ArrayType type) {
  Optional<int64_t> size = type.getResolvedSize();
  Type elementType = type.getResolvedElementType();
  if (!size || !elementType)
    return p.emitError(p.getCurrentLocation(),
                       "array attribute expected a fully-resolved array type");
  values.emplace();
  return failableInterleave(
      llvm::seq<unsigned>(0, *size),
      [&](unsigned) {
        return parseParamValue(p, values->emplace_back(), elementType);
      },
      [&] { return p.parseComma(); });
}

static void printArrayElements(AsmPrinter &p, ArrayRef<TypedAttr> values,
                               POP::ArrayType type) {
  llvm::interleaveComma(values, p, [&](TypedAttr value) {
    printParamValue(value, p.getStream());
  });
}

/// The array attribute is a constant if all element values are constants.
bool POP::ArrayAttr::isConstant() const {
  return llvm::all_of(getValues(), ParameterAttr::isSimpleConstant);
}

LogicalResult
POP::ArrayAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                       ArrayRef<TypedAttr> values, ArrayType type) {
  Optional<int64_t> size = type.getResolvedSize();
  Type elementType = type.getResolvedElementType();
  if (!size || !elementType)
    return emitError()
           << "array attribute expected a fully-resolved array type";
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
// StructAttr
//===----------------------------------------------------------------------===//

static ParseResult
parseStructElements(AsmParser &p, FailureOr<SmallVector<TypedAttr>> &values,
                    StructType type) {
  SmallVector<Type> types;
  if (failed(type.resolveElementTypes(types)))
    return p.emitError(
        p.getCurrentLocation(),
        "struct attribute expected a fully-resolved struct type");

  values.emplace();
  return failableInterleave(
      types,
      [&](Type type) {
        return parseParamValue(p, values->emplace_back(), type);
      },
      [&] { return p.parseComma(); });
}

static void printStructElements(AsmPrinter &p, ArrayRef<TypedAttr> values,
                                StructType type) {
  llvm::interleaveComma(values, p, [&](TypedAttr value) {
    printParamValue(value, p.getStream());
  });
}

/// The struct attribute is a constant if all element values are constants.
bool POP::StructAttr::isConstant() const {
  return llvm::all_of(getValues(), ParameterAttr::isSimpleConstant);
}

LogicalResult
POP::StructAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                        ArrayRef<TypedAttr> values, StructType type) {
  SmallVector<Type> types;
  if (failed(type.resolveElementTypes(types)))
    return emitError()
           << "struct attribute expected a fully-resolved struct type";
  if (types.size() != values.size())
    return emitError() << "struct attribute type requires " << types.size()
                       << " elements but value has " << values.size();
  for (auto [idx, value, type] :
       llvm::zip(llvm::seq<unsigned>(0, types.size()), values, types))
    if (value.getType() != type)
      return emitError() << "struct element #" << idx << " has type "
                         << value.getType() << " but expected " << type;
  return success();
}

//===----------------------------------------------------------------------===//
// VariantAttr
//===----------------------------------------------------------------------===//

LogicalResult VariantAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                  TypedAttr value, VariantType type) {
  if (type.getTypeIndex(value.getType()))
    return success();
  return emitError() << "variant attribute value type " << value.getType()
                     << " is not a possible variant subtype";
}

/// The variant attribute is a constant if the value type is a constant.
bool VariantAttr::isConstant() const {
  return ParameterAttr::isSimpleConstant(getValue());
}

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "KGEN/POPDialect/POPEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "KGEN/POPDialect/POPAttrs.cpp.inc"
