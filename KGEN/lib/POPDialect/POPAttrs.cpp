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
#include "llvm/Support/Base64.h"
#include "llvm/Support/Compression.h"
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

DTypeValue::DTypeValue(APInt data, KGENDType dtype)
    : data(std::move(data)), dtype(dtype) {
  assert(dtype.isAddress() || dtype.isIndex() ||
         this->data.getBitWidth() == dtype.getWidthInBits());
}

DTypeValue::DTypeValue(APSInt value, KGENDType dtype)
    : DTypeValue(APInt(std::move(value)), dtype) {}

DTypeValue::DTypeValue(APFloat value, KGENDType dtype)
    : DTypeValue(value.bitcastToAPInt(), dtype) {
  assert(dtype.getFloatSemantics());
}

DTypeValue::DTypeValue(bool value, KGENDType dtype)
    : DTypeValue(APInt(8, value), dtype) {
  assert(dtype.isBool());
}

DTypeValue::DTypeValue(int64_t value, KGENDType dtype)
    : DTypeValue(APInt(64, value), dtype) {}

APSInt DTypeValue::getIntVal() const {
  assert(dtype.isIntLike());
  return APSInt(data, /*isUnsigned=*/dtype.isUInt());
}

APFloat DTypeValue::getFloatVal() const {
  auto *sem = dtype.getFloatSemantics();
  assert(sem && "not a float type");
  return APFloat(*sem, data);
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
    auto *semantics = dtype.getFloatSemantics();
    if (!semantics) {
      p.emitError(loc, "unknown float semantics for type");
      return failure();
    }

    APFloat apFp(*semantics);
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

/// Create a "zero" value initialized to 0, 0.0, false, etc.  NOTE: This can
/// return null if the SIMD type is parametric or we don't know how to make a
/// zero value of the specified dtype.
SIMDAttr SIMDAttr::getZeroValue(SIMDType type) {
  auto optDT = type.getResolvedDType();
  auto optSize = type.getResolvedSize();
  if (!optDT || !optSize)
    return {};
  auto dtype = optDT.value();

  std::optional<DTypeValue> zeroValue;
  if (dtype.isBool()) {
    zeroValue = DTypeValue(false, dtype);
  } else if (dtype.isInt()) {
    APSInt aps(dtype.getIntegerWidthInBits(), /*isUnsigned=*/dtype.isUInt());
    zeroValue = DTypeValue(aps, dtype);
  } else if (auto *sem = dtype.getFloatSemantics()) {
    zeroValue = DTypeValue(APFloat::getZero(*sem), dtype);
  }

  if (!zeroValue)
    return {};

  SmallVector<DTypeValue> elements(optSize.value(), zeroValue.value());
  return SIMDAttr::get(elements, type);
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

/// Parse a splat or a list of values delimited as by `<` and `>`. If
/// `optionalParse` is set and no valid tokens were found, then `std::nullopt`
/// is returned, otherwise failure.
template <bool optionalParse>
static std::conditional_t<optionalParse, OptionalParseResult, ParseResult>
parseSplatOrVector(AsmParser &p, SmallVector<DTypeValue> &values,
                   SIMDType type) {
  std::optional<KGENDType> dtype = type.getResolvedDType();
  std::optional<int64_t> size = type.getResolvedSize();
  auto checkDType = [&]() -> ParseResult {
    if (dtype->isInt() || dtype->getFloatSemantics() || dtype->isBool() ||
        dtype->isIndex() || dtype->isAddress())
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
/// or a list of values delimited by `<` and `>`.
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
// IntLiteralAttr
//===----------------------------------------------------------------------===//

static ParseResult parseIntLiteral(AsmParser &p, IPInt &result) {
  APInt resultAP;
  OptionalParseResult parseResult = p.parseInteger(resultAP);
  if (!parseResult.has_value() || failed(*parseResult)) {
    result = {};
    return failure();
  }
  result = IPInt(resultAP);
  return success();
}

static void printIntLiteral(AsmPrinter &p, const IPInt &value) {
  p.getStream() << value;
}

//===----------------------------------------------------------------------===//
// IntLiteralAttr
//===----------------------------------------------------------------------===//

Type IntLiteralAttr::getType() const {
  return IntLiteralType::get(this->getContext());
}

bool IntLiteralAttr::isConstant() const { return true; }

//===----------------------------------------------------------------------===//
// FloatLiteralAttr
//===----------------------------------------------------------------------===//

Attribute FloatLiteralAttr::parse(AsmParser &p, Type type) {
  if (p.parseLess())
    p.emitError(p.getCurrentLocation(), "expected '<' character");
  std::optional<IPRational> rational;
  FloatLiteralSpecialValuesAttr specialAttr;

  // Try to parse rational number first, then fall back to parsing special
  // value.
  APInt numerator;
  OptionalParseResult isRational = p.parseOptionalInteger(numerator);
  if (isRational.has_value() && !isRational.value()) {
    // MLIR's AsmParser doesn't have `parseSlash` or a more generic way to parse
    // literal strings/characters, so we will use the pipe "|" character
    // instead. https://github.com/modularml/modular/issues/23387
    APInt denominator;
    if (p.parseVerticalBar() || p.parseInteger(denominator)) {
      p.emitError(p.getCurrentLocation(),
                  "expected rational number with pipe in parens");
      return {};
    }
    if (denominator == 0) {
      p.emitError(p.getCurrentLocation(),
                  "expected rational number with non-zero denominator");
    }
    rational = IPRational(numerator, denominator);
    specialAttr = FloatLiteralSpecialValuesAttr::get(
        p.getContext(), FloatLiteralSpecialValues::Normal);
  } else if (p.parseCustomAttributeWithFallback(specialAttr)) {
    p.emitError(p.getCurrentLocation(),
                "expected FloatLiteralSpecialValueAttr");
    return {};
  }

  if (p.parseGreater()) {
    p.emitError(p.getCurrentLocation(), "expected '>' character");
  }

  return FloatLiteralAttr::get(p.getContext(), specialAttr, rational);
}

void FloatLiteralAttr::print(AsmPrinter &p) const {
  p.getStream() << "<";
  if (getSpecial().getValue() == FloatLiteralSpecialValues::Normal) {
    assert(getRational().has_value() &&
           "rational has value when special value is normal");
    p.getStream() << getRational().value();
  } else {
    p.getStream() << getSpecial().getValue();
  }
  p.getStream() << ">";
}

Type FloatLiteralAttr::getType() const {
  return FloatLiteralType::get(this->getContext());
}

bool FloatLiteralAttr::isConstant() const { return true; }

FloatLiteralAttr FloatLiteralAttr::get(MLIRContext *context,
                                       FloatLiteralSpecialValuesAttr input,
                                       std::optional<IPRational> value) {
  // Canonicalize special attributes to have no value.
  if (input.getValue() != FloatLiteralSpecialValues::Normal)
    value = {};
  return Base::get(context, input, value);
}

FloatLiteralAttr FloatLiteralAttr::get(MLIRContext *context, IPRational value) {
  return get(context,
             FloatLiteralSpecialValuesAttr::get(
                 context, FloatLiteralSpecialValues::Normal),
             value);
}

//===----------------------------------------------------------------------===//
// IntLiteralConvertAttr
//===----------------------------------------------------------------------===//

static ErrorOr<IntegerAttr> foldIntLiteralConvert(TypedAttr input, Type outType,
                                                  bool treatIndexAsUnsigned) {
  auto literal = ::dyn_cast<IntLiteralAttr>(input);
  if (!literal)
    return Error("input must be IntLiteralAttr");

  const IPInt &invalIP = literal.getValue();
  const APInt &invalAP = invalIP.getAPInt();
  unsigned outWidth = 64;
  bool isUnsigned = treatIndexAsUnsigned;
  if (!outType.isIndex()) {
    outWidth = outType.getIntOrFloatBitWidth();
    isUnsigned = outType.isUnsignedInteger();
  }
  if (invalIP < 0 && isUnsigned) {
    std::string msg;
    llvm::raw_string_ostream msgStream(msg);
    msgStream << "integer value " << invalIP
              << " is negative, but is being converted to an unsigned type";
    return Error(msgStream.str());
  }
  uint64_t effectiveInputWidth = invalAP.getBitWidth();
  // Positive IPInts are stored with an extra leading zero.  If converting to an
  // unsigned type, we can strip the leading zero.
  if (isUnsigned)
    effectiveInputWidth -= 1;
  if (effectiveInputWidth > outWidth) {
    std::string msg;
    llvm::raw_string_ostream msgStream(msg);
    msgStream << "integer value " << invalIP << " requires "
              << effectiveInputWidth
              << " bits to store, but the destination bit width is only "
              << outWidth << " bits wide";
    return Error(msgStream.str());
  }

  APInt result;
  if (isUnsigned)
    result = invalAP.zextOrTrunc(outWidth);
  else
    result = invalAP.sextOrTrunc(outWidth);
  return IntegerAttr::get(outType, result);
}

TypedAttr IntLiteralConvertAttr::get(MLIRContext *ctx, Type type,
                                     TypedAttr input,
                                     bool treatIndexAsUnsigned) {
  // If this is a literal constant coming in, we can fold this.  If not, stage
  // it until elaboration or something else simplifies things.
  auto result = foldIntLiteralConvert(input, type, treatIndexAsUnsigned);
  if (!result.isError())
    return result.get();

  return Base::get(ctx, type, input, treatIndexAsUnsigned);
}

bool IntLiteralConvertAttr::isConstant() const { return false; }

ErrorOrSuccess IntLiteralConvertAttr::validateForElaborator() const {
  auto result =
      foldIntLiteralConvert(getInput(), getType(), getTreatIndexAsUnsigned());
  assert(result.isError() && "Should be folded if present");
  return result.takeError();
}

//===----------------------------------------------------------------------===//
// IntLiteralBinAttr
//===----------------------------------------------------------------------===//

TypedAttr IntLiteralBinAttr::get(MLIRContext *ctx, TypedAttr lhs, TypedAttr rhs,
                                 IntLiteralBinKindAttr oper) {
  // If this is a literal constant coming in, we can fold this.  If not, stage
  // it until elaboration or something else simplifies things.
  IntLiteralAttr lAttr = ::dyn_cast_or_null<IntLiteralAttr>(lhs);
  IntLiteralAttr rAttr = ::dyn_cast_or_null<IntLiteralAttr>(rhs);
  if (!lAttr || !rAttr)
    return Base::get(ctx, lhs, rhs, oper);

  IPInt l = lAttr.getValue();
  IPInt r = rAttr.getValue();

  IPInt result;
  switch (oper.getValue()) {
  case IntLiteralBinKind::Add:
    result = l + r;
    break;
  case IntLiteralBinKind::Sub:
    result = l - r;
    break;
  case IntLiteralBinKind::Mul:
    result = l * r;
    break;
  case IntLiteralBinKind::FloorDiv: {
    IPInt zero(0);
    if (r == zero) { // x // 0 = 0.
      result = zero;
      break;
    }

    if ((l >= zero) == (r >= zero) || l % r == zero)
      result = l / r;
    else
      result = (l / r) - IPInt(1);
    break;
  }
  case IntLiteralBinKind::Mod: {
    // Python's mod:
    // The result sign matches the RHS sign.
    // If the signs match, the value is the same as: sign(abs(l) % abs(r)),
    // where sign is determined by the RHS sign. If the signs don't match, the
    // value is the same as: sign((abs(r) - (abs(l) % abs(r))) % abs(r)).
    IPInt zero(0);
    if (r == zero) { // x % 0 = 0.
      result = zero;
      break;
    }
    bool signMatch = (l >= zero) == (r >= zero);
    IPInt lAbs = l.abs();
    IPInt rAbs = r.abs();
    result = (lAbs % rAbs).abs();
    if (!signMatch && result != zero)
      result = rAbs - result;
    if (r < zero)
      result = zero - result;
    break;
  }
  case IntLiteralBinKind::Lshift:
    if (r < IPInt(0))
      result = IPInt(0);
    else
      result = l << r;
    break;
  case IntLiteralBinKind::Rshift:
    if (r < IPInt(0))
      result = IPInt(0);
    else
      result = l >> r;
    break;
  case IntLiteralBinKind::And:
    result = l & r;
    break;
  case IntLiteralBinKind::Or:
    result = l | r;
    break;
  case IntLiteralBinKind::Xor:
    result = l ^ r;
    break;
  }

  return IntLiteralAttr::get(lAttr.getContext(), IPInt(result));
}

bool IntLiteralBinAttr::isConstant() const { return false; }

Type IntLiteralBinAttr::getType() const {
  return IntLiteralType::get(getContext());
}

//===----------------------------------------------------------------------===//
// IntLiteralCmpAttr
//===----------------------------------------------------------------------===//

TypedAttr IntLiteralCmpAttr::get(MLIRContext *ctx, IntLiteralCmpPredAttr pred,
                                 TypedAttr lhs, TypedAttr rhs) {
  // If this is a literal constant coming in, we can fold this.  If not, stage
  // it until elaboration or something else simplifies things.
  IntLiteralAttr lAttr = ::dyn_cast_or_null<IntLiteralAttr>(lhs);
  IntLiteralAttr rAttr = ::dyn_cast_or_null<IntLiteralAttr>(rhs);
  if (!lAttr || !rAttr)
    return Base::get(ctx, pred, lhs, rhs);

  IPInt l = lAttr.getValue();
  IPInt r = rAttr.getValue();
  switch (pred.getValue()) {
  case IntLiteralCmpPred::Eq:
    return BoolAttr::get(lAttr.getContext(), l == r);
  case IntLiteralCmpPred::Ne:
    return BoolAttr::get(lAttr.getContext(), l != r);
  case IntLiteralCmpPred::Lt:
    return BoolAttr::get(lAttr.getContext(), l < r);
  case IntLiteralCmpPred::Le:
    return BoolAttr::get(lAttr.getContext(), l <= r);
  case IntLiteralCmpPred::Gt:
    return BoolAttr::get(lAttr.getContext(), l > r);
  case IntLiteralCmpPred::Ge:
    return BoolAttr::get(lAttr.getContext(), l >= r);
  }
  llvm_unreachable("invalid cmp predicate");
}

bool IntLiteralCmpAttr::isConstant() const { return false; }

Type IntLiteralCmpAttr::getType() const {
  return IntegerType::get(getContext(), 1);
}

//===----------------------------------------------------------------------===//
// FloatLiteralConvertAttr
//===----------------------------------------------------------------------===//

/// Take an IPRational along with a specification for an output float type and
/// return the IEEE-style float bit string as an APInt.
static APInt
floatLiteralConvertGetBitstring(const IPRational &input,
                                const llvm::fltSemantics &fltSemantics) {
  unsigned totalLength = APFloat::getSizeInBits(fltSemantics);
  // For semantics without inf e.g. e4m3fn, the max exponent is bias + 1
  // since inf (all ones for exponent) is used for values.
  unsigned bias = APFloat::semanticsHasInf(fltSemantics)
                      ? APFloat::semanticsMaxExponent(fltSemantics)
                      : APFloat::semanticsMaxExponent(fltSemantics) - 1;
  unsigned exponentLength =
      llvm::Log2_64(APFloat::semanticsMaxExponent(fltSemantics) -
                    APFloat::semanticsMinExponent(fltSemantics) + 3);

  // Throughout this function I use “significand” to mean the float value
  // including the digit before the decimal, and “mantissa” to mean just the
  // part after the decimal, IE the bit pattern that is actually present in the
  // float value.  That's not technically correct, but it was helpful for me to
  // distinguish the two.
  unsigned mantissaLength = totalLength - exponentLength - 1;
  IPInt maxExponentZeroBias = (IPInt(1) << exponentLength) - 1;
  IPInt maxExponent = maxExponentZeroBias - bias;
  IPInt minExponent = IPInt(-1) * IPInt(bias - 1);

  // The maxSignificandIPIntLength is longer than the float mantissa bit width
  // to allow for:
  // * leading 0 in IPInt format
  // * most significant 1 bit that is removed in final encoding
  // * extra precision bits to ensure correct rounding
  unsigned maxSignificandIPIntRoundedLength = mantissaLength + 2;
  static const unsigned kSignificandRoundingLength = 3;
  unsigned maxSignificandIPIntLength =
      maxSignificandIPIntRoundedLength + kSignificandRoundingLength;

  // To support subnormal numbers (IE numbers with minimum exponent that have an
  // implicit leading 0 instead of implicit leading 1), we need to support lower
  // exponents during calculation.
  IPInt minCalculationExponent = minExponent - mantissaLength;

  if (input.getNumerator() == 0)
    return APInt(totalLength, 0);

  bool negativeSign = input.getNumerator() < 0;
  APInt signBits = APInt(totalLength, negativeSign ? 1 : 0);
  signBits = signBits << (totalLength - 1);

  IPInt initialNumerator = input.getNumerator().abs();
  const IPInt &denominator = input.getDenominator();
  IPInt significand = initialNumerator / denominator;
  IPInt remainder = initialNumerator % denominator;
  IPInt exponent = 0;
  bool exponentFinalized = false;
  if (significand > 0) {
    // The IPInt encoding of the number will have a leading 0 bit (because it is
    // positive), and the exponent when treating the most significant one bit is
    // one less than the number of bits representing the number with no leading
    // zeroes.
    exponent = significand.getAPInt().getBitWidth() - 2;
    exponentFinalized = true;
  }

  auto keepDoingLongDivision = [&]() -> bool {
    if (remainder == 0)
      return false;
    if (exponent < minCalculationExponent || exponent > maxExponent)
      return false;
    if (significand.getAPInt().getBitWidth() > maxSignificandIPIntLength)
      return false;
    return true;
  };

  // Do long division loop.
  while (keepDoingLongDivision()) {
    unsigned nBitsToShift = denominator.getAPInt().getBitWidth() -
                            remainder.getAPInt().getBitWidth();
    if (nBitsToShift == 0)
      nBitsToShift = 1;
    IPInt nCur = remainder << nBitsToShift;
    if (!exponentFinalized) {
      exponent = exponent - nBitsToShift;
    }
    IPInt quotient = nCur / denominator;
    remainder = nCur % denominator;
    if (quotient > 0)
      exponentFinalized = true;
    significand = (significand << nBitsToShift) + quotient;
  }

  // If we finished long division with “enough” rounding bits, but the remainder
  // is still not zero, it means that eventually there will be another 1 bit,
  // which would break a rounding tie.  Appending any further 1 bit will have
  // the same effect on rounding (no effect other than tie breaking), so we just
  // add the next one.
  if (remainder != 0)
    significand = (significand << 1) + 1;

  // Early return for obvious zero case because our later logic requires a
  // non-zero significand.
  if (significand == 0)
    return signBits;

  // Pad to mantissa length before performing rounding, etc.
  if (significand.getAPInt().getBitWidth() < maxSignificandIPIntLength) {
    significand = significand << (maxSignificandIPIntLength -
                                  significand.getAPInt().getBitWidth());
  }

  auto performRounding = [](IPInt &significand, IPInt &exponent,
                            unsigned maxSignificandIPIntRoundedLength) {
    APInt roundingBits = significand.getAPInt().extractBits(
        /*numBits=*/significand.getAPInt().getBitWidth() -
            maxSignificandIPIntRoundedLength,
        /*bitPosition=*/0);
    unsigned roundingBitsActualLength = roundingBits.getBitWidth();
    APInt roundingMidpoint = APInt(roundingBitsActualLength, 1)
                             << (roundingBitsActualLength - 1);
    // Truncate bits first.
    significand = significand >> roundingBitsActualLength;
    // Now that we've truncated, rounding either means doing nothing (for
    // round toward zero) or adding one to the significand representation
    // (for rounding away from zero). The default rounding mode for IEEE
    // floats is “round to nearest, ties to even”. It might be good to take
    // an option to do other rounding modes, but for now we just support the
    // default.
    if (roundingBits.ugt(roundingMidpoint))
      significand = significand + 1;
    else if (roundingBits == roundingMidpoint && significand % 2 == 1)
      significand = significand + 1;
    // If rounding up increased digit count, we need to convert that into a
    // larger exponent and re-truncate.
    if (significand.getAPInt().getBitWidth() >
        maxSignificandIPIntRoundedLength) {
      exponent = exponent + 1;
      significand = significand >> 1;
    }
  };

  // Do rounding now unless we are dealing with a subnormal number, which needs
  // some extra handling before rounding.
  if (exponent >= minExponent)
    performRounding(significand, exponent, maxSignificandIPIntRoundedLength);

  if (exponent > maxExponent) {
    // Return +/- infinity.
    APInt exponentOnes = APInt::getAllOnes(exponentLength);
    APInt exponentBits = APInt(totalLength, 0);
    exponentBits.insertBits(exponentOnes, mantissaLength);
    // Mantissa for infinity is zero.
    return signBits | exponentBits;
  }

  // Handle subnormal numbers, including zero values.  (I'm not sure whether
  // zero counts technically as a subnormal number, but it fits the subnormal
  // encoding.)
  if (exponent < minExponent) {
    // Below the minExponent we can still convert to subnormal numbers.
    // The subnormal range is tagged with minExponent - 1, but the exponent
    // value is effectively the same as minExponent. However, instead of an
    // implicit leading 1 before the decimal, there is a leading 0. So subnormal
    // numbers cover down to minExponent - mantissaWidth exponent, but
    // losing one bit of mantissa precision for each exponent lowering.
    if (exponent < minCalculationExponent) {
      // We could let this fall through and be handled by the shifting and bit
      // mangling, but at this point we know that every bit is zero except
      // (maybe) the sign.
      return signBits;
    }
    IPInt shiftBits = minExponent - exponent;
    IPInt shiftTag = IPInt(1) << (IPInt(significand.getAPInt().getBitWidth()) -
                                  IPInt(2) + shiftBits);
    // The significand is now
    // `01<correct-bit-pattern><at-least-one-extra-bit>`.
    significand = shiftTag + significand;
    exponent = minExponent - 1;
    // If rounding increases the exponent and carries to a new high bit, then we
    // end up at 1000... for the significand with minExponent, and thus the
    // right number.  Cool.
    performRounding(significand, exponent, maxSignificandIPIntRoundedLength);
  }

  // Whether or not the value was subnormal, the significand now has the bit
  // pattern `01<correct-bit-pattern><maybe-extra-bit-due-to-rounding>`.  So we
  // drop the leading 2 bits and the trailing extra bits to arrive at the final
  // bit pattern for the mantissa.

  unsigned extraSignificandBits =
      significand.getAPInt().getBitWidth() - (mantissaLength + 2);
  significand = significand >> extraSignificandBits;
  assert(significand.getAPInt().getBitWidth() == mantissaLength + 2 &&
         "proper mantissa bit length");
  APInt mantissaLowBits = significand.getAPInt().extractBits(
      /*numBits=*/mantissaLength,
      /*bitPosition=*/0);
  APInt mantissaBits = APInt(totalLength, 0);
  mantissaBits.insertBits(mantissaLowBits, /*bitPosition=*/0);

  // Floating point numbers encode the exponent as `bias + exponent`, so that
  // the result is always a natural number, where `bias + exponent = 0`
  // signifies subnormal (including zero) numbers, and all ones is the
  // exponent for infinity and the NAN values.
  exponent = exponent + bias;
  // Place the bits into an APInt at the appropriate place.
  APInt exponentBits = APInt(totalLength, 0);
  exponentBits.insertBits(exponent.getAPInt(), mantissaLength);

  // Combine pieces to get final bit string: <sign><exponent><mantissa>.
  return signBits | exponentBits | mantissaBits;
}

static ErrorOr<TypedAttr> foldFloatLiteralConvert(TypedAttr input,
                                                  Type outType) {
  auto inputLitAttr = ::dyn_cast_or_null<FloatLiteralAttr>(input);
  if (!inputLitAttr)
    return Error("input must be FloatLiteralAttr");

  const llvm::fltSemantics *fltSemantics = nullptr;

  // Handle !scalar<f32> aka !simd<f32, 1>
  auto simd = dyn_cast<SIMDType>(outType);
  if (auto dtype = simd.getResolvedDType())
    if (simd.getResolvedSize() && dtype->isFloat())
      fltSemantics = dtype->getFloatSemantics();

  if (!fltSemantics) {
    std::string str;
    llvm::raw_string_ostream os(str);
    os << outType;
    return Error("float literal conversion: unsupported output type: " +
                 os.str());
  }

  APFloat resultValue(*fltSemantics, APFloat::uninitialized);
  switch (inputLitAttr.getSpecial().getValue()) {
  case FloatLiteralSpecialValues::Nan:
    // Set the payload to uint64_t::max to make the NaN fill all the low bits
    // to 1. This makes the NaN value aligned with the NaN values generated by
    // CUDA libraries.
    resultValue = APFloat::getNaN(
        *fltSemantics,
        /*Negative=*/false, /*payload=*/std::numeric_limits<uint64_t>::max());
    break;
  case FloatLiteralSpecialValues::Inf:
    resultValue = APFloat::getInf(*fltSemantics, /*negative=*/false);
    break;
  case FloatLiteralSpecialValues::NegInf:
    resultValue = APFloat::getInf(*fltSemantics, /*negative=*/true);
    break;
  case FloatLiteralSpecialValues::NegZero:
    resultValue = APFloat::getZero(*fltSemantics, /*negative=*/true);
    break;
  case FloatLiteralSpecialValues::Normal: {
    std::optional<IPRational> inRat = inputLitAttr.getRational();
    assert(inRat.has_value() && "normal FloatLiteral values have a rational");
    APInt floatBits =
        floatLiteralConvertGetBitstring(inRat.value(), *fltSemantics);
    resultValue = APFloat(*fltSemantics, floatBits);
    break;
  }
  }

  // Form a SIMDAttr for values of !simd type, splating the value out as needed.
  DTypeValue value(resultValue, *simd.getResolvedDType());
  SmallVector<DTypeValue> values(*simd.getResolvedSize(), value);
  return SIMDAttr::get(values, simd);
}

TypedAttr FloatLiteralConvertAttr::get(MLIRContext *ctx, Type type,
                                       TypedAttr input) {
  assert(!::isa<FloatLiteralType>(type) && !type.isF64() &&
         "should convert to SIMD type");

  // If this is a literal constant coming in, we can fold this.  If not, stage
  // it until elaboration simplifies things.
  auto errOrAttr = foldFloatLiteralConvert(input, type);
  if (errOrAttr.isError())
    return Base::get(ctx, type, input);
  return errOrAttr.get();
}

bool FloatLiteralConvertAttr::isConstant() const { return false; }

ErrorOrSuccess FloatLiteralConvertAttr::validateForElaborator() const {
  auto result = foldFloatLiteralConvert(getInput(), getType());
  assert(result.isError() && "Should be folded if present");
  return result.takeError();
}

//===----------------------------------------------------------------------===//
// IntToFloatLiteralAttr
//===----------------------------------------------------------------------===//

TypedAttr IntToFloatLiteralAttr::get(MLIRContext *ctx, TypedAttr input) {
  // If this is a literal constant coming in, we can fold this.  If not, stage
  // it until elaboration or something else simplifies things.
  auto inputAttr = ::dyn_cast_or_null<IntLiteralAttr>(input);
  if (!inputAttr)
    return Base::get(ctx, input);

  return FloatLiteralAttr::get(inputAttr.getContext(),
                               IPRational(inputAttr.getValue(), IPInt(1)));
}

bool IntToFloatLiteralAttr::isConstant() const { return false; }

Type IntToFloatLiteralAttr::getType() const {
  return FloatLiteralType::get(getContext());
}

//===----------------------------------------------------------------------===//
// FloatToIntLiteralAttr
//===----------------------------------------------------------------------===//

TypedAttr FloatToIntLiteralAttr::get(MLIRContext *ctx, TypedAttr input) {
  // If this is a literal constant coming in, we can fold this.  If not, stage
  // it until elaboration or something else simplifies things.
  auto inputAttr = ::dyn_cast_or_null<FloatLiteralAttr>(input);
  if (!inputAttr)
    return Base::get(ctx, input);

  IPInt result;
  switch (inputAttr.getSpecial().getValue()) {
  case FloatLiteralSpecialValues::Nan:
  case FloatLiteralSpecialValues::Inf:
  case FloatLiteralSpecialValues::NegInf:
  case FloatLiteralSpecialValues::NegZero:
    result = 0;
    break;
  case FloatLiteralSpecialValues::Normal:
    assert(inputAttr.getRational().has_value() &&
           "normal FloatLiterals have rational");
    result = inputAttr.getRational()->getNumerator() /
             inputAttr.getRational()->getDenominator();
    break;
  }
  return IntLiteralAttr::get(inputAttr.getContext(), result);
}

bool FloatToIntLiteralAttr::isConstant() const { return false; }

Type FloatToIntLiteralAttr::getType() const {
  return IntLiteralType::get(getContext());
}

//===----------------------------------------------------------------------===//
// FloatLiteralBinAttr
//===----------------------------------------------------------------------===//

static bool isNan(FloatLiteralSpecialValues v) {
  return v == FloatLiteralSpecialValues::Nan;
}
static bool isNegZero(FloatLiteralSpecialValues v) {
  return v == FloatLiteralSpecialValues::NegZero;
}
static bool isInf(FloatLiteralSpecialValues v) {
  return v == FloatLiteralSpecialValues::Inf;
}
static bool isNegInf(FloatLiteralSpecialValues v) {
  return v == FloatLiteralSpecialValues::NegInf;
}
static bool isNormal(FloatLiteralSpecialValues v) {
  return v == FloatLiteralSpecialValues::Normal;
}

static std::pair<FloatLiteralSpecialValues, IPRational>
floatLiteralAdd(FloatLiteralSpecialValues lSpecial,
                FloatLiteralSpecialValues rSpecial, IPRational lhs,
                IPRational rhs) {
  switch (lSpecial) {
  case FloatLiteralSpecialValues::NegZero:
    if (isNegZero(rSpecial))
      return {FloatLiteralSpecialValues::Normal, 0};
    return {rSpecial, rhs};
  case FloatLiteralSpecialValues::Inf:
    if (isNegInf(rSpecial) || isNan(rSpecial))
      return {FloatLiteralSpecialValues::Nan, 0};
    return {FloatLiteralSpecialValues::Inf, 0};
  case FloatLiteralSpecialValues::NegInf:
    if (isInf(rSpecial) || isNan(rSpecial))
      return {FloatLiteralSpecialValues::Nan, 0};
    return {FloatLiteralSpecialValues::NegInf, 0};
  case FloatLiteralSpecialValues::Nan:
    return {FloatLiteralSpecialValues::Nan, 0};
  case FloatLiteralSpecialValues::Normal:
    if (isNormal(rSpecial))
      return {FloatLiteralSpecialValues::Normal, lhs + rhs};
    return floatLiteralAdd(rSpecial, lSpecial, rhs, lhs);
  }
  llvm_unreachable("unknown FloatLiteral special type");
}

static std::pair<FloatLiteralSpecialValues, IPRational>
floatLiteralSub(FloatLiteralSpecialValues lSpecial,
                FloatLiteralSpecialValues rSpecial, IPRational lhs,
                IPRational rhs) {
  switch (lSpecial) {
  case FloatLiteralSpecialValues::NegZero:
    // When adding zeroes, the signs are basically XORed, like with
    // multiplication.
    if (isNegZero(rSpecial))
      return {FloatLiteralSpecialValues::Normal, 0};
    if (isNormal(rSpecial) && rhs == 0)
      return {FloatLiteralSpecialValues::NegZero, 0};
    return floatLiteralSub(FloatLiteralSpecialValues::Normal, rSpecial, 0, rhs);
  case FloatLiteralSpecialValues::Inf:
    if (isInf(rSpecial) || isNan(rSpecial))
      return {FloatLiteralSpecialValues::Nan, 0};
    return {FloatLiteralSpecialValues::Inf, 0};
  case FloatLiteralSpecialValues::NegInf:
    if (isNegInf(rSpecial) || isNan(rSpecial))
      return {FloatLiteralSpecialValues::Nan, 0};
    return {FloatLiteralSpecialValues::NegInf, 0};
  case FloatLiteralSpecialValues::Nan:
    return {FloatLiteralSpecialValues::Nan, 0};
  case FloatLiteralSpecialValues::Normal:
    switch (rSpecial) {
    case FloatLiteralSpecialValues::NegZero:
      return {lSpecial, lhs};
    case FloatLiteralSpecialValues::Inf:
      return {FloatLiteralSpecialValues::NegInf, 0};
    case FloatLiteralSpecialValues::NegInf:
      return {FloatLiteralSpecialValues::Inf, 0};
    case FloatLiteralSpecialValues::Nan:
      return {FloatLiteralSpecialValues::Nan, 0};
    case FloatLiteralSpecialValues::Normal:
      return {FloatLiteralSpecialValues::Normal, lhs - rhs};
    }
  }
  llvm_unreachable("unknown FloatLiteral special type");
}

/// Helper for multiplication, to keep the special case matching table separate.
/// Assumes that at least one of lSpecial and rSpecial is non-normal.
static FloatLiteralSpecialValues
floatLiteralMulSpecialCases(const FloatLiteralSpecialValues &lSpecial,
                            const FloatLiteralSpecialValues &rSpecial,
                            const IPRational &lhs, const IPRational &rhs) {
  switch (lSpecial) {
  case FloatLiteralSpecialValues::NegZero:
    switch (rSpecial) {
    case FloatLiteralSpecialValues::Nan:
    case FloatLiteralSpecialValues::Inf:
    case FloatLiteralSpecialValues::NegInf:
      return FloatLiteralSpecialValues::Nan;
    case FloatLiteralSpecialValues::NegZero:
      return FloatLiteralSpecialValues::Normal;
    case FloatLiteralSpecialValues::Normal:
      if (rhs < 0)
        return FloatLiteralSpecialValues::Normal;
      return FloatLiteralSpecialValues::NegZero;
    }
    llvm_unreachable("all specials covered");
  case FloatLiteralSpecialValues::Inf:
    switch (rSpecial) {
    case FloatLiteralSpecialValues::Nan:
    case FloatLiteralSpecialValues::NegZero:
      return FloatLiteralSpecialValues::Nan;
    case FloatLiteralSpecialValues::NegInf:
      return FloatLiteralSpecialValues::NegInf;
    case FloatLiteralSpecialValues::Inf:
      return FloatLiteralSpecialValues::Inf;
    case FloatLiteralSpecialValues::Normal:
      if (rhs == 0)
        return FloatLiteralSpecialValues::Nan;
      if (rhs < 0)
        return FloatLiteralSpecialValues::NegInf;
      return FloatLiteralSpecialValues::Inf;
    }
    llvm_unreachable("all specials covered");
  case FloatLiteralSpecialValues::NegInf:
    switch (rSpecial) {
    case FloatLiteralSpecialValues::Nan:
    case FloatLiteralSpecialValues::NegZero:
      return FloatLiteralSpecialValues::Nan;
    case FloatLiteralSpecialValues::NegInf:
      return FloatLiteralSpecialValues::Inf;
    case FloatLiteralSpecialValues::Inf:
      return FloatLiteralSpecialValues::NegInf;
    case FloatLiteralSpecialValues::Normal:
      if (rhs == 0)
        return FloatLiteralSpecialValues::Nan;
      if (rhs < 0)
        return FloatLiteralSpecialValues::Inf;
      return FloatLiteralSpecialValues::NegInf;
    }
    llvm_unreachable("all specials covered");
  case FloatLiteralSpecialValues::Nan:
    return FloatLiteralSpecialValues::Nan;
  case FloatLiteralSpecialValues::Normal:
    // The case of both being normal is handled up front, so we don't worry
    // about it here.  Instead just recur with flipped operand order to handle
    // the case that LHS is normal.
    return floatLiteralMulSpecialCases(rSpecial, lSpecial, rhs, lhs);
  }
  llvm_unreachable("unknown FloatLiteral special type");
}

static std::pair<FloatLiteralSpecialValues, IPRational>
floatLiteralMul(FloatLiteralSpecialValues lSpecial,
                FloatLiteralSpecialValues rSpecial, IPRational lhs,
                IPRational rhs) {
  if (isNormal(lSpecial) && isNormal(rSpecial)) {
    IPRational ratResult = lhs * rhs;
    if (ratResult == 0 && ((lhs < 0) || (rhs < 0)))
      return {FloatLiteralSpecialValues::NegZero, {}};
    return {FloatLiteralSpecialValues::Normal, ratResult};
  }
  return {floatLiteralMulSpecialCases(lSpecial, rSpecial, lhs, rhs), 0};
}

/// Helper to separate the special case logic for division.  Assumes that at
/// least one of lSpecial and rSpecial is non-normal.
static FloatLiteralSpecialValues
floatLiteralDivSpecialCases(const FloatLiteralSpecialValues &lSpecial,
                            const FloatLiteralSpecialValues &rSpecial,
                            const IPRational &lhs, const IPRational &rhs) {
  switch (lSpecial) {
  case FloatLiteralSpecialValues::NegZero:
    switch (rSpecial) {
    case FloatLiteralSpecialValues::Nan:
      return FloatLiteralSpecialValues::Nan;
    case FloatLiteralSpecialValues::Inf:
      return FloatLiteralSpecialValues::NegZero;
    case FloatLiteralSpecialValues::NegInf:
      return FloatLiteralSpecialValues::Normal;
    case FloatLiteralSpecialValues::NegZero:
      return FloatLiteralSpecialValues::Nan;
    case FloatLiteralSpecialValues::Normal:
      if (rhs == 0)
        return FloatLiteralSpecialValues::Nan;
      if (rhs < 0)
        return FloatLiteralSpecialValues::Normal;
      return FloatLiteralSpecialValues::NegZero;
    }
    llvm_unreachable("all specials covered");
  case FloatLiteralSpecialValues::Inf:
    switch (rSpecial) {
    case FloatLiteralSpecialValues::Nan:
    case FloatLiteralSpecialValues::NegZero:
    case FloatLiteralSpecialValues::NegInf:
    case FloatLiteralSpecialValues::Inf:
      return FloatLiteralSpecialValues::Nan;
    case FloatLiteralSpecialValues::Normal:
      if (rhs == 0)
        return FloatLiteralSpecialValues::Nan;
      if (rhs < 0)
        return FloatLiteralSpecialValues::NegInf;
      return FloatLiteralSpecialValues::Inf;
    }
    llvm_unreachable("all specials covered");
  case FloatLiteralSpecialValues::NegInf:
    switch (rSpecial) {
    case FloatLiteralSpecialValues::Nan:
    case FloatLiteralSpecialValues::NegZero:
    case FloatLiteralSpecialValues::NegInf:
    case FloatLiteralSpecialValues::Inf:
      return FloatLiteralSpecialValues::Nan;
    case FloatLiteralSpecialValues::Normal:
      if (rhs == 0)
        return FloatLiteralSpecialValues::Nan;
      if (rhs < 0)
        return FloatLiteralSpecialValues::Inf;
      return FloatLiteralSpecialValues::NegInf;
    }
    llvm_unreachable("all specials covered");
  case FloatLiteralSpecialValues::Nan:
    return FloatLiteralSpecialValues::Nan;
  case FloatLiteralSpecialValues::Normal:
    switch (rSpecial) {
    case FloatLiteralSpecialValues::Nan:
      return FloatLiteralSpecialValues::Nan;
    case FloatLiteralSpecialValues::Inf:
      if (lhs < 0)
        return FloatLiteralSpecialValues::NegZero;
      return FloatLiteralSpecialValues::Normal;
    case FloatLiteralSpecialValues::NegInf:
      if (lhs < 0)
        return FloatLiteralSpecialValues::Normal;
      return FloatLiteralSpecialValues::NegZero;
    case FloatLiteralSpecialValues::NegZero:
      return FloatLiteralSpecialValues::Nan;
    case FloatLiteralSpecialValues::Normal:
      llvm_unreachable("double normal case handled above");
    }
  }
  llvm_unreachable("unknown FloatLiteral special type");
}

static std::pair<FloatLiteralSpecialValues, IPRational>
floatLiteralTrueDiv(FloatLiteralSpecialValues lSpecial,
                    FloatLiteralSpecialValues rSpecial, IPRational lhs,
                    IPRational rhs) {
  if (isNormal(lSpecial) && isNormal(rSpecial)) {
    if (rhs == 0)
      return {FloatLiteralSpecialValues::Nan, 0};
    IPRational ratResult = lhs / rhs;
    if (lhs == 0 && rhs < 0)
      return {FloatLiteralSpecialValues::NegZero, 0};
    return {FloatLiteralSpecialValues::Normal, ratResult};
  };
  return {floatLiteralDivSpecialCases(lSpecial, rSpecial, lhs, rhs), 0};
}

static std::pair<FloatLiteralSpecialValues, IPRational>
floatLiteralFloorDiv(FloatLiteralSpecialValues lSpecial,
                     FloatLiteralSpecialValues rSpecial, IPRational lhs,
                     IPRational rhs) {
  auto truediv = floatLiteralTrueDiv(lSpecial, rSpecial, lhs, rhs);

  // Special values are propagated.
  if (!isNormal(truediv.first))
    return truediv;

  // Get the result as an integer value rounded towards zero.
  auto intval = truediv.second.getNumerator() / truediv.second.getDenominator();

  // Ensure this equality doesn't hit any implicit conversions.
  if (truediv.second >= 0 || truediv.second == intval)
    return {truediv.first, intval};
  return {truediv.first, intval - 1};
}

TypedAttr FloatLiteralBinAttr::get(MLIRContext *ctx, TypedAttr lhsA,
                                   TypedAttr rhsA,
                                   FloatLiteralBinKindAttr oper) {
  // If this is a literal constant coming in, we can fold this.  If not, stage
  // it until elaboration or something else simplifies things.
  auto lAttr = ::dyn_cast_or_null<FloatLiteralAttr>(lhsA);
  auto rAttr = ::dyn_cast_or_null<FloatLiteralAttr>(rhsA);
  if (!lAttr || !rAttr)
    return Base::get(ctx, lhsA, rhsA, oper);

  std::pair<FloatLiteralSpecialValues, IPRational> (*implFunc)(
      FloatLiteralSpecialValues, FloatLiteralSpecialValues, IPRational,
      IPRational) = nullptr;
  switch (oper.getValue()) {
  case FloatLiteralBinKind::Add:
    implFunc = floatLiteralAdd;
    break;
  case FloatLiteralBinKind::Sub:
    implFunc = floatLiteralSub;
    break;
  case FloatLiteralBinKind::Mul:
    implFunc = floatLiteralMul;
    break;
  case FloatLiteralBinKind::TrueDiv:
    implFunc = floatLiteralTrueDiv;
    break;
  case FloatLiteralBinKind::FloorDiv:
    implFunc = floatLiteralFloorDiv;
    break;
  }
  assert(implFunc && "unknown FloatLiteralBinop type");

  FloatLiteralSpecialValues lSpecial = lAttr.getSpecial().getValue();
  IPRational lhs;
  if (isNormal(lSpecial)) {
    assert(lAttr.getRational().has_value() &&
           "rational has value when special value is normal");
    lhs = lAttr.getRational().value();
  }
  FloatLiteralSpecialValues rSpecial = rAttr.getSpecial().getValue();
  IPRational rhs;
  if (isNormal(rSpecial)) {
    assert(rAttr.getRational().has_value() &&
           "rational has value when special value is normal");
    rhs = rAttr.getRational().value();
  }

  auto [result, rational] = implFunc(lSpecial, rSpecial, lhs, rhs);
  return FloatLiteralAttr::get(
      lAttr.getContext(),
      FloatLiteralSpecialValuesAttr::get(lAttr.getContext(), result), rational);
}

bool FloatLiteralBinAttr::isConstant() const { return false; }

Type FloatLiteralBinAttr::getType() const {
  return FloatLiteralType::get(getContext());
}

//===----------------------------------------------------------------------===//
// FloatLiteralCmpAttr
//===----------------------------------------------------------------------===//

/// Helper for float literal comparison.  The lhs/rhs values are only meaningful
/// when lSpecial/rSpecial are normal.
static bool floatLiteralCmpHelper(const FloatLiteralCmpPred &pred,
                                  const FloatLiteralSpecialValues &lSpecial,
                                  const FloatLiteralSpecialValues &rSpecial,
                                  const IPRational &lhs,
                                  const IPRational &rhs) {
  switch (pred) {
  case FloatLiteralCmpPred::Eq:
    if (lSpecial == rSpecial) {
      if (isNormal(lSpecial))
        return lhs == rhs;
      return !isNan(lSpecial);
    }
    // Python treats -0 and 0 as equal.
    if (isNegZero(lSpecial) && isNormal(rSpecial) && rhs == 0)
      return true;
    if (isNegZero(rSpecial) && isNormal(lSpecial) && lhs == 0)
      return true;
    return false;
  case FloatLiteralCmpPred::Ne:
    return !floatLiteralCmpHelper(FloatLiteralCmpPred::Eq, lSpecial, rSpecial,
                                  lhs, rhs);
  case FloatLiteralCmpPred::Lt:
    switch (lSpecial) {
    case FloatLiteralSpecialValues::Normal:
      switch (rSpecial) {
      case FloatLiteralSpecialValues::Normal:
        return lhs < rhs;
      case FloatLiteralSpecialValues::Inf:
        return true;
      case FloatLiteralSpecialValues::NegZero:
        return lhs < 0;
      default:
        return false;
      }
    case FloatLiteralSpecialValues::NegZero:
      switch (rSpecial) {
      case FloatLiteralSpecialValues::Normal:
        // This would be <=, but Python treats -0 as equal to 0, so the RHS
        // needs to be strictly greater than positive zero.
        return IPRational(0) < rhs;
      case FloatLiteralSpecialValues::Inf:
        return true;
      default:
        return false;
      }
    case FloatLiteralSpecialValues::Inf:
    case FloatLiteralSpecialValues::Nan:
      return false;
    case FloatLiteralSpecialValues::NegInf:
      return !isNan(rSpecial) && !isNegInf(rSpecial);
    }
    llvm_unreachable("all specials covered");
  case FloatLiteralCmpPred::Le:
    return floatLiteralCmpHelper(FloatLiteralCmpPred::Lt, lSpecial, rSpecial,
                                 lhs, rhs) ||
           floatLiteralCmpHelper(FloatLiteralCmpPred::Eq, lSpecial, rSpecial,
                                 lhs, rhs);
  case FloatLiteralCmpPred::Gt:
    if (isNan(lSpecial) || isNan(rSpecial))
      return false;
    return !floatLiteralCmpHelper(FloatLiteralCmpPred::Le, lSpecial, rSpecial,
                                  lhs, rhs);
  case FloatLiteralCmpPred::Ge:
    return floatLiteralCmpHelper(FloatLiteralCmpPred::Gt, lSpecial, rSpecial,
                                 lhs, rhs) ||
           floatLiteralCmpHelper(FloatLiteralCmpPred::Eq, lSpecial, rSpecial,
                                 lhs, rhs);
  }
  llvm_unreachable("invalid cmp predicate");
}

TypedAttr FloatLiteralCmpAttr::get(MLIRContext *ctx,
                                   FloatLiteralCmpPredAttr pred, TypedAttr lhsA,
                                   TypedAttr rhsA) {
  // If this is a literal constant coming in, we can fold this.  If not, stage
  // it until elaboration or something else simplifies things.
  auto lAttr = ::dyn_cast_or_null<FloatLiteralAttr>(lhsA);
  auto rAttr = ::dyn_cast_or_null<FloatLiteralAttr>(rhsA);
  if (!lAttr || !rAttr)
    return Base::get(ctx, pred, lhsA, rhsA);

  FloatLiteralSpecialValues lSpecial = lAttr.getSpecial().getValue();
  FloatLiteralSpecialValues rSpecial = rAttr.getSpecial().getValue();
  IPRational lhs;
  IPRational rhs;
  if (isNormal(lSpecial)) {
    assert(lAttr.getRational().has_value() &&
           "rational does not have a value when special value is normal");
    lhs = lAttr.getRational().value();
  }
  if (isNormal(rSpecial)) {
    assert(rAttr.getRational().has_value() &&
           "rational does not have a value when special value is normal");
    rhs = rAttr.getRational().value();
  }
  return BoolAttr::get(
      lAttr.getContext(),
      floatLiteralCmpHelper(pred.getValue(), lSpecial, rSpecial, lhs, rhs));
}

bool FloatLiteralCmpAttr::isConstant() const { return false; }

Type FloatLiteralCmpAttr::getType() const {
  return IntegerType::get(getContext(), 1);
}

//===----------------------------------------------------------------------===//
// FloatLiteralIsaAttr
//===----------------------------------------------------------------------===//

TypedAttr FloatLiteralIsaAttr::get(MLIRContext *ctx,
                                   FloatLiteralSpecialValuesAttr kind,
                                   TypedAttr input) {
  // If this is a literal constant coming in, we can fold this.  If not, stage
  // it until elaboration simplifies things.
  if (auto inputAttr = ::dyn_cast_or_null<FloatLiteralAttr>(input))
    return BoolAttr::get(ctx, inputAttr.getSpecial() == kind);

  return Base::get(ctx, kind, input);
}

bool FloatLiteralIsaAttr::isConstant() const { return false; }

Type FloatLiteralIsaAttr::getType() const {
  return IntegerType::get(getContext(), 1);
}

//===----------------------------------------------------------------------===//
// StringSizeAttr
//===----------------------------------------------------------------------===//

TypedAttr StringSizeAttr::get(MLIRContext *ctx, TypedAttr str) {
  // If input is a string literal, we can fold this
  if (auto strAttr = ::dyn_cast_or_null<StringAttr>(str))
    return IntegerAttr::get(IndexType::get(ctx), strAttr.getValue().size());

  return Base::get(ctx, str);
}

bool StringSizeAttr::isConstant() const { return false; }

Type StringSizeAttr::getType() const { return IndexType::get(getContext()); }

//===----------------------------------------------------------------------===//
// StringConcatAttr
//===----------------------------------------------------------------------===//

TypedAttr StringConcatAttr::get(MLIRContext *ctx, TypedAttr lhs,
                                TypedAttr rhs) {
  // If both inputs are string literals, we can fold this
  if (auto lhsStr = ::dyn_cast_or_null<StringAttr>(lhs))
    if (auto rhsStr = ::dyn_cast_or_null<StringAttr>(rhs)) {
      return StringAttr::get(lhsStr.getValue() + rhsStr.getValue(),
                             lhsStr.getType());
    }

  return Base::get(ctx, lhs, rhs);
}

bool StringConcatAttr::isConstant() const { return false; }

Type StringConcatAttr::getType() const { return StringType::get(getContext()); }

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "KGEN/POPDialect/POPEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "KGEN/POPDialect/POPAttrs.cpp.inc"
