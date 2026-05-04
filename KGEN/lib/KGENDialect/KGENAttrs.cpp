//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/Interpreter/InterpreterAttrs.h"
#include "KGEN/KGENDialect/KGENDType.h"
#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/MDialect/MTypeInterfaces.h"
#include "Support/STLExtras.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include <numeric>

using namespace M;
using namespace KGEN;

// Provide implementations for the enums we use.
#include "KGEN/KGENDialect/KGENEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// ODS Boilerplate
//===----------------------------------------------------------------------===//

namespace mlir {
/// Parse a dtype.
template <>
struct FieldParser<KGENDType> {
  static FailureOr<KGENDType> parse(AsmParser &parser) {
    StringRef value;
    if (parser.parseKeyword(&value))
      return failure();
    return KGENDType::getFromString(value);
  }
};
} // namespace mlir

//===----------------------------------------------------------------------===//
// KGENDialect attribute support
//===----------------------------------------------------------------------===//

void KGENDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "KGEN/KGENDialect/KGENAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Type Reference Resolution
//===----------------------------------------------------------------------===//

namespace M::KGEN {
TypedAttr getTypeRefForTypeValueIfResolved(TypedAttr typeRef) {
  typeRef = SugarAttr::strip(typeRef);
  if (isa<TypeGeneratorRefAttr, TypeInstanceRefAttr>(typeRef))
    return typeRef;

  auto typeParam = dyn_cast<TypeParamAttr>(typeRef);
  if (!typeParam)
    return {};

  auto typeValueType = sugarDynCast<TypeValueType>(typeParam.getTypeValue());
  if (!typeValueType)
    return {};

  typeRef = SugarAttr::strip(typeValueType.getTypeValue());
  if (isa<TypeGeneratorRefAttr, TypeInstanceRefAttr>(typeRef))
    return typeRef;
  return {};
}
} // namespace M::KGEN

//===----------------------------------------------------------------------===//
// EmitAsAttr
//===----------------------------------------------------------------------===//

bool EmitAsAttr::classof(Attribute attr) {
  auto intAttr = ::dyn_cast<IntegerAttr>(attr);
  if (!intAttr)
    return false;

#ifndef MODULAR_PRODUCTION
  return ::isa<IndexType>(intAttr.getType()) &&
         contains_if(
             ArrayRef{
                 EmitAs::ASM,
                 EmitAs::LLVM,
                 EmitAs::LLVM_OPT,
                 EmitAs::OBJECT,
                 EmitAs::LLVM_BITCODE,
                 EmitAs::LLVM_OPT_BITCODE,
             },
             [&](EmitAs kind) { return (int)kind == intAttr.getInt(); });
#else
  return ::isa<IndexType>(intAttr.getType()) &&
         contains_if(
             ArrayRef{
                 EmitAs::ASM,
                 EmitAs::LLVM,
                 EmitAs::LLVM_OPT,
                 EmitAs::OBJECT,
                 EmitAs::LLVM_BITCODE,
                 EmitAs::LLVM_OPT_BITCODE,
             },
             [&](EmitAs kind) { return (int)kind == intAttr.getInt(); });
#endif
}

EmitAsAttr EmitAsAttr::get(MLIRContext *ctx, EmitAs val) {
  return ::cast<EmitAsAttr>(Builder(ctx).getIndexAttr((int)val));
}

EmitAs EmitAsAttr::getValue() const { return (EmitAs)getInt(); }

//===----------------------------------------------------------------------===//
// ParamListAttr
//===----------------------------------------------------------------------===//

/// The variadic attribute is a constant if all element values are constants.
bool ParamListAttr::isConstant() const {
  return llvm::all_of(getValues(), ParameterAttr::isSimpleConstant) &&
         !isParameterizedType(getType());
}

LogicalResult
ParamListAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                      ArrayRef<TypedAttr> values, ParamListType type) {
  Type elementType = type.getElementType();
  for (auto [idx, value] : llvm::enumerate(values))
    if (value.getType() != elementType)
      return emitError() << "variadic sequence element #" << idx << " has type "
                         << value.getType() << " but expected " << elementType;
  return success();
}

static ParseResult parseVariadicValue(AsmParser &p,
                                      SmallVector<TypedAttr> &values,
                                      ParamListType type) {
  return p.parseCommaSeparatedList([&] {
    return parseParamValue(p, values.emplace_back(), type.getElementType());
  });
}

OptionalParseResult ParamListType::parseValue(AsmParser &p,
                                              TypedAttr &value) const {
  if (failed(p.parseOptionalLSquare()))
    return std::nullopt;
  if (succeeded(p.parseOptionalRSquare())) {
    value = ParamListAttr::get({}, *this);
    return mlir::success();
  }
  SmallVector<TypedAttr> values;
  if (failed(parseVariadicValue(p, values, *this)))
    return failure();
  value = ParamListAttr::get(values, *this);
  return p.parseRSquare();
}

LogicalResult ParamListType::printValue(AsmPrinter &p, TypedAttr value) const {
  auto variadic = ::dyn_cast<ParamListAttr>(value);
  if (!variadic)
    return failure();
  p << '[';
  llvm::interleaveComma(variadic.getValues(), p,
                        [&](TypedAttr value) { printParamValue(p, value); });
  p << ']';
  return success();
}

//===----------------------------------------------------------------------===//
// Variadic-related pAttr
//===----------------------------------------------------------------------===//

// If not folded, they are not a constant.
bool ParamListReduceAttr::isConstant() const { return false; }
bool ParamListSizeAttr::isConstant() const { return false; }
bool ParamListGetAttr::isConstant() const { return false; }
bool ParamListTabulateAttr::isConstant() const { return false; }
bool ParamListConcatAttr::isConstant() const { return false; }

LogicalResult
ParamListReduceAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                            Type type, TypedAttr base, TypedAttr variadic,
                            TypedAttr mapper) {
  if (type != base.getType())
    return emitError() << "mismatch between reduce base value type and output "
                          "type, expected output type: "
                       << type << ", got: " << base.getType();

  auto toApply = dyn_cast<GeneratorType>(mapper.getType());
  auto srcTp = dyn_cast<ParamListType>(variadic.getType());

  if (!toApply || !srcTp)
    return emitError()
           << "expected an input of variadic type and a GeneratorAttr for "
              "the mapper";

  // Adjust the depth by -1 before type comparison, since the generator attr
  // increases the depth by one.
  IndexDepthAdjuster adjuster(-1);
  toApply = cast<GeneratorType>(adjuster.replace(toApply));

  // Verify that the mapper takes (base, *Ts, index) as input.
  ArrayRef<Type> mapperInputTps = toApply.getInputParamTypes();
  if (mapperInputTps.size() != 3)
    return emitError() << "expected a GeneratorAttr that takes 3 argument";

  if (base.getType() != mapperInputTps[0] || srcTp != mapperInputTps[1] ||
      mapperInputTps[2] != IndexType::get(type.getContext()))
    return emitError() << "expected a GeneratorAttr that takes ["
                       << base.getType() << " ," << srcTp << ", "
                       << "index] for the mapper, but got ["
                       << mapperInputTps[0] << ", " << mapperInputTps[1] << ", "
                       << mapperInputTps[2] << "]";

  if (toApply.getBody() != type)
    return emitError() << "expected a GeneratorAttr with an output type of"
                       << type << ", but got " << toApply.getBody();

  return success();
}

LogicalResult
ParamListTabulateAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                              ParamListType type, TypedAttr count,
                              TypedAttr generator) {
  if (!isa<IndexType>(count.getType()))
    return emitError() << "expected 'index' type for count, got: "
                       << count.getType();

  auto genType = dyn_cast<GeneratorType>(generator.getType());
  if (!genType)
    return emitError() << "expected generator to have GeneratorType, got: "
                       << generator.getType();

  IndexDepthAdjuster adjuster(-1);
  genType = cast<GeneratorType>(adjuster.replace(genType));
  ArrayRef<Type> inputTps = genType.getInputParamTypes();
  if (inputTps.size() != 1 || inputTps[0] != IndexType::get(type.getContext()))
    return emitError()
           << "expected generator to take single index parameter, got "
           << inputTps.size() << " parameter(s)";

  if (genType.getBody() != type.getElementType())
    return emitError()
           << "expected generator body type to match variadic element type: "
           << type.getElementType() << ", got: " << genType.getBody();

  return success();
}

TypedAttr ParamListTabulateAttr::get(ParamListType type, TypedAttr count,
                                     TypedAttr generator) {
  // We can always fold an empty tabulate.
  auto cntAttr = sugarDynCast<IntegerAttr>(count);
  if (cntAttr && cntAttr.getInt() == 0)
    return ParamListAttr::get({}, type);

  // Defer to evaluateWithContext when we don't have an evaluation context
  // (e.g. we can't specialize the generator here).
  return Base::get(type.getContext(), type, count, generator);
}

TypedAttr ParamListTabulateAttr::getChecked(
    function_ref<::mlir::InFlightDiagnostic()> emitError, ParamListType type,
    TypedAttr count, TypedAttr generator) {
  if (failed(verify(emitError, type, count, generator)))
    return {};
  return get(type, count, generator);
}

LogicalResult
ParamListSizeAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                          TypedAttr variadic) {
  if (!isa<ParamListType>(variadic.getType()))
    return emitError() << "expected a 'variadic' type for the count, got: "
                       << variadic.getType();
  return success();
}

TypedAttr ParamListSizeAttr::get(TypedAttr variadic) {
  auto vaAttr = sugarDynCast<ParamListAttr>(variadic);
  if (vaAttr)
    return IntegerAttr::get(IndexType::get(variadic.getContext()),
                            vaAttr.getValues().size());

  return Base::get(variadic.getContext(), variadic);
}

TypedAttr ParamListSizeAttr::getChecked(
    function_ref<::mlir::InFlightDiagnostic()> emitError, TypedAttr variadic) {
  if (failed(verify(emitError, variadic)))
    return {};
  return get(variadic);
}

LogicalResult
ParamListGetAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                         Type type, TypedAttr variadic, TypedAttr index) {
  auto variadicType = dyn_cast<ParamListType>(variadic.getType());
  if (!variadicType)
    return emitError()
           << "expected a 'variadic' type for the variadic operand, "
              "got: "
           << variadic.getType();
  if (type && type != variadicType.getElementType())
    return emitError() << "type must match variadic element type, expected: "
                       << variadicType.getElementType() << ", got: " << type;
  if (!isa<IndexType>(index.getType()))
    return emitError() << "expected 'index' type for the index, got: "
                       << index.getType();
  return success();
}

TypedAttr ParamListGetAttr::get(TypedAttr variadic, TypedAttr index) {
  if (auto vaAttr = sugarDynCast<ParamListAttr>(variadic)) {
    auto idxAttr = sugarDynCast<IntegerAttr>(index);
    // If the index is known-constant and in-range, we can simplify it.
    if (idxAttr && size_t(idxAttr.getInt()) < vaAttr.getValues().size())
      return vaAttr.getValues()[size_t(idxAttr.getInt())];

    // Fold if all elements are the same (e.g. if there is only one element!).
    if (!vaAttr.getValues().empty()) {
      auto first = vaAttr.getValues()[0];
      if (llvm::all_of(vaAttr.getValues().drop_front(),
                       [&](auto elt) { return elt == first; }))
        return first;
    }
  }

  auto resultType = cast<ParamListType>(variadic.getType()).getElementType();

  // Canonicalize upcast out of the variadic list:
  //   From: variadic_get<upcast<!Copyable> : !AnyType> : !AnyType
  // To: upcast<variadic_get<Copyable> : !Copyable> : !AnyType
  if (auto upcast = sugarDynCast<UpcastAttr>(variadic)) {
    TypedAttr originalVA = upcast.getInputTypeValue();
    if (!isa<ParamListType>(originalVA.getType()))
      return {};
    auto beforeCast = ParamListGetAttr::get(originalVA, index);
    return UpcastAttr::get(resultType, beforeCast);
  }

  return Base::get(variadic.getContext(), resultType, variadic, index);
}

TypedAttr ParamListGetAttr::getChecked(
    function_ref<::mlir::InFlightDiagnostic()> emitError, TypedAttr variadic,
    TypedAttr index) {
  if (failed(verify(emitError, /*type*/ {}, variadic, index)))
    return {};
  return get(variadic, index);
}

LogicalResult
ParamListConcatAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                            ParamListType type, TypedAttr variadics) {

  auto toConcatVA = dyn_cast<ParamListType>(variadics.getType());
  if (!toConcatVA)
    return emitError() << "expected to concat a variadic of variadic values";

  if (type != toConcatVA.getElementType())
    return emitError() << "mismatch between variadics to concatenate and "
                          "output type, expected output type: "
                       << type << ", got:" << toConcatVA.getElementType();

  return success();
}

TypedAttr ParamListConcatAttr::get(ParamListType type, TypedAttr variadics) {
  auto va = sugarDynCast<ParamListAttr>(variadics);
  if (!va)
    return Base::get(type.getContext(), type, variadics);

  size_t concatLen = 0;
  bool fullyResolved = true;
  SmallVector<ParamListAttr> elts =
      llvm::map_to_vector(va.getValues(), [&](TypedAttr elt) {
        auto vaElt = sugarDynCast<ParamListAttr>(elt);
        concatLen += (vaElt ? vaElt.getValues().size() : 0);
        fullyResolved = fullyResolved && vaElt;
        return vaElt;
      });

  if (!fullyResolved)
    return Base::get(type.getContext(), type, variadics);

  // Fold the attribute aggressively whenever possible upon creation.
  SmallVector<TypedAttr> concatElts;
  concatElts.reserve(concatLen);
  for (auto elt : elts)
    concatElts.append(elt.getValues().begin(), elt.getValues().end());

  return ParamListAttr::get(concatElts, type);
}

TypedAttr ParamListConcatAttr::getChecked(
    function_ref<::mlir::InFlightDiagnostic()> emitError, ParamListType type,
    TypedAttr variadics) {
  if (failed(verify(emitError, type, variadics)))
    return {};
  return get(type, variadics);
}

//===----------------------------------------------------------------------===//
// UnknownAttr
//===----------------------------------------------------------------------===//

bool UnknownAttr::isConstant() const { return !isParameterizedType(getType()); }

//===----------------------------------------------------------------------===//
// UnboundAttr
//===----------------------------------------------------------------------===//

bool UnboundAttr::isConstant() const { return !isParameterizedType(getType()); }

//===----------------------------------------------------------------------===//
// NoneAttr
//===----------------------------------------------------------------------===//

Type NoneAttr::getType() const { return KGEN::NoneType::get(getContext()); }

bool NoneAttr::isConstant() const { return true; }

//===----------------------------------------------------------------------===//
// ParamDeclRefAttr
//===----------------------------------------------------------------------===//

/// A parameter reference forms the basis of a non-constant parameter attribute.
bool ParamDeclRefAttr::isConstant() const { return false; }

/// Sort the parameter references by name.
bool ParamDeclRefAttr::isLessThan(Attribute rhs) const {
  auto ref = ::cast<ParamDeclRefAttr>(rhs);
  return getName().getValue() < ref.getName().getValue();
}

//===----------------------------------------------------------------------===//
// ParamIndexRefAttr
//===----------------------------------------------------------------------===//

/// A parameter reference is not a constant by definition.
bool ParamIndexRefAttr::isConstant() const { return false; }

/// Sort index references by index then kind.
bool ParamIndexRefAttr::isLessThan(Attribute rhs) const {
  auto ref = ::cast<ParamIndexRefAttr>(rhs);
  return std::make_tuple(getDepth(), getIndex()) <
         std::make_tuple(ref.getDepth(), ref.getIndex());
}

//===----------------------------------------------------------------------===//
// TypeParamAttr
//===----------------------------------------------------------------------===//

Attribute TypeParamAttr::parse(AsmParser &p, Type type) {
  if (p.parseLess())
    return {};

  TypedAttr value;
  OptionalParseResult result =
      parseTypeValueBody(p, value, type, parseOptionalKGENType);
  if (!result.has_value()) {
    p.emitError(p.getCurrentLocation(), "expected a type value");
    return {};
  }

  if (failed(*result) || p.parseGreater())
    return {};
  return value;
}

void TypeParamAttr::print(AsmPrinter &p) const {
  p << '<';
  void (*typePrinter)(AsmPrinter &, Type) = &printKGENType; // Select overload.
  printTypeValueBody(p, *this, typePrinter);
  p << '>';
}

TypedAttr TypeParamAttr::get(MLIRContext *ctx, Type typeValue, Type mlirType,
                             Type metaType) {
  // If this is a trivial mlir Type (i.e. has identical type & value
  // representation), and the trivial type is a ParamType, then we're
  // unwrapping a wrapper. Remove this to keep the types canonical.
  if (isEqualCanon(mlirType, typeValue)) {
    TypedAttr result;
    if (auto refType = dyn_cast<ParamType>(mlirType))
      result = refType.getParam();
    if (auto typeValueType = dyn_cast<TypeValueType>(mlirType))
      result = typeValueType.getTypeValue();
    if (result && isEqualCanon(result.getType(), metaType))
      return ParamOperatorAttr::getRebind(result, metaType);
  }

  if (auto typeValueType = dyn_cast<TypeValueType>(typeValue)) {
    // Unwrap immediately-nested TypeParamAttr as the typeValue. This is
    // casting the metatype of the inner type constant.
    if (auto innerTypeConstant =
            sugarDynCast<TypeParamAttr>(typeValueType.getTypeValue()))
      typeValue = innerTypeConstant.getTypeValue();

    // Unwrap identity type parameter wrapper.
    if (auto paramType = dyn_cast<ParamType>(mlirType)) {
      if (paramType.getParam() == typeValueType.getTypeValue() &&
          paramType.getParam().getType() == metaType)
        return paramType.getParam();
    }
  }

  return Base::get(ctx, typeValue, mlirType, metaType);
}

TypedAttr TypeParamAttr::get(Type typeValue, Type mlirType, Type type) {
  return get(typeValue.getContext(), typeValue, mlirType, type);
}

TypedAttr TypeParamAttr::get(Type mlirType, Type type) {
  return get(mlirType.getContext(), mlirType, mlirType, type);
}

TypeParamAttr TypeParamAttr::getFromBytecode(Type typeValue, Type mlirType,
                                             Type type) {
  return Base::get(mlirType.getContext(), typeValue, mlirType, type);
}

bool TypeParamAttr::isConstant() const {
  return !isParameterizedType(getMlirType());
}

bool TypeParamAttr::hasIdenticalRepresentation() {
  return getMlirType() == getTypeValue();
}

//===----------------------------------------------------------------------===//
// Upcast/DowncastAttr
//===----------------------------------------------------------------------===//

template <typename CastAttr>
static TypedAttr getCastAttr(Type type, TypedAttr inputTypeValue) {
  if (type == inputTypeValue.getType())
    return inputTypeValue;

  // If this is a constant type coming in, we can fold this.  If not, stage it
  // until elaboration.
  if constexpr (std::is_same_v<CastAttr, UpcastAttr>) {
    if (auto typeAttr = sugarDynCast<TypeParamAttr>(inputTypeValue)) {
      return TypeParamAttr::get(typeAttr.getTypeValue(), typeAttr.getMlirType(),
                                type);
    }
  }

  // This is a constant variadic of type value, fold each elements.
  if (auto variadicAttr = sugarDynCast<ParamListAttr>(inputTypeValue)) {
    auto dstVATp = cast<ParamListType>(type);
    Type elemTp = dstVATp.getElementType();

    SmallVector<TypedAttr> converted;
    for (auto typeValue : variadicAttr.getValues())
      converted.push_back(getCastAttr<CastAttr>(elemTp, typeValue));

    return ParamListAttr::get(converted, dstVATp);
  }

  // FIXME(MOCO-3601): unified upcast/downcast.
  // upcast(upcast(x)) = upcast(x)
  if constexpr (std::is_same_v<CastAttr, UpcastAttr>) {
    if (auto upcast = sugarDynCast<UpcastAttr>(inputTypeValue))
      return CastAttr::get(type, upcast.getInputTypeValue());

    // If we are upcasting an downcasted type value, we can not guarantee that
    // the outcome is an upcast. However, we can still fold it to a downcast,
    // and downcast knows how to fold it back to an upcast when it is provable.
    if (auto upcast = sugarDynCast<UpcastAttr>(inputTypeValue))
      return DowncastAttr::get(type, upcast.getInputTypeValue());
  }

  if constexpr (std::is_same_v<CastAttr, DowncastAttr>) {
    // downcast(downcast(x)) = downcast(x)
    if (auto downcast = sugarDynCast<DowncastAttr>(inputTypeValue))
      return DowncastAttr::get(type, downcast.getInputTypeValue());

    // If we are downcasting an upcasted type value, we can not guarantee that
    // the outcome is an upcast. However, we can still fold it to a downcast,
    // and downcast knows how to fold it back to an upcast when it is provable.
    if (auto upcast = sugarDynCast<UpcastAttr>(inputTypeValue))
      return DowncastAttr::get(type, upcast.getInputTypeValue());
  }

  return CastAttr::Base::get(type.getContext(), type, inputTypeValue);
}

TypedAttr UpcastAttr::get(Type type, TypedAttr inputTypeValue) {
  return getCastAttr<UpcastAttr>(type, inputTypeValue);
}
TypedAttr DowncastAttr::get(Type type, TypedAttr inputTypeValue) {
  return getCastAttr<DowncastAttr>(type, inputTypeValue);
}

template <typename CastAttr>
static LogicalResult
verifyCastAttr(function_ref<mlir::InFlightDiagnostic()> emitError, Type type,
               TypedAttr inputTypeValue) {
  bool isInputVA = isa<ParamListType>(inputTypeValue.getType());
  bool isResultVA = isa<ParamListType>(type);
  if (isInputVA != isResultVA)
    return emitError()
           << "must be casting from a variadic type to a variadic type";
  // NOTE: we should also verify that the output type must be a trait type, but
  // we don't have access to LIT::TraitType here due to build dependency.
  return success();
}

LogicalResult
UpcastAttr::verify(function_ref<mlir::InFlightDiagnostic()> emitError,
                   Type type, TypedAttr inputTypeValue) {
  return verifyCastAttr<UpcastAttr>(emitError, type, inputTypeValue);
}
LogicalResult
DowncastAttr::verify(function_ref<mlir::InFlightDiagnostic()> emitError,
                     Type type, TypedAttr inputTypeValue) {
  return verifyCastAttr<DowncastAttr>(emitError, type, inputTypeValue);
}

TypedAttr
UpcastAttr::getChecked(function_ref<::mlir::InFlightDiagnostic()> emitError,
                       Type type, TypedAttr inputTypeValue) {
  if (failed(verify(emitError, type, inputTypeValue)))
    return {};
  return get(type, inputTypeValue);
}
TypedAttr
DowncastAttr::getChecked(function_ref<::mlir::InFlightDiagnostic()> emitError,
                         Type type, TypedAttr inputTypeValue) {
  if (failed(verify(emitError, type, inputTypeValue)))
    return {};
  return get(type, inputTypeValue);
}

bool UpcastAttr::isConstant() const { return false; }
bool DowncastAttr::isConstant() const { return false; }

TypedAttr DowncastAttr::getTypeRefIfResolved() {
  return getTypeRefForTypeValueIfResolved(getInputTypeValue());
}

//===----------------------------------------------------------------------===//
// TypeConformsToAttr
//===----------------------------------------------------------------------===//

Type TypeConformsToTraitAttr::getType() const {
  return IntegerType::get(getContext(), 1);
}

TypedAttr TypeConformsToTraitAttr::getTypeRefIfResolved() {
  return getTypeRefForTypeValueIfResolved(getTypeValue());
}

FailureOr<TypedAttr>
TypeConformsToTraitAttr::simplify(const SymbolTable &traitTableOp,
                                  ParameterEvaluator &evaluator) const {
  SmallVector<TypedAttr> props;
  for (SymbolRefAttr traitSym : getTraitSymbols()) {
    auto conformOp = cast_or_null<ConformanceOp>(
        traitTableOp.lookup(getFlattenedSymbolName(traitSym)));

    if (!conformOp)
      return {BoolAttr::get(getContext(), false)};

    props.push_back(evaluator.replace(
        getCanonicalAttr(conformOp.getConstraint().getProposition())));
  }

  if (props.empty())
    return {BoolAttr::get(getContext(), true)};

  return {ParamOperatorAttr::get(POC::And, props)};
}

LogicalResult
TypeConformsToTraitAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                TypedAttr typeValue,
                                ArrayRef<SymbolRefAttr> traitSymbols) {
  return success();
}

//===----------------------------------------------------------------------===//
// FnTypeIsCABIAttr
//===----------------------------------------------------------------------===//

FnTypeIsCABIAttr FnTypeIsCABIAttr::get(MLIRContext *ctx, TypedAttr typeValue) {
  return Base::get(ctx, typeValue, getResultType(ctx));
}

//===----------------------------------------------------------------------===//
// GetWitnessAttr
//===----------------------------------------------------------------------===//

bool GetWitnessAttr::isConstant() const { return false; }

GetWitnessAttr GetWitnessAttr::get(MLIRContext *ctx, TypedAttr typeValue,
                                   StringAttr traitName, StringAttr witnessName,
                                   Type type) {
  return Base::get(ctx, UpcastAttr::strip(typeValue), traitName, witnessName,
                   type);
}

TypedAttr GetWitnessAttr::getTypeRefIfResolved() {
  return getTypeRefForTypeValueIfResolved(getTypeValue());
}

FailureOr<TypedAttr>
GetWitnessAttr::simplify(ConformanceOp witnessTable,
                         ParameterEvaluator *evaluator) const {
  for (WitnessOp entry : witnessTable.getOps<WitnessOp>()) {
    if (entry.getName() != getWitnessName().getValue())
      continue;

    Type entryType = entry.getValue().getType();
    if (evaluator) {
      entryType = evaluator->getReboundType(entryType);
      // If the type is not resolved, evaluation was skipped by the evaluator
      // due to a blocking dependency. Also return an empty attribute to skip
      // this evaluation too.
      if (!entryType)
        return TypedAttr();
    }

    if (!isEqualCanon(entryType, getType()))
      return failure();

    auto value = evaluator ? evaluator->getReboundAttribute(entry.getValue())
                           : entry.getValue();
    if (!value)
      return TypedAttr();
    // Realign sugar if needed.
    return ParamOperatorAttr::getRebind(value, getType());
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// CompileOffloadClosureAttr
//===----------------------------------------------------------------------===//

bool CompileOffloadClosureAttr::isConstant() const { return false; }

LogicalResult CompileOffloadClosureAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, TypedAttr target,
    TypedAttr func, Type type) {
  if (!::isa<TargetType>(target.getType()))
    return emitError() << "target operand must be of `!kgen.target` type";
  return success();
}

//===----------------------------------------------------------------------===//
// GetLinkageNameAttr
//===----------------------------------------------------------------------===//

bool GetLinkageNameAttr::isConstant() const { return false; }

//===----------------------------------------------------------------------===//
// LinkageNameAttr
//===----------------------------------------------------------------------===//

bool LinkageNameAttr::isConstant() const { return false; }

LinkageNameAttr LinkageNameAttr::get(MLIRContext *ctx, StringRef name,
                                     bool mangle) {
  return get(StringAttr::get(name, StringType::get(ctx)), mangle);
}

LogicalResult GetLinkageNameAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, TypedAttr target,
    TypedAttr func, Type type) {
  if (!::isa<TargetType>(target.getType()))
    return emitError() << "target operand must be of `!kgen.target` type";
  return success();
}

//===----------------------------------------------------------------------===//
// GetSourceNameAttr
//===----------------------------------------------------------------------===//

bool GetSourceNameAttr::isConstant() const { return false; }

//===----------------------------------------------------------------------===//
// CompileAssemblyAttr
//===----------------------------------------------------------------------===//

bool CompileAssemblyAttr::isConstant() const { return false; }

LogicalResult CompileAssemblyAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, TypedAttr target,
    TypedAttr emissionKind, TypedAttr emissionOptions, BoolAttr propagateError,
    TypedAttr func, Type type) {
  if (!::isa<TargetType>(target.getType()))
    return emitError() << "target operand must be of `!kgen.target` type";

  if (!::isa<IndexType>(emissionKind.getType()))
    return emitError() << "emissionKind operand should have index type";
  if (auto emissionIntAttr = ::dyn_cast<IntegerAttr>(emissionKind)) {
    if (!::isa<EmitAsAttr>(emissionIntAttr)) {
      return emitError() << "emissionKind operand should evaluate to either "
                            "'asm', 'llvm', 'llvm-opt', 'object', "
                            "'llvm-bitcode', or 'llvm-opt-bitcode'";
    }
  }

  if (!::isa<StringType>(emissionOptions.getType())) {
    return emitError()
           << "emissionOptions operand must be of `!kgen.string` type";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// GetTypeNameAttr
//===----------------------------------------------------------------------===//

bool GetTypeNameAttr::isConstant() const { return false; }

//===----------------------------------------------------------------------===//
// StructFieldTypesAttr
//===----------------------------------------------------------------------===//

bool StructFieldTypesAttr::isConstant() const { return false; }

//===----------------------------------------------------------------------===//
// StructFieldNamesAttr
//===----------------------------------------------------------------------===//

bool StructFieldNamesAttr::isConstant() const { return false; }

//===----------------------------------------------------------------------===//
// StructFieldIndexByNameAttr
//===----------------------------------------------------------------------===//

bool StructFieldIndexByNameAttr::isConstant() const { return false; }

//===----------------------------------------------------------------------===//
// StructFieldTypeByNameAttr
//===----------------------------------------------------------------------===//

bool StructFieldTypeByNameAttr::isConstant() const { return false; }

//===----------------------------------------------------------------------===//
// StructFieldOffsetByIndexAttr
//===----------------------------------------------------------------------===//

Type StructFieldOffsetByIndexAttr::getType() const {
  return IndexType::get(getContext());
}

bool StructFieldOffsetByIndexAttr::isConstant() const { return false; }

LogicalResult StructFieldOffsetByIndexAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    TypedAttr typeValue, TypedAttr fieldIndex, TypedAttr target) {
  // Only verify target type - fieldIndex type is checked during evaluation
  // since it may be a Mojo Int that gets converted to index.
  if (!::isa<TargetType>(target.getType()))
    return emitError() << "target operand must be of `!kgen.target` type";
  return success();
}

//===----------------------------------------------------------------------===//
// StructFieldOffsetByNameAttr
//===----------------------------------------------------------------------===//

Type StructFieldOffsetByNameAttr::getType() const {
  return IndexType::get(getContext());
}

bool StructFieldOffsetByNameAttr::isConstant() const { return false; }

LogicalResult StructFieldOffsetByNameAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    TypedAttr typeValue, TypedAttr fieldName, TypedAttr target) {
  // Only verify target type - fieldName type is checked during evaluation
  // since it may be a Mojo StringLiteral that gets converted to kgen.string.
  if (!::isa<TargetType>(target.getType()))
    return emitError() << "target operand must be of `!kgen.target` type";
  return success();
}

//===----------------------------------------------------------------------===//
// GetBaseTypeNameAttr
//===----------------------------------------------------------------------===//

Type GetBaseTypeNameAttr::getType() const {
  return StringType::get(getContext());
}

bool GetBaseTypeNameAttr::isConstant() const { return false; }

//===----------------------------------------------------------------------===//
// BindParamsAttr
//===----------------------------------------------------------------------===//

/// Merge two lists of parameter bindings into a single binding list.
/// Holes in a binding list are represented by UnboundAttrs.
static SmallVector<TypedAttr>
mergeParamBindings(ArrayRef<TypedAttr> prevBindings,
                   ArrayRef<TypedAttr> newBindings) {
  SmallVector<TypedAttr> mergedBindings;
  auto newIt = newBindings.begin();
  for (TypedAttr prevParam : prevBindings) {
    // If there was a hole here, and we still have new bindings left, fill it
    // with the next new binding. Otherwise, keep whatever was in the previous
    // binding.
    if (::isa<UnboundAttr>(prevParam) && newIt != newBindings.end())
      mergedBindings.push_back(*newIt++);
    else
      mergedBindings.push_back(prevParam);
  }
  // If we still have new bindings left, add them to the end.
  mergedBindings.append(newIt, newBindings.end());
  return mergedBindings;
}

static bool isEagerlyInstantiatable(GeneratorType generator) {
  // We can not eagerly instantiate a generator that contains a FuncType, since
  // it might contain implicit origins that must be used within a generator.
  return !sugarIsa<FuncType>(generator.getBody()) &&
         !sugarIsa<FuncLiteralType>(generator.getBody());
}

static Type inferBindParamsType(TypedAttr generator,
                                ArrayRef<TypedAttr> paramValues,
                                ParameterEvaluationContext *evaluationContext) {
  auto genType = sugarCast<GeneratorType>(generator.getType());
  GeneratorType specializedType =
      genType.getSpecializedGenerator(paramValues, evaluationContext,
                                      /*emitErrorFn=*/{});
  assert(specializedType && "Failed to specialize generator");
  // By back-compat, we never eliminate the empty generator type wrapper on
  // func types. This should eventually be made consistent with other types.
  bool canEagerInstantiate = specializedType.isFullyBound() &&
                             isEagerlyInstantiatable(specializedType);
  return canEagerInstantiate ? specializedType.getBody() : specializedType;
}

static TypedAttr simplifyBindParams(TypedAttr generator,
                                    ArrayRef<TypedAttr> paramValues,
                                    std::optional<Type> typeOpt,
                                    ParameterEvaluationContext *evalContext) {
  if (paramValues.empty())
    return generator;

  // If the actual generator is a BindParamsAttr, then we can flatten the new
  // bindings into the existing ones.
  if (auto bindParams = sugarDynCast<BindParamsAttr>(generator)) {
    SmallVector<TypedAttr> mergedParamValues =
        mergeParamBindings(bindParams.getParamValues(), paramValues);
    if (typeOpt)
      return BindParamsAttr::get(bindParams.getContext(),
                                 bindParams.getGenerator(), mergedParamValues,
                                 *typeOpt, evalContext);
    return BindParamsAttr::get(bindParams.getGenerator(), mergedParamValues,
                               evalContext);
  }

  // Can simplify if the generator is a GeneratorAttr.
  if (auto genAttr = sugarDynCast<GeneratorAttr>(generator)) {
    assert(
        evalContext &&
        "A foldable BindParamsAttr must be created with an evaluation context");
    // If the params are fully bound, just return the specialized generator.
    if (paramValues.size() == genAttr.getInputParamTypes().size()) {
      GeneratorAttr specializedGenerator =
          genAttr.getSpecializedGenerator(paramValues, evalContext);
      if (!specializedGenerator)
        return TypedAttr();

      return [&]() -> TypedAttr {
        if (isEagerlyInstantiatable(cast<GeneratorType>(genAttr.getType())) &&
            specializedGenerator.isFullyBound())
          return specializedGenerator.getInstantiatedValue();
        return specializedGenerator;
      }();
    }

    // Otherwise, fill in with unbound params to perform partial specialization.
    SmallVector<TypedAttr> partialParamValues;
    partialParamValues.reserve(genAttr.getInputParamTypes().size());
    for (auto [idx, type] : llvm::enumerate(genAttr.getInputParamTypes())) {
      if (idx < paramValues.size())
        partialParamValues.push_back(paramValues[idx]);
      else
        partialParamValues.push_back(UnboundAttr::get(type));
    }

    GeneratorAttr specializedGenerator =
        genAttr.getSpecializedGenerator(partialParamValues, evalContext);
    return cast_if_present<TypedAttr>(specializedGenerator);
  }

  // If the actual generator is a SymbolConstantAttr, then we can simplify by
  // folding the parameter values into it directly (this will be cleaned up once
  // we remove param bindings from SymbolConstantAttr).
  if (auto symbolConstant = sugarDynCast<SymbolConstantAttr>(generator)) {
    [[maybe_unused]] bool hasUnboundParameters =
        symbolConstant.getParamValues().empty();
    hasUnboundParameters |=
        llvm::any_of(symbolConstant.getParamValues(),
                     [](TypedAttr value) { return ::isa<UnboundAttr>(value); });
    assert(hasUnboundParameters &&
           "cannot have already bound all the input parameters, because we'd "
           "end up with a nongeneric signature that would fail verification");

    Type resultType = typeOpt.value_or(
        inferBindParamsType(generator, paramValues, evalContext));
    if (symbolConstant.getParamValues().empty())
      return SymbolConstantAttr::get(symbolConstant.getSymbol(),
                                     ::cast<FuncTypeGeneratorType>(resultType),
                                     paramValues);

    // We have to interleave the new values wherever there's an unbound thing
    // so we preserve the order.
    SmallVector<TypedAttr> mergedParamValues =
        mergeParamBindings(symbolConstant.getParamValues(), paramValues);

    return SymbolConstantAttr::get(symbolConstant.getSymbol(),
                                   ::cast<FuncTypeGeneratorType>(resultType),
                                   mergedParamValues);
  }

  return {};
}

TypedAttr BindParamsAttr::get(MLIRContext *context, TypedAttr generator,
                              ArrayRef<TypedAttr> paramValues, Type type,
                              ParameterEvaluationContext *evaluationContext) {
  if (auto simplified =
          simplifyBindParams(generator, paramValues, type, evaluationContext))
    return simplified;
  return Base::get(generator.getContext(), generator, paramValues, type);
}

TypedAttr
BindParamsAttr::getChecked(function_ref<InFlightDiagnostic()> emitError,
                           MLIRContext *context, TypedAttr generator,
                           ArrayRef<TypedAttr> paramValues, Type type,
                           ParameterEvaluationContext *evaluationContext) {
  if (failed(verify(emitError, generator, paramValues, type)))
    return {};
  return get(context, generator, paramValues, type, evaluationContext);
}

TypedAttr BindParamsAttr::get(TypedAttr generator,
                              ArrayRef<TypedAttr> paramValues,
                              ParameterEvaluationContext *evaluationContext) {
  if (auto simplified = simplifyBindParams(generator, paramValues, std::nullopt,
                                           evaluationContext))
    return simplified;
  Type resultType =
      inferBindParamsType(generator, paramValues, evaluationContext);
  return Base::get(generator.getContext(), generator, paramValues, resultType);
}

TypedAttr
BindParamsAttr::getChecked(function_ref<InFlightDiagnostic()> emitError,
                           TypedAttr generator, ArrayRef<TypedAttr> paramValues,
                           ParameterEvaluationContext *evaluationContext) {
  if (failed(verify(emitError, generator, paramValues, {})))
    return {};
  return get(generator, paramValues, evaluationContext);
}

bool BindParamsAttr::isConstant() const { return false; }

LogicalResult
BindParamsAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                       TypedAttr generator, ArrayRef<TypedAttr> paramValues,
                       Type type) {
  auto genType = sugarDynCast<GeneratorType>(generator.getType());
  if (!genType)
    return emitError()
           << "bind_params generator operand must have a GeneratorType, got "
           << generator.getType();

  if (paramValues.size() > genType.getInputParamTypes().size())
    return emitError()
           << "bind_params has more parameters than the generator expects";

  // It is possible that the parameter values do not have identical types as
  // what the generator type expects. This happens for example during
  // LiftAndFoldApply, where we may lift an apply from the operand, but not from
  // its type due to us not lifting apply operators out of generator types.
  // We will have to rely on post-elaboration verification to catch these.
  return success();
}

//===----------------------------------------------------------------------===//
// TypeGeneratorRefAttr
//===----------------------------------------------------------------------===//

/// Generator references are not constant. Must be evaluated into an instance
/// reference (during elaboration).
bool TypeGeneratorRefAttr::isConstant() const { return false; }

LogicalResult TypeGeneratorRefAttr::verifySymbolUses(
    SymTabEvaluationContext &evaluationContext, Location loc) const {
  VerboseCompilerTimeTraceScope traceScope(
      "TypeGeneratorRefAttr::verifySymbolUses");

  Operation *module = evaluationContext.module;
  mlir::LockedSymbolTableCollection &symtab = evaluationContext.symtab;

  // The leaf symbol is expected to only refer to a struct generator now.
  SymbolRefAttr symbol = getSymbol();
  StructGeneratorOp structGen;
  {
    VerboseCompilerTimeTraceScope traceScope("lookupSymbolIn");
    // TODO(MOCO-1360): Implement ExternStructGeneratorOp to support graph
    // compiler "forward decls".
    if (!(structGen = symtab.lookupSymbolIn<StructGeneratorOp>(module, symbol)))
      return success();
  }

  ParameterEvaluator evaluator;
  evaluator.setEvaluationContext(&evaluationContext);
  SmallVector<ParamDeclAttr> remappedParamDecls;
  for (auto [decl, value] :
       llvm::zip(structGen.getInputParams(), getParamValues())) {
    remappedParamDecls.push_back(ParamDeclAttr::get(
        decl.getName(), evaluator.getReboundType(decl.getType())));
    evaluator.setDeclBinding(decl.getName(), value);
    evaluator.appendIndexBinding(value);
  }

  // Check parameter types.
  if (failed(verifyParamDeclsMatch("input parameter",
                                   "!kgen.struct.generator symbol use",
                                   getParamValues(), loc, structGen.getName(),
                                   remappedParamDecls, structGen.getLoc())))
    return failure();

  // Check result type. Most likely it's not parameterized.
  Type specializedType = evaluator.getReboundType(structGen.getMetaType());
  if (!isEqualCanon(getType(), specializedType)) {
    return emitError(loc) << " result type mismatch. Reference has type "
                          << getType() << ", symbol has specialized type "
                          << specializedType;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TypeInstanceRefAttr
//===----------------------------------------------------------------------===//

/// This symbol is a constant its bindings are constants.
bool TypeInstanceRefAttr::isConstant() const { return true; }

//===----------------------------------------------------------------------===//
// DTypeConstantAttr
//===----------------------------------------------------------------------===//

Type DTypeConstantAttr::getType() const { return DTypeType::get(getContext()); }

bool DTypeConstantAttr::isConvertibleTo(Type type) {
  KGENDType dtype = getDType();

  // Bool can only be `i1`.
  if (dtype.isBool())
    return type.isSignlessInteger(1);

  // Index DType can only be the mlir `index` type.
  if (dtype.isIndex() || dtype.isUIndex())
    return type.isIndex();

  // Integer dtypes can be converted to MLIR integers of the same width and
  // un-opposing signedness; signed integer dtypes can be converted to signless
  // and signed MLIR integer types but not unsigned.
  if (dtype.isInt()) {
    auto intType = sugarDynCast<IntegerType>(type);
    if (!intType || intType.getWidth() != dtype.getWidthInBits())
      return false;
    return intType.isSignless() || intType.isSigned() == dtype.isSInt();
  }

  // Floating point dtypes can be converted to equivalent MLIR float types.
  if (dtype.isFloat()) {
    if (auto fpType = sugarDynCast<FloatType>(type))
      return areEquivalentFloatTypes(dtype, fpType);
    return false;
  }

  return false;
}

bool DTypeConstantAttr::isConvertibleFrom(Type type) {
  KGENDType dtype = getDType();

  if (dtype.isBool())
    return ::isa<IntegerType>(type);

  // Signless integers cannot be converted.
  if (type.isSignlessInteger() && !dtype.isIndex() && !dtype.isUIndex())
    return false;

  // Index dtypes can be converted if the type is an IndexType.
  if ((dtype.isIndex() || dtype.isUIndex()) && ::isa<IndexType>(type))
    return true;

  if (auto intType = sugarDynCast<IntegerType>(type)) {
    if (dtype.isIndex() || dtype.isUIndex())
      return true;
    // Integers can be converted to dtypes of the same width and signedness.
    if (dtype.isInt() && dtype.getWidthInBits() == intType.getWidth() &&
        dtype.isSInt() == intType.isSigned())
      return true;
    // Otherwise, we risk loosing bits, so we conservatively disallow.
    return false;
  }

  // Floating point types can be converted to equivalent dtypes.
  if (auto fpType = ::dyn_cast<FloatType>(type))
    return dtype.isFloat() && areEquivalentFloatTypes(dtype, fpType);

  return false;
}

/// Always a constant by definition.
bool DTypeConstantAttr::isConstant() const { return true; }

/// Sort by dtype value.
bool DTypeConstantAttr::isLessThan(Attribute rhs) const {
  auto dtype = ::cast<DTypeConstantAttr>(rhs);
  return getDType().getValue() < dtype.getDType().getValue();
}

//===----------------------------------------------------------------------===//
// SymbolConstantAttr
//===----------------------------------------------------------------------===//

static FuncTypeGeneratorType
getSymbolSignature(FuncInterface func, ArrayRef<Operation *> symbolOps) {
  if (symbolOps.size() == 1)
    return func.getFuncTypeGenerator();

  // Collect the contextual parameter values.
  SmallVector<ParamDeclAttr> paramDecls;
  for (Operation *op : llvm::drop_end(symbolOps))
    llvm::append_range(paramDecls, ::cast<DeclInterface>(op).getInputParams());

  IndexRefRemapper remapper(paramDecls, paramDecls.size());
  FuncTypeGeneratorType baseSigGen = func.getFuncTypeGenerator();
  FuncType baseSig = baseSigGen.getBody();
  SmallVector<Type> inputParamTypes;
  for (ParamDeclAttr param : paramDecls)
    inputParamTypes.push_back(remapper.replace(param.getType()));
  for (Type type : baseSigGen.getInputParamTypes())
    inputParamTypes.push_back(remapper.replace(type));

  GeneratorMetadataAttrInterface genMetadata = baseSigGen.getMetadata();
  if (genMetadata) {
    SmallVector<StringAttr> paramNames = llvm::map_to_vector(
        paramDecls, [](const ParamDeclAttr &param) { return param.getName(); });
    genMetadata = remapper.replace(genMetadata.prependContextualParamsFromOps(
        paramNames, symbolOps.drop_back()));
  }

  FnMetadataAttrInterface fnMetadata = baseSig.getMetadata();
  if (fnMetadata)
    fnMetadata = remapper.replace(fnMetadata);

  return FuncTypeGeneratorType::get(
      inputParamTypes, remapper.replace(baseSig.getValues()),
      baseSig.getArgConventions(), baseSig.getFnEffects(), fnMetadata,
      genMetadata);
}

/// This symbol is a constant its bindings are constants.
bool SymbolConstantAttr::isConstant() const {
  return llvm::all_of(getParamValues(), ParameterAttr::isSimpleConstant) &&
         !isParameterizedType(getType());
}

LogicalResult
SymbolConstantAttr::verifySymbolUses(SymTabEvaluationContext &evaluationContext,
                                     Location loc) const {
  VerboseCompilerTimeTraceScope traceScope(
      "SymbolConstantAttr::verifySymbolUses");

  Operation *module = evaluationContext.module;
  mlir::LockedSymbolTableCollection &symtab = evaluationContext.symtab;

  // Build the signature of the referenced symbol.
  SymbolRefAttr symbol = getSymbol();
  SmallVector<Operation *> symbolOps;
  {
    VerboseCompilerTimeTraceScope traceScope("lookupSymbolIn");
    if (failed(symtab.lookupSymbolIn(module, symbol, symbolOps))) {
      return emitError(loc)
             << symbol << " does not reference a KGEN declaration";
    }
  }

  // The leaf symbol must refer to a function.
  auto func = ::dyn_cast<FuncInterface>(symbolOps.back());
  if (!func)
    return emitError(loc) << symbol << " does not reference a KGEN function";

  // Everything else must be a declaration.
  for (Operation *op : llvm::drop_end(symbolOps)) {
    if (!::isa<DeclInterface>(op)) {
      return emitError(loc)
             << "symbol @" << ::cast<mlir::SymbolOpInterface>(op).getName()
             << " does not reference a KGEN declaration";
    }
  }

  FuncTypeGeneratorType declSignature = getSymbolSignature(func, symbolOps);
  declSignature = declSignature.getSpecializedGenerator(
      getParamValues(), &evaluationContext, [&] { return emitError(loc); });
  if (!declSignature)
    return failure();

  // Parameter types match exactly.  We could support higher order rebinding
  // if there is a need.
  return verifyDeclSignaturesMatch("symbol use", getType(), loc,
                                   symbol.getLeafReference(), declSignature,
                                   func->getLoc());
}

ParseResult parseColonTypeSymbolConstant(AsmParser &p,
                                         SymbolConstantAttr &value) {
  mlir::SMLoc loc = p.getCurrentLocation();

  TypedAttr typedAttr;
  if (parseColonTypeParamValue(p, typedAttr))
    return failure();

  if (auto symbol = mlir::dyn_cast<SymbolConstantAttr>(typedAttr)) {
    value = symbol;
    return success();
  }

  return p.emitError(loc) << "symbol constant expected, got" << typedAttr;
}

void printColonTypeSymbolConstant(AsmPrinter &p, SymbolConstantAttr value) {
  printColonTypeParamValue(p, value);
}

//===----------------------------------------------------------------------===//
// FuncSymbolAttr
//===----------------------------------------------------------------------===//

/// This symbol is a constant its bindings are constants.
bool FuncSymbolAttr::isConstant() const {
  return !isParameterizedType(getType());
}

LogicalResult
FuncSymbolAttr::verifySymbolUses(SymTabEvaluationContext &evaluationContext,
                                 Location loc) const {
  VerboseCompilerTimeTraceScope traceScope("FuncSymbolAttr::verifySymbolUses");

  Operation *module = evaluationContext.module;
  mlir::LockedSymbolTableCollection &symtab = evaluationContext.symtab;

  // Build the signature of the referenced symbol.
  SymbolRefAttr symbol = getSymbol();
  SmallVector<Operation *> symbolOps;
  {
    VerboseCompilerTimeTraceScope traceScope("lookupSymbolIn");
    if (failed(symtab.lookupSymbolIn(module, symbol, symbolOps)))
      return emitError(loc)
             << symbol << " does not reference a KGEN declaration";
  }

  // The leaf symbol must refer to a function.
  auto func = ::dyn_cast<FuncInterface>(symbolOps.back());
  if (!func)
    return emitError(loc) << symbol << " does not reference a KGEN function";

  // Everything else must be a declaration.
  for (Operation *op : llvm::drop_end(symbolOps)) {
    if (!::isa<DeclInterface>(op)) {
      return emitError(loc)
             << "symbol @" << ::cast<mlir::SymbolOpInterface>(op).getName()
             << " does not reference a KGEN declaration";
    }
  }

  // We are pulling out the index ref and evaluated it under a different scope,
  // -1 depth to compensate the extra depth pushed by getSpecializedGenerator.
  IndexDepthAdjuster adjuster(-1);
  SmallVector<TypedAttr> adjustedParam = adjuster.replace(getParamValues());
  FuncTypeGeneratorType declSignature = getSymbolSignature(func, symbolOps);
  declSignature = declSignature.getSpecializedGenerator(
      adjustedParam, &evaluationContext, [&] { return emitError(loc); });

  if (!declSignature)
    return failure();

  // Parameter types match exactly.  We could support higher order rebinding
  // if there is a need.
  return verifyFuncTypesMatch("symbol use", getType(), loc,
                              symbol.getLeafReference(),
                              declSignature.getBody(), func->getLoc());
}

//===----------------------------------------------------------------------===//
// GeneratorAttr
//===----------------------------------------------------------------------===//

// Custom assembly format implementation for GeneratorAttr
::mlir::Attribute GeneratorAttr::parse(::mlir::AsmParser &odsParser,
                                       ::mlir::Type odsType) {
  TypedAttr body;
  GeneratorType genType = ::cast<GeneratorType>(odsType);
  if (odsParser.parseLess() ||
      parseParamValue(odsParser, body, genType.getBody()) ||
      odsParser.parseGreater())
    return {};
  return GeneratorAttr::get(genType.getInputParamTypes(), body,
                            genType.getMetadata());
}

void GeneratorAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  odsPrinter << '<';
  printParamValue(odsPrinter, getBody(), getBody().getType());
  odsPrinter << '>';
}

/// A generator value needs to be instantiated.
bool GeneratorAttr::isConstant() const { return false; }

bool GeneratorAttr::isLessThan(Attribute rhs) const {
  return ParameterAttr::compare(getBody(),
                                ::cast<GeneratorAttr>(rhs).getBody());
}

GeneratorAttr GeneratorAttr::getSpecializedGenerator(
    ArrayRef<TypedAttr> paramBindings,
    ParameterEvaluationContext *evaluationContext,
    function_ref<InFlightDiagnostic()> emitErrorFn) {
  VerboseCompilerTimeTraceScope traceScope(
      "GeneratorAttr::getSpecializedGenerator");

  if (paramBindings.empty())
    return *this;

  std::optional<PartiallySpecializedInputParams> specializationOpt =
      PartiallySpecializedInputParams::from(getInputParamTypes(), paramBindings,
                                            evaluationContext, emitErrorFn);
  if (!specializationOpt)
    return {}; // Error already emitted to emitErrorFn.
  PartiallySpecializedInputParams &specialization = *specializationOpt;

  TypedAttr newBody =
      cast<TypedAttr>(specialization.evaluator.getReboundAttribute(getBody()));
  // Propagate null back to the caller. This only happens in materialization
  // contexts. It either indicates a failure (in which case errors must have
  // already been emitted to the evaluation context) or an async materialization
  // currently in-progress.
  if (!newBody)
    return {};

  // Create specialized metadata if needed
  GeneratorMetadataAttrInterface genMetadata = getMetadata();
  if (genMetadata) {
    genMetadata = genMetadata.getSpecializedMetadata(
        specialization.evaluator, specialization.boundParams, emitErrorFn);
    if (!genMetadata)
      return {}; // Error already emitted to emitErrorFn.
  }

  return GeneratorAttr::get(newBody.getContext(), newBody,
                            specialization.unboundParamTypes, genMetadata);
}

GeneratorAttr GeneratorAttr::getSpecializedGenerator(
    ArrayRef<TypedAttr> paramBindings,
    ParameterEvaluationContext *evaluationContext, Location location) {
  return getSpecializedGenerator(
      paramBindings, evaluationContext,
      [&]() -> InFlightDiagnostic { return emitError(location); });
}

TypedAttr GeneratorAttr::getInstantiatedValue() {
  assert(isFullyBound() && "cannot instantiate parameterized body");
  IndexDepthAdjuster adjuster(-1);
  TypedAttr body = adjuster.replace(getBody());
  return body;
}

//===----------------------------------------------------------------------===//
// TargetParamAttr
//===----------------------------------------------------------------------===//

Attribute TargetParamAttr::parse(AsmParser &p, Type type) {
  auto targetType = ::dyn_cast_or_null<TargetType>(type);
  if (!targetType) {
    p.emitError(p.getCurrentLocation(),
                "target parameter expected a target type");
    return {};
  }

  // Otherwise, parse the whole target info attribute.
  TargetInfoAttr target;
  if (p.parseCustomAttributeWithFallback(target))
    return {};
  return TargetParamAttr::get(target);
}

void TargetParamAttr::print(AsmPrinter &p) const { getTarget().print(p); }

Type TargetParamAttr::getType() const { return TargetType::get(getContext()); }

/// Always a constant.
bool TargetParamAttr::isConstant() const { return true; }

//===----------------------------------------------------------------------===//
// StructAttr
//===----------------------------------------------------------------------===//

static ParseResult parseStructElements(AsmParser &p,
                                       SmallVector<TypedAttr> &values,
                                       StructType type) {
  std::optional<SmallVector<Type>> elementTypes = type.getElementTypes();
  if (!elementTypes)
    return p.emitError(p.getCurrentLocation(),
                       "cannot parse elements of parametric struct");
  return failableInterleave(
      *elementTypes,
      [&](Type type) {
        return parseParamValue(p, values.emplace_back(), type);
      },
      [&] { return p.parseComma(); });
}

static void printStructElements(AsmPrinter &p, ArrayRef<TypedAttr> values,
                                StructType type) {
  llvm::interleaveComma(values, p,
                        [&](TypedAttr value) { printParamValue(p, value); });
}

OptionalParseResult StructType::parseValue(AsmParser &p,
                                           TypedAttr &value) const {
  if (failed(p.parseOptionalLBrace()))
    return std::nullopt;
  SmallVector<TypedAttr> values;
  if (failed(parseStructElements(p, values, *this)))
    return failure();
  value = StructAttr::get(values, *this);
  return p.parseRBrace();
}

LogicalResult StructType::printValue(AsmPrinter &p, TypedAttr value) const {
  auto structAttr = ::dyn_cast<StructAttr>(value);
  if (!structAttr)
    return failure();
  p << "{ ";
  llvm::interleaveComma(structAttr.getValues(), p,
                        [&](TypedAttr value) { printParamValue(p, value); });
  p << " }";
  return mlir::success();
}

/// The struct attribute is a constant if all element values are constants.
bool StructAttr::isConstant() const {
  return llvm::all_of(getValues(), ParameterAttr::isSimpleConstant);
}

LogicalResult StructAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<TypedAttr> values, StructType type) {
  std::optional<SmallVector<Type>> types = type.getElementTypes();
  if (!types)
    return emitError() << "cannot verify struct attribute with parametric type";
  if (types->size() != values.size())
    return emitError() << "struct attribute type requires " << types->size()
                       << " elements but value has " << values.size();
  for (auto [idx, value, type] :
       llvm::zip(llvm::seq<unsigned>(0, types->size()), values, *types)) {
    if (value.getType() != type) {
      return emitError() << "struct element #" << idx << " has type "
                         << value.getType() << " but expected " << type;
    }
  }
  return success();
}

StructType structTypeFromValues(ArrayRef<TypedAttr> values) {
  SmallVector<Type> types;
  types.reserve(values.size());
  for (TypedAttr value : values)
    types.push_back(value.getType());
  return StructType::get(types);
}

StructAttr StructAttr::get(ArrayRef<TypedAttr> values) {
  assert(!values.empty() && "expected at least one value");
  return StructAttr::get(values, structTypeFromValues(values));
}

StructAttr StructAttr::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                  ArrayRef<TypedAttr> values) {
  StructType type = structTypeFromValues(values);
  if (failed(verify(emitError, values, type)))
    return {};
  return StructAttr::get(values, type);
}

//===----------------------------------------------------------------------===//
// StructExtractAttr
//===----------------------------------------------------------------------===//

TypedAttr StructExtractAttr::get(TypedAttr structValue, unsigned fieldNo) {
  auto structType = ::cast<StructType>(structValue.getType());
  std::optional<SmallVector<Type>> elementTypes = structType.getElementTypes();
  assert(elementTypes && "cannot extract from parametric struct");
  assert(fieldNo < elementTypes->size() && "struct extract index out of range");
  auto fieldNoAttr =
      IntegerAttr::get(IndexType::get(structType.getContext()), fieldNo);
  return get(structValue, fieldNoAttr, (*elementTypes)[fieldNo]);
}

TypedAttr StructExtractAttr::get(TypedAttr structValue, TypedAttr fieldIdx,
                                 Type resultType) {
  if (auto value = dyn_cast<StructAttr>(structValue))
    if (auto fieldIdxAttr = dyn_cast<IntegerAttr>(fieldIdx))
      return value.getValues()[fieldIdxAttr.getInt()];
  if (::isa<UninitMemAttr>(structValue))
    return UninitMemAttr::get(resultType);

  return Base::get(structValue.getContext(), structValue, fieldIdx, resultType);
}

StructExtractAttr StructExtractAttr::getFromBytecode(TypedAttr structValue,
                                                     TypedAttr fieldIdx,
                                                     Type resultType) {
  return Base::get(resultType.getContext(), structValue, fieldIdx, resultType);
}

//===----------------------------------------------------------------------===//
// VariantAttr
//===----------------------------------------------------------------------===//

LogicalResult VariantAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                  TypedAttr value, unsigned index,
                                  VariantType type) {
  if (index >= type.getNumTypes())
    return emitError() << "variant index " << index << " is out of bounds";
  if (type.getType(index) == value.getType())
    return success();
  return emitError() << "variant attribute value type " << value.getType()
                     << " does not match type at index " << index
                     << " which is " << type.getType(index);
}

/// The variant attribute is a constant if the value type is a constant and its
/// type is not parameterized. It is possible to materialize a constant value
/// for a parametric variant type.
bool VariantAttr::isConstant() const {
  return ParameterAttr::isSimpleConstant(getValue()) &&
         !isParameterizedType(getType());
}

//===----------------------------------------------------------------------===//
// EnvAttr
//===----------------------------------------------------------------------===//

LogicalResult EnvAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                              DictionaryAttr values) {
  // Only index, bool, string, and unit attributes are allowed.
  for (const NamedAttribute &attr : values) {
    Attribute value = attr.getValue();
    if (auto intVal = ::dyn_cast<IntegerAttr>(value)) {
      if (!::isa<IndexType>(intVal.getType()))
        return emitError() << "environment value " << attr.getName()
                           << " is an integer not of `index` type";
    } else if (auto strVal = ::dyn_cast<StringAttr>(value)) {
      if (!::isa<StringType>(strVal.getType()))
        return emitError() << "environment value " << attr.getName()
                           << " is a string not of `!kgen.string` type";
    } else if (!::isa<UnitAttr>(value)) {
      return emitError() << "environment value " << attr.getName()
                         << " is neither an index, string, or unit attribute";
    }
  }
  return success();
}

ErrorOr<EnvAttr> EnvAttr::parseDefines(MLIRContext *ctx,
                                       ArrayRef<std::string> defines) {
  NamedAttrList attrs;
  Builder b(ctx);
  for (StringRef define : defines) {
    size_t idx = define.find("=");
    // If '=' is not present in the string, then this is a unit define.
    if (idx == StringRef::npos) {
      if (attrs.set(define, b.getUnitAttr()))
        return Error("'" + define + "' was defined more than once");
      continue;
    }
    StringRef name = define.slice(0, idx);
    StringRef value = define.slice(idx + 1, StringRef::npos);

    // Try to convert the value to an integer.
    APInt intVal(IndexType::kInternalStorageBitWidth, 0);
    if (!value.getAsInteger(/*Radix=*/10, intVal)) {
      if (attrs.set(name, b.getIndexAttr(intVal.getSExtValue())))
        return Error("'" + define + "' was defined more than once");
      continue;
    }

    // Otherwise, use it as a string value.
    if (attrs.set(name, StringAttr::get(value, StringType::get(ctx))))
      return Error("'" + define + "' was defined more than once");
  }

  return EnvAttr::get(attrs.getDictionary(ctx));
}

EnvAttr EnvAttr::extend(EnvAttr attr) {
  NamedAttrList values = getValues();
  for (NamedAttribute val : attr.getValues()) {
    // If the key exists, then we remove it so that we overwrite.
    if (values.getNamed(val.getName()))
      values.erase(val.getName());

    values.append(val);
  }

  return EnvAttr::get(values.getDictionary(attr.getContext()));
}

static TypedAttr implicitConversionToString(Attribute stored, Type desired) {
  if (!stored)
    return {};
  if (auto integerValue = dyn_cast<mlir::IntegerAttr>(stored)) {
    if (auto desiredStringType = dyn_cast<StringType>(desired)) {
      SmallString<32> strValue;
      integerValue.getValue().toString(strValue, /*Radix=*/10, /*Signed=*/true);
      return StringAttr::get(strValue, desiredStringType);
    }
  }
  return {};
}

ErrorOr<TypedAttr> EnvAttr::queryValue(StringRef name, Type outputType) const {
  MLIRContext *ctx = getContext();
  Attribute value = getValues().get(name);

  // Mental model of EnvAttr is a bit complicated.
  //
  // There are three possible types of stored values:
  // int (index)
  // string
  // unit

  // There are two types involved: (1) the type of the stored value,
  // and (2) the type being requested by the ParamOperatorAttr.
  // They do not necessarily match! It is permitted to query an integer
  // attribute and request a string output, the code below handles that.
  //
  // If the requested type is string or index, the attribute MUST
  // exist in the dictionary. If the requested type is bool and the attribute
  // does not exist, then the code returns false.

  if (isa<IndexType, StringType>(outputType) && !value) {
    return Error("define '" + name +
                 "' does not exist, please provide it via -D");
  }

  if (isa<IndexType>(outputType)) {
    if (auto intVal = dyn_cast<IntegerAttr>(value))
      return intVal;
    return Error("define '" + name + "' is not an integer, got " +
                 mlir::debugString(value));
  }

  if (isa<StringType>(outputType)) {
    if (auto strVal = dyn_cast<StringAttr>(value))
      return strVal;
    // This supports converting from IntegerAttr to StringAttr.
    if (TypedAttr stringAttr = implicitConversionToString(value, outputType))
      return stringAttr;
    return Error("define '" + name + "' is not a string, got " +
                 mlir::debugString(value));
  }

  // Now, the only remaining legal type on the ParamOperatorAttr is i1.
  assert(cast<IntegerType>(outputType).isSignlessInteger(1));
  // The only legal attr is a UnitAttr, its absence/presence means True/False.
  // Here we do not even cast the value to UnitAttr, just check its presence.
  return BoolAttr::get(ctx, static_cast<bool>(value));
}

//===----------------------------------------------------------------------===//
// ParamOperatorAttr
//===----------------------------------------------------------------------===//

ParamOperatorAttr
ParamOperatorAttr::getFromBytecode(POC opcode, ArrayRef<TypedAttr> operands,
                                   Type type) {
  return Base::get(type.getContext(), opcode, operands, type);
}

/// The 'apply' operator is able to call a generator value inside a parameter
/// expression. Therefore, it is one place where an index parameter reference
/// can cross upwards across a signature. We need to decrement any index
/// references in the result type of the signature because we are pulling it out
/// of the signature. See STCHDDDOS for more. This is STCHDDDOS-B.
static Type upbindApplyResult(Type resultType) {
  IndexDepthAdjuster adjuster(/*adjustDepth=*/-1);
  return adjuster.replace(resultType);
}

/// Verify the apply and apply_result ParameterOperatorExprs.
static LogicalResult
verifyApplyLike(ArrayRef<TypedAttr> operands, bool isApplyResult,
                function_ref<InFlightDiagnostic()> emitError) {
  StringRef prefix = isApplyResult ? "'apply_result_slot' " : "'apply' ";
  if (operands.empty())
    return emitError() << prefix << "expected a callee operand";

  auto callee = operands.front();

  auto sigGen = cast<FuncTypeGeneratorType>(operands.front().getType());
  if (!sigGen.getInputParamTypes().empty())
    return emitError() << prefix << "function cannot be parametric: " << callee;

  FuncType sig = sigGen.getBody();
  // Verify the inputs.
  // Drop the callee and the result slot type for apply_result.
  operands = operands.drop_front();
  ArrayRef<Type> inputTypes = sig.getArguments();
  if (isApplyResult)
    inputTypes = inputTypes.drop_back();

  if (operands.size() != inputTypes.size()) {
    return emitError() << "'apply' function expected " << inputTypes.size()
                       << " inputs but got " << operands.size() << "\n";
  }
  for (auto [i, operand, type] : llvm::enumerate(operands, inputTypes)) {
    Type expected = upbindApplyResult(type);
    // This is a strict type equality check, sugar shouldn't be allowed in the
    // way, otherwise we can't print/parse the operation.
    if (operand.getType() != expected) {
      auto diag = emitError()
                  << "'apply' operand #" << i << " type " << operand.getType()
                  << " does not match expected type " << expected;
      diag.attachNote() << "callee: " << callee;
      return failure();
    }
  }

  return success();
}

static LogicalResult verifyApply(ArrayRef<TypedAttr> operands, Type type,
                                 function_ref<InFlightDiagnostic()> emitError) {
  if (failed(verifyApplyLike(operands, /*isApplyResult=*/false, emitError)))
    return failure();

  // Verify the result.
  auto sig = cast<FuncTypeGeneratorType>(operands.front().getType()).getBody();
  if (sig.getResults().size() != 1)
    return emitError() << "'apply' function must return one result";
  Type resultType = upbindApplyResult(sig.getResults().front());
  if (type != resultType)
    return emitError() << "'apply' function result type must be " << type
                       << " but got " << resultType;

  return success();
}

static LogicalResult
verifyApplyResultSlot(ArrayRef<TypedAttr> operands, Type type,
                      function_ref<InFlightDiagnostic()> emitError) {
  if (failed(verifyApplyLike(operands, /*isApplyResult=*/true, emitError)))
    return failure();

  auto sig = cast<FuncTypeGeneratorType>(operands.front().getType()).getBody();
  // TODO: Cannot check !lit.ref reference types in KGEN.
  auto resultArgType = upbindApplyResult(sig.getArguments().back());
  if (auto resultPtr = dyn_cast<PointerType>(resultArgType)) {
    auto expectedResult = resultPtr.getElementType();
    if (expectedResult != type)
      return emitError() << "'apply_result' function result type must be "
                         << expectedResult << " but got " << type;
  }
  return success();
}

static LogicalResult
verifyVariadicPtrMap(ArrayRef<TypedAttr> operands, Type type,
                     function_ref<InFlightDiagnostic()> emitError) {
  if (operands.size() != 2)
    return emitError() << "'variadic_ptr_map' requires 2 operands";

  auto srcVariadic = dyn_cast<ParamListType>(operands[0].getType());
  if (!srcVariadic || !isa<TypeType, ParamType>(srcVariadic.getElementType()) ||
      type != srcVariadic)
    return emitError() << "'variadic_ptr_map' operand should have "
                          "!kgen.param_list<!kgen.type> type, not "
                       << operands[0].getType();
  if (!operands[1].getType().isIndex())
    return emitError()
           << "'variadic_ptr_map' addr space operand should have 'index' type";

  return success();
}

static LogicalResult
verifyVariadicPtrRemoveMap(ArrayRef<TypedAttr> operands, Type type,
                           function_ref<InFlightDiagnostic()> emitError) {
  if (operands.size() != 1)
    return emitError() << "'variadic_ptrremove_map' requires 1 operand";

  auto srcVariadic = dyn_cast<ParamListType>(operands[0].getType());
  if (!srcVariadic || // May still be parametric
      !isa<TypeType>(srcVariadic.getElementType()))
    return emitError() << "'variadic_ptrremove_map' operand should have "
                          "!kgen.param_list<!kgen.type> type, not "
                       << operands[0].getType();
  auto dstVariadic = dyn_cast<ParamListType>(type);
  if (!dstVariadic || !isa<TypeType>(dstVariadic.getElementType()))
    return emitError() << "'variadic_ptrremove_map' result should be "
                          "!kgen.param_list<!kgen.type> type, not "
                       << type;
  return success();
}

LogicalResult ParamOperatorAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, POC opcode,
    ArrayRef<TypedAttr> operands, Type type) {
  // All the operand types must match except for these operators.
  switch (opcode) {
  case POC::Cond:
  case POC::TargetHasFeature:
  case POC::TargetGetField:
  case POC::GetEnv:
  case POC::GetSizeOf:
  case POC::GetAlignOf:
  case POC::Apply:
  case POC::ApplyResultSlot:
  case POC::Rebind:
  case POC::PtrBitcast:
  case POC::AttrToStr:
  case POC::DataToStr:
  case POC::VariadicPtrMap:
  case POC::VariadicPtrRemoveMap:
    break;
  default:
    if (!llvm::all_of(operands, [&](auto operand) {
          return operand.getType() == operands.front().getType();
        }))
      return emitError() << "operand type mismatch";
  }

  // Check invariants on the expression.
  switch (opcode) {
  case POC::Add:
  case POC::Mul:
  case POC::MulNoWrap:
  case POC::And:
  case POC::Or:
  case POC::Xor:
  case POC::Max:
  case POC::Min:
    if (operands.empty())
      return emitError() << stringifyEnum(opcode)
                         << " operator must have at least one operand";
    if (type != operands[0].getType())
      return emitError() << "result type should match operand types";
    // Check the types that are supported.
    if (type.isIntOrIndex())
      break; // Index and fixed-width integer types supported for all of these.
    return emitError() << "operator requires an index or integer type";
    break;
  // Binary expressions.
  case POC::Shl:
  case POC::Shr:
  case POC::Div:
  case POC::Mod:
  case POC::DivS:
  case POC::DivU:
  case POC::CeilDivS:
  case POC::CeilDivU:
  case POC::FloorDivS:
  case POC::RemS:
  case POC::RemU:
    if (operands.size() != 2)
      return emitError() << stringifyEnum(opcode) << " must have two operands";
    if (type != operands[0].getType())
      return emitError() << "result type should match operand types";
    if (!operands[0].getType().isIntOrIndex())
      return emitError() << "operator requires an index or integer type";
    break;
  case POC::EQ:
    if (operands.size() != 2)
      return emitError() << "comparison operators must have two operands";
    if (!type.isInteger(1))
      return emitError() << "comparisons return i1";
    break;
  case POC::LT:
  case POC::LE:
    if (operands.size() != 2)
      return emitError() << "comparison operators must have two operands";

    if (!type.isInteger(1))
      return emitError() << "comparisons return i1";

    // Relational operations only work on index types.
    if (!operands[0].getType().isIntOrIndex())
      return emitError() << "relational comparisons only allowed on index or "
                            "integer values";
    break;
  case POC::CurrentTarget:
    if (!operands.empty())
      return emitError() << "'current_target' expected no operands";
    if (!llvm::isa<TargetType>(type))
      return emitError() << "'current_target' must return a target type";
    break;
  case POC::TargetHasFeature:
  case POC::TargetGetField:
    if (operands.size() != 2)
      return emitError() << "target_get_field must have two operands";
    if (!llvm::isa<TargetType>(operands[0].getType()))
      return emitError() << "target_get_field operand 0 must be a target type";
    if (!llvm::isa<StringType>(operands[1].getType()))
      return emitError() << "target_get_field operand 1 must be a string type";
    break;
  case POC::CrossCompilation:
    if (!operands.empty())
      return emitError() << "'cross_compilation' expected no operands";
    if (!type.isInteger(1))
      return emitError() << "cross_compilation return i1";
    break;
  case POC::AcceleratorArch:
    if (!operands.empty())
      return emitError() << "'accelerator_arch' expected no operands";
    if (!llvm::isa<StringType>(type))
      return emitError() << "'accelerator_arch' must return a string type";
    break;
  case POC::In:
    if (operands.empty())
      return emitError() << "operator requires at least one operand";
    if (!type.isInteger(1))
      return emitError() << "comparisons return i1";
    break;
  case POC::GetSizeOf:
  case POC::GetAlignOf:
    if (operands.size() != 2) {
      return emitError() << stringifyEnum(opcode)
                         << " operator requires two operands";
    }
    if (!isTypeExpr(operands.front())) {
      return emitError() << stringifyEnum(opcode)
                         << " operand 0 should be a type expression";
    }
    if (!::isa<TargetType>(operands[1].getType())) {
      return emitError() << stringifyEnum(opcode)
                         << " operand 1 should be a !kgen.target";
    }
    if (!::isa<IndexType>(type))
      return emitError() << stringifyEnum(opcode) << " should return index";
    break;
  case POC::Apply:
    if (failed(verifyApply(operands, type, emitError)))
      return failure();
    break;
  case POC::ApplyResultSlot:
    if (failed(verifyApplyResultSlot(operands, type, emitError)))
      return failure();
    break;
  case POC::Rebind:
    if (operands.size() != 1)
      return emitError() << "'rebind' expects one operand";
    break;
  case POC::Cond:
    if (operands.size() != 3)
      return emitError() << "conditional expressions must have three operands";
    if (!operands[0].getType().isInteger(1))
      return emitError() << "conditional expression operand 0 must be i1";
    if (operands[1].getType() != operands[2].getType())
      return emitError() << "conditional expression operands 1 and 2 must have "
                            "the same type";
    if (operands[1].getType() != type)
      return emitError() << "result type should match operands 1 and 2 types";
    break;
  case POC::GetEnv:
    if (operands.size() != 1 || !::isa<StringType>(operands.front().getType()))
      return emitError() << "'get_env' expects one string-typed operand";
    if (auto intType = ::dyn_cast<IntegerType>(type)) {
      if (!intType.isSignlessInteger(1))
        return emitError() << "'get_env' must return index, i1, or string";
    } else if (!::isa<IndexType, StringType>(type)) {
      return emitError() << "'get_env' must return index, i1, or string";
    }
    break;
  case POC::PtrBitcast:
    if (operands.size() != 1)
      return emitError() << "'ptr_bitcast' expects one operand";
    if (!::isa<PointerType>(type) ||
        !::isa<PointerType>(operands.front().getType()))
      return emitError() << "'ptr_bitcast' requires operand and result types "
                            "to both be pointers";
    break;
  case POC::LoadFromMem:
    if (operands.size() != 1)
      return emitError() << "'load_from_mem' expects one operand";
    break;
  case POC::VariadicPtrMap:
    return verifyVariadicPtrMap(operands, type, emitError);
  case POC::VariadicPtrRemoveMap:
    return verifyVariadicPtrRemoveMap(operands, type, emitError);
  case POC::AttrToStr:
    break;
  case POC::DataToStr:
    if (operands.size() != 2)
      return emitError() << "'data_to_str' expects two operands, one "
                            "string slice and a variadic of string slices";

    if (!isEqualCanon(ParamListType::get(operands[0].getType()),
                      operands[1].getType()))
      return emitError() << "'data_to_str' expects two operands, one "
                            "string slice and a variadic of string slices\n"
                         << operands[0].getType() << "\n"
                         << operands[1].getType();

    break;
  case POC::StringAddress:
    if (operands.size() != 1 || !::isa<StringType>(operands[0].getType()))
      return emitError() << "'string_address' expects one '!kgen.string'";
    break;
  case POC::StrConcat:
    // Already checked the input/result types all match.
    if (operands.size() != 2 || !::isa<StringType>(operands[0].getType()))
      return emitError() << "'str_concat' expects two !kgen.string operands";
    break;
  case POC::FunctionGetArgTypes:
    if (operands.size() != 1)
      return emitError() << "'function_get_arg_types' expects one "
                            "!kgen.func operand, but got nothing.";
    auto operand = operands[0];
    if (auto paramRef1 = sugarDynCast<ParamDeclRefAttr>(operand)) {
      if (auto paramRefType1 = sugarDynCast<ParamType>(paramRef1.getType())) {
        auto param1 = paramRefType1.getParam();
        if (!::isa<ParamDeclRefAttr>(param1))
          return emitError() << "'function_get_arg_types' operand paramref's "
                                "type should be a typeconstantattr, but got: "
                             << param1;
      } else {
        return emitError() << "'function_get_arg_types' operand paramref's "
                              "type should be a signature, but got: "
                           << paramRef1.getType();
      }
    } else if (auto typeConstAttr = sugarDynCast<TypeParamAttr>(operand)) {
      auto mlirType = typeConstAttr.getMlirType();
      if (!::isa<FuncTypeGeneratorType>(mlirType))
        return emitError()
               << "'function_get_arg_types' operand typeconstantattr's mlir "
                  "type should be a signature, but got: "
               << mlirType;
    } else if (auto paramIndexRef = sugarDynCast<ParamIndexRefAttr>(operand)) {
      auto mlirType = paramIndexRef.getType();
      if (::isa<ParamType>(mlirType)) {
        // Do nothing, is fine
      } else if (::isa<FuncTypeGeneratorType>(mlirType)) {
        // Do nothing, is fine
      } else
        return emitError()
               << "'function_get_arg_types' operand paramindexref's type "
                  "should be a paramref or signature, but got: "
               << mlirType;
    } else if (::isa<SymbolConstantAttr>(operand)) {
      // Do nothing, is fine.
    } else {
      return emitError()
             << "'function_get_arg_types' expects one kgen.paramref or "
                "typeconstantattr operand, but got: "
             << operand;
    }
    break;
  }
  return success();
}

/// If the specified attribute is a ParamOperatorAttr with the specified opcode,
/// return it.  Otherwise return null.
static ParamOperatorAttr dyn_castPE(POC opcode, TypedAttr value) {
  if (auto expr = sugarDynCast<ParamOperatorAttr>(value))
    if (expr.getOpcode() == opcode)
      return expr;
  return {};
}

/// Treat `index` and signed integers as signed. Treat signless and unsigned
/// integers as unsigned.
static bool isSignedIntType(Type type) {
  return type.isIndex() || type.isSignedInteger();
}

/// Given a function_ref from `(APInt,APInt)->T` and two APInt's, compute the
/// result value T and return it.
///
/// Note that this function has special behavior when 'valueTy' (the MLIR type
/// of the two operand values) is 'index' type. In this case, it does extra work
/// to make sure that a 32-bit and 64-bit target will compute the same result
/// using the same approach as the index dialect.  If they differ, this refuses
/// to fold the operation, returning a null IntegerAttr.
template <typename ResultTy>
static IntegerAttr foldBinaryValues(
    const llvm::function_ref<ResultTy(const APInt &, const APInt &)>
        &unsignedCalculateFn,
    const llvm::function_ref<ResultTy(const APInt &, const APInt &)>
        &signedCalculateFn,
    const APInt &lhs, const APInt &rhs, Type valueTy, Type resultTy = {}) {
  const auto &calculateFn =
      isSignedIntType(valueTy) ? signedCalculateFn : unsignedCalculateFn;

  // Clients can specify resultTy if it differs from valueTy (e.g. for
  // compares), but not specifying it defaults to the result being the same type
  // as the operands.
  if (!resultTy)
    resultTy = valueTy;

  auto result1 = calculateFn(lhs, rhs);
  if (!llvm::isa<IndexType>(valueTy))
    return IntegerAttr::get(resultTy, result1);

  // If this is an index computation, then we just did the 64-bit computation,
  // see what would happen on a 32-bit host.
  assert(lhs.getBitWidth() == 64);

  // We require that the computation satisfy the invariant that:
  //   trunc(f(a, b)) = f(trunc(a), trunc(b))
  auto result2 = calculateFn(lhs.trunc(32), rhs.trunc(32));

  // If not bool result (e.g. a compare), truncate the LHS for our check.
  auto result1test = result1;
  if constexpr (!std::is_same_v<bool, ResultTy>)
    result1test = result1.trunc(result2.getBitWidth());

  // We can use the full 64-bit folded result if they match, otherwise leave
  // unfolded.
  return result1test == result2 ? IntegerAttr::get(resultTy, result1)
                                : IntegerAttr();
}

/// Duplicate the operands in-place for ops like `min` and `max`.
static void deduplicateOperands(SmallVectorImpl<TypedAttr> &operands) {
  llvm::SetVector<TypedAttr, SmallVector<TypedAttr>, SmallPtrSet<Attribute, 4>>
      uniqueOperands(operands.begin(), operands.end());
  operands = uniqueOperands.takeVector();
}

/// Given a fully associative variadic integer operation, constant fold any
/// constant operands and move them to the right.  If the whole expression is
/// constant, then return that, otherwise update the operands list.
static Attribute simplifyAssocOp(
    POC opcode, SmallVectorImpl<TypedAttr> &operands,
    llvm::function_ref<APInt(const APInt &, const APInt &)> unsignedFn,
    llvm::function_ref<APInt(const APInt &, const APInt &)> signedFn = {},
    llvm::function_ref<bool(const APInt &)> identityConstantFn = {},
    llvm::function_ref<bool(const APInt &)> destructiveConstantFn = {},
    bool shouldDeduplicateOperands = false) {
  auto type = operands[0].getType();
  if (operands.size() == 1)
    return operands[0];

  // Flatten any of the same operation into the operand list:
  // `(add x, (add y, z))` => `(add x, y, z)`.
  for (size_t i = 0, e = operands.size(); i != e; ++i) {
    if (auto subexpr = dyn_castPE(opcode, operands[i])) {
      operands[i] = operands.back();
      operands.pop_back();
      --e;
      --i;
      operands.append(subexpr.getOperands().begin(),
                      subexpr.getOperands().end());
    }
  }

  // If allowed, deduplicate operands after flattening
  if (shouldDeduplicateOperands)
    deduplicateOperands(operands);

  // Impose an ordering on the operands, pushing subexpressions to the left and
  // constants to the right, with ParamRefs in the middle - but predictably
  // ordered w.r.t. each other.
  llvm::stable_sort(operands, ParameterAttr::compare);

  // Merge any constants, they will appear at the back of the operand list now.
  if (llvm::isa<IntegerAttr>(operands.back())) {
    while (operands.size() >= 2 &&
           llvm::isa<IntegerAttr>(operands[operands.size() - 2])) {
      APInt c1 =
          llvm::cast<IntegerAttr>(operands[operands.size() - 2]).getValue();
      APInt c2 = llvm::cast<IntegerAttr>(operands.back()).getValue();
      if (auto resultConstant = foldBinaryValues(
              unsignedFn, signedFn ? signedFn : unsignedFn, c1, c2, type)) {
        operands.pop_back();
        operands.pop_back();
        operands.push_back(resultConstant);
      } else {
        // If we couldn't fold the two values, bail.
        break;
      }
    }

    auto resultCst = llvm::cast<IntegerAttr>(operands.back());

    // If the resulting constant is the destructive constant (e.g. `x*0`), then
    // return it.
    if (destructiveConstantFn && destructiveConstantFn(resultCst.getValue()))
      return resultCst;

    // Remove the constant back to our operand list if it is the identity
    // constant for this operator (e.g. `x*1`) and there are other operands.
    if (identityConstantFn && identityConstantFn(resultCst.getValue()) &&
        operands.size() != 1)
      operands.pop_back();
  }

  return operands.size() == 1 ? operands[0] : Attribute();
}

struct DecomposedAddend {
  POC opcode;
  TypedAttr nonConstant;
  IntegerAttr constant;
};

/// Analyze an operand to an add.  If it is a multiplication by a constant (e.g.
/// `(a*b*42)` then split it into the non-constant and the constant portions
/// (e.g. `a*b` and `42`).  Otherwise return the operand as the first value and
/// null as the second (stand-in for "multiplication by 1").
static DecomposedAddend decomposeAddend(TypedAttr operand) {
  auto mul = sugarDynCast<ParamOperatorAttr>(operand);
  if (mul && llvm::is_contained({POC::MulNoWrap, POC::Mul}, mul.getOpcode())) {
    if (auto cst = sugarDynCast<IntegerAttr>(mul.getOperands().back())) {
      auto nonCst = ParamOperatorAttr::get(mul.getOpcode(),
                                           mul.getOperands().drop_back());
      return {mul.getOpcode(), nonCst, cst};
    }
  }

  auto opcode = mul ? mul.getOpcode() : POC::Mul;
  return {opcode, operand, IntegerAttr()};
}

/// Infer the preferred multiplication opcode from two decomposed addends.
/// The goal is to avoid accidentally converting MulNoWrap to the more strict
/// Mul when Mul is not present in the original expression.
static POC inferOpcode(const DecomposedAddend &lhs,
                       const DecomposedAddend &rhs) {
  if (lhs.constant && rhs.constant) {
    if (lhs.opcode == POC::Mul || rhs.opcode == POC::Mul)
      return POC::Mul;
    return POC::MulNoWrap;
  }

  if (lhs.constant)
    return lhs.opcode;
  if (rhs.constant)
    return rhs.opcode;
  return KGEN::POC::Mul;
}

static IntegerAttr getOneOfType(Type type) { return IntegerAttr::get(type, 1); }

static Attribute simplifyAdd(SmallVectorImpl<TypedAttr> &operands) {
  if (auto result = simplifyAssocOp(
          POC::Add, operands, [](auto a, auto b) { return a + b; }, {},
          /*identityCst*/ [](auto cst) { return cst.isZero(); }))
    return result;

  // Canonicalize the add by splitting all addends into their variable and
  // constant factors.
  SmallVector<DecomposedAddend> decomposedOperands;
  llvm::SmallDenseSet<TypedAttr> nonConstantParts;
  for (auto &op : operands) {
    decomposedOperands.push_back(decomposeAddend(op));

    // Keep track of non-constant parts we've already seen.  If we see multiple
    // uses of the same value, then we can fold them together with a multiply.
    // This handles things like `(a+b+a)` => `(a*2 + b)` and `(a*2 + b + a)` =>
    // `(a*3 + b)`.
    if (!nonConstantParts.insert(decomposedOperands.back().nonConstant)
             .second) {
      // The thing we multiply will be the common expression.
      TypedAttr mulOperand = decomposedOperands.back().nonConstant;

      // Find the index of the first occurrence.
      size_t i = 0;
      while (decomposedOperands[i].nonConstant != mulOperand)
        ++i;

      // Remove both occurrences from the operand list.
      operands.erase(operands.begin() + (&op - &operands[0]));
      operands.erase(operands.begin() + i);

      auto type = mulOperand.getType();
      auto c1 = decomposedOperands[i].constant,
           c2 = decomposedOperands.back().constant;

      // Fill in missing constant multiplicands with 1.
      if (!c1)
        c1 = getOneOfType(type);
      if (!c2)
        c2 = getOneOfType(type);

      auto opcode =
          inferOpcode(decomposedOperands[i], decomposedOperands.back());

      // Re-add the "a"*(c1+c2) expression to the operand list and
      // re-canonicalize.
      auto constant = ParamOperatorAttr::get(POC::Add, c1, c2);
      auto mulCst = ParamOperatorAttr::get(opcode, mulOperand, constant);
      operands.push_back(mulCst);
      return ParamOperatorAttr::get(POC::Add, operands);
    }
  }

  return {};
}

static Attribute simplifyGenericMul(SmallVectorImpl<TypedAttr> &operands,
                                    POC opcode) {
  if (auto result = simplifyAssocOp(
          opcode, operands, [](auto a, auto b) { return a * b; }, {},
          /*identityCst*/ [](auto cst) { return cst.isOne(); },
          /*destructiveCst*/ [](auto cst) { return cst.isZero(); }))
    return result;

  // We always build a sum-of-products representation, so if we see an addition
  // as a subexpr, we need to pull it out: (a+b)*c*d ==> (a*c*d + b*c*d).
  for (size_t i = 0, e = operands.size(); i != e; ++i) {
    if (auto addSubExpr = dyn_castPE(POC::Add, operands[i])) {
      // Pull the `c*d` operands out - it is whatever operands remain after
      // removing the `(a+b)` term.
      operands.erase(operands.begin() + i);

      // Build each add operand.
      SmallVector<TypedAttr> addOperands;
      for (auto addOperand : addSubExpr.getOperands()) {
        operands.push_back(addOperand);
        addOperands.push_back(ParamOperatorAttr::get(opcode, operands));
        operands.pop_back();
      }
      // Canonicalize and form the add expression.
      return ParamOperatorAttr::get(POC::Add, addOperands);
    }
  }

  return {};
}

static Attribute simplifyAnd(SmallVectorImpl<TypedAttr> &operands) {
  return simplifyAssocOp(
      POC::And, operands, [](auto a, auto b) { return a & b; }, {},
      /*identityCst*/ [](auto cst) { return cst.isAllOnes(); },
      /*destructiveCst*/ [](auto cst) { return cst.isZero(); },
      /*shouldDeduplicateOperands=*/true);
}

static Attribute simplifyOr(SmallVectorImpl<TypedAttr> &operands) {
  return simplifyAssocOp(
      POC::Or, operands, [](auto a, auto b) { return a | b; }, {},
      /*identityCst*/ [](auto cst) { return cst.isZero(); },
      /*destructiveCst*/ [](auto cst) { return cst.isAllOnes(); },
      /*shouldDeduplicateOperands=*/true);
}

static Attribute simplifyXor(SmallVectorImpl<TypedAttr> &operands) {
  return simplifyAssocOp(
      POC::Xor, operands, [](auto a, auto b) { return a ^ b; }, {},
      /*identityCst*/ [](auto cst) { return cst.isZero(); });
}

/// Returns true if the integer is at its max value.
static bool intIsMaxValue(Type type, const APInt &value) {
  return isSignedIntType(type) ? value.isMaxSignedValue() : value.isMaxValue();
}

/// Returns true if the integer is at its min value.
static bool intIsMinValue(Type type, const APInt &value) {
  return isSignedIntType(type) ? value.isMinSignedValue() : value.isMinValue();
}

static Attribute simplifyMax(SmallVectorImpl<TypedAttr> &operands) {
  Type type = operands.front().getType();
  Attribute maybeConstAttr = simplifyAssocOp(
      POC::Max, operands, llvm::APIntOps::umax, llvm::APIntOps::smax,
      [&](auto cst) { return intIsMinValue(type, cst); },
      [&](auto cst) { return intIsMaxValue(type, cst); },
      /*shouldDeduplicateOperands*/ true);
  if (maybeConstAttr)
    return maybeConstAttr;

  // Add folding rule: max(a*x, a*y) --> a*max(x, y)
  IntegerAttr commonFactor;
  for (TypedAttr operand : operands) {
    // Operand must be a product.
    auto mulAttr = dyn_castPE(POC::MulNoWrap, operand);
    if (!mulAttr)
      return {};

    // The product must end with a constant integer attribute, which (if
    // present) will be canonicalized to be in the back
    auto factor = sugarDynCast<IntegerAttr>(mulAttr.getOperands().back());
    if (!factor)
      return {};

    if (!commonFactor) {
      commonFactor = factor;
      continue;
    }

    // Else we have a common factor from previous operand, the new factor must
    // match it.
    if (commonFactor != factor)
      return {};

    // At this point, the invariant is `operand` is a product that ends with a
    // constant integer, which matches for all previous `operand`s.
  }

  // New operands with the common factor dropped from the end of each product.
  SmallVector<TypedAttr> newOperands;
  for (TypedAttr operand : operands) {
    auto mulAttr = dyn_castPE(POC::MulNoWrap, operand);

    // If the product has the form `x * commonFactor`, the new operand is `x`.
    size_t numOperands = mulAttr.getNumOperands();
    if (numOperands == 2) {
      newOperands.push_back(mulAttr.getOperand(0));
    } else {
      auto newOperand = ParamOperatorAttr::get(
          mulAttr.getOpcode(), mulAttr.getOperands().slice(0, numOperands - 1));
      newOperands.push_back(newOperand);
    }
  }
  auto newMax = ParamOperatorAttr::get(POC::Max, newOperands);
  auto product = ParamOperatorAttr::get(POC::MulNoWrap, {newMax, commonFactor});
  return product;
}

static Attribute simplifyMin(SmallVectorImpl<TypedAttr> &operands) {
  Type type = operands.front().getType();
  return simplifyAssocOp(
      POC::Min, operands, llvm::APIntOps::umin, llvm::APIntOps::smin,
      [&](auto cst) { return intIsMaxValue(type, cst); },
      [&](auto cst) { return intIsMinValue(type, cst); },
      /*shouldDeduplicateOperands*/ true);
}

/// Given a binary function, if the two operands are known constant integers,
/// use the specified fold functions to compute the result.
static Attribute
foldBinaryOp(ArrayRef<TypedAttr> operands,
             llvm::function_ref<APInt(const APInt &, const APInt &)> unsignedFn,
             llvm::function_ref<APInt(const APInt &, const APInt &)> signedFn) {
  assert(operands.size() == 2 && "binary operator always has two operands");
  if (auto lhs = sugarDynCast<IntegerAttr>(operands[0]))
    if (auto rhs = sugarDynCast<IntegerAttr>(operands[1])) {
      if (auto resultConstant =
              foldBinaryValues(unsignedFn, signedFn, lhs.getValue(),
                               rhs.getValue(), lhs.getType()))
        return resultConstant;
    }
  return {};
}

/// Folds constants given a comparison function that returns bool.  The client
/// must handle signedness etc.
static IntegerAttr foldCompareOp(
    TypedAttr lhs, TypedAttr rhs,
    llvm::function_ref<bool(const APInt &, const APInt &)> unsignedCompareFn,
    llvm::function_ref<bool(const APInt &, const APInt &)> signedCompareFn =
        {}) {
  if (auto lhsInt = sugarDynCast<IntegerAttr>(lhs))
    if (auto rhsInt = sugarDynCast<IntegerAttr>(rhs)) {
      if (auto resultConstant = foldBinaryValues(
              unsignedCompareFn,
              signedCompareFn ? signedCompareFn : unsignedCompareFn,
              lhsInt.getValue(), rhsInt.getValue(), lhsInt.getType(),
              IntegerType::get(rhs.getContext(), 1)))
        return resultConstant;
    }
  return {};
}

static SymbolRefAttr getOptionalTypeSymbolRef(Type type) {
  if (auto structIfaceType = dyn_cast<StructTypeInterface>(type))
    return structIfaceType.getSymbolRef();
  if (auto typeValueType = dyn_cast<TypeValueType>(type)) {
    TypedAttr typeRef = typeValueType.getTypeValue();
    if (auto genRef = dyn_cast<TypeGeneratorRefAttr>(typeRef))
      return genRef.getSymbol();
    else if (auto instRef = dyn_cast<TypeInstanceRefAttr>(typeRef))
      return instRef.getSymbol();
  }
  return {};
}

static TypedAttr stripUpcast(TypedAttr a) {
  while (auto u = sugarDynCast<UpcastAttr>(a))
    a = u.getInputTypeValue();
  return a;
}

static Type getTypeValueAsType(TypedAttr a) {
  if (auto tp = dyn_cast<TypeParamAttr>(a))
    return tp.getTypeValue();
  if (auto ref = dyn_cast<ParamDeclRefAttr>(a))
    return ParamType::get(ref);
  return {};
}

/// Compute the result of == for the two specified attributes, handling the
/// index truncation issue but otherwise relying on MLIR's canonicalization of
/// attributes to do the job for us.  Both operands may be null, and this
/// returns null if no folding is possible.
static IntegerAttr foldEquality(TypedAttr lhs, TypedAttr rhs) {
  // This depends on pointer comparison, so sure to strip all sugar.
  lhs = getCanonicalAttr(lhs);
  rhs = getCanonicalAttr(rhs);

  // foldCompareOp handles 32-bit truncation of input values correctly.
  if (lhs.getType().isIndex() && isa<IntegerAttr>(lhs) && isa<IntegerAttr>(rhs))
    return foldCompareOp(lhs, rhs, [](auto a, auto b) { return a == b; });

  // Folding to True is easy: If the values have pointer equality, we know they
  // are equal.
  if (lhs == rhs)
    return BoolAttr::get(rhs.getContext(), true);

  Type lhsTypeVal = getTypeValueAsType(stripUpcast(lhs));
  Type rhsTypeVal = getTypeValueAsType(stripUpcast(rhs));
  if (lhsTypeVal && rhsTypeVal && isEqualCanon(lhsTypeVal, rhsTypeVal))
    return BoolAttr::get(rhs.getContext(), true);

  // Folding to False is a lot harder:
  // If either side contains expression nodes that still need to be evaluated,
  // we cannot fold to False since after evaluation they may become equal.
  // Conservatively we only fold if both sides are simple constants (fully
  // evaluated & contains no parameter references).
  bool lhsSimpleConstant = ParameterAttr::isSimpleConstant(lhs);
  bool rhsSimpleConstant = ParameterAttr::isSimpleConstant(rhs);
  if (lhsSimpleConstant && rhsSimpleConstant)
    return BoolAttr::get(rhs.getContext(), lhs == rhs);

  // Type inequality is a bit stronger due to nominality of struct types.
  // If both sides are type values and they point to different type references,
  // we can fold to False.
  if (auto lhsTypeParam = dyn_cast<TypeParamAttr>(lhs)) {
    if (auto rhsTypeParam = dyn_cast<TypeParamAttr>(rhs)) {
      auto lhsStructRef = getOptionalTypeSymbolRef(lhsTypeParam.getTypeValue());
      auto rhsStructRef = getOptionalTypeSymbolRef(rhsTypeParam.getTypeValue());
      if (lhsStructRef && rhsStructRef) {
        // Both sides are struct types. If the referenced symbols are different,
        // they are never equal.
        if (lhsStructRef != rhsStructRef)
          return BoolAttr::get(rhs.getContext(), false);
      } else if (static_cast<bool>(lhsStructRef) !=
                 static_cast<bool>(rhsStructRef)) {
        // If one side is a struct type and the other is not, we can only fold
        // to false if the non-struct side is already fully evaluated. Otherwise
        // we do not yet know whether the non-struct side will evaluate to a
        // struct.
        if (lhsStructRef && rhsSimpleConstant)
          return BoolAttr::get(rhs.getContext(), false);
        if (rhsStructRef && lhsSimpleConstant)
          return BoolAttr::get(rhs.getContext(), false);
      }
    }
  }
  // Otherwise can't fold something like "x == y".
  return {};
}

static Attribute simplifyShl(SmallVectorImpl<TypedAttr> &operands) {
  // Canonicalize `x << cst` => `x * (1<<cst)` to compose correctly with
  // add/mul canonicalization (also handles constant folding).
  if (auto rhs = sugarDynCast<IntegerAttr>(operands[1])) {
    // NOTE: This is correct even for index types because an overlong shift will
    // turn the result to zero.
    if (rhs.getValue().getZExtValue() >= rhs.getValue().getBitWidth())
      return IntegerAttr::get(rhs.getType(), 0);

    auto rhsCst = APInt::getOneBitSet(rhs.getValue().getBitWidth(),
                                      rhs.getValue().getZExtValue());
    return ParamOperatorAttr::get(POC::Mul, operands[0],
                                  IntegerAttr::get(rhs.getType(), rhsCst));
  }
  return {};
}

static Attribute simplifyShr(SmallVectorImpl<TypedAttr> &operands) {
  if (auto rhs = sugarDynCast<IntegerAttr>(operands[1]))
    if (rhs.getValue().isZero())
      return operands[0]; // `x >> 0 = x`.
  // TODO: 0 >> x, -1 >>> x

  // FIXME: Must care about high bits.
  return foldBinaryOp(
      operands, [](auto a, auto b) { return a.lshr(b); },
      [](auto a, auto b) { return a.ashr(b); });
}

/// Tracks the operands of MulNoWrap in a form which allows easy simplification.
namespace {
struct DivOperandInfo {
  // tracks the occurrences of non-integral operands only, e.g. D1
  SmallDenseMap<TypedAttr, size_t> symOccurrences;

  // tracks the coalesced constant terms, e.g. mul_no_wrap(5, 10, D1)
  // this would be 5 * 10 = 50.
  APInt constant;

  Type attrType;

  // whether folding of `constant` leads to overflow on the current system
  // OR initialized with wrong attr (only support IntegerAttr and MulNoWrap)
  // OR when dealing with potential expressions which are sufficiently large
  //    as to differ in behavior on 32/64 bit systems
  bool isPoisoned = false;

  // Multiplies `constant` by `num`, checking overflow
  inline void updateConstant(IntegerAttr integerAttr) {
    APInt num = integerAttr.getValue();

    if (num == 0) {
      // the power of 0 -- it's always going to be 0!
      constant = 0;

      // TODO: think of this more
      isPoisoned = false;
      return;
    }

    bool overflow = false;
    constant = isSignedIntType(integerAttr.getType())
                   ? constant.smul_ov(num, overflow)
                   : constant.umul_ov(num, overflow);

    bool isIndex = attrType.isIndex();

    // poison if overflow on the current system OR would overflow on
    // 32 bit system for `index` types
    isPoisoned = isPoisoned || overflow ||
                 (isIndex && (constant.trunc(32).sext(64) != constant));
  }

  /// Construct an Info object using a MulNoWrap operator, or constant
  /// IntegerAttr
  DivOperandInfo(TypedAttr attr) {
    constant = IntegerAttr::get(attr.getType(), 1).getValue();
    attrType = attr.getType();

    if (auto constAttr = sugarDynCast<IntegerAttr>(attr)) {
      updateConstant(constAttr);
      return;
    }

    if (auto mulAttr = dyn_castPE(POC::MulNoWrap, attr)) {
      for (TypedAttr numOpAttr : mulAttr.getOperands()) {
        if (auto constAttr = sugarDynCast<IntegerAttr>(numOpAttr)) {
          updateConstant(constAttr);
        } else {
          ++symOccurrences[numOpAttr];
        }
      }
      return;
    }

    if (auto declAttr = sugarDynCast<KGEN::ParamDeclRefAttr>(attr)) {
      ++symOccurrences[declAttr];
      return;
    }

    // Not supported attr
    isPoisoned = true;
  }

  /// Create a new MulNoWrap expression from the info stored. If no symbolic
  /// variables are left, return an IntegerAttr, else return a MulNoWrap
  TypedAttr getExpression() {
    SmallVector<TypedAttr> operands;

    IntegerAttr constTerm = IntegerAttr::get(attrType, constant);

    operands.push_back(constTerm);
    for (auto [operand, occurrences] : symOccurrences)
      operands.append(occurrences, operand);

    if (operands.size() == 1) {
      // Implies `constant` only term
      return constTerm;
    }

    return ParamOperatorAttr::get(POC::MulNoWrap, operands);
  }

  /// Simplify terms in `numerator` and `denominator` assuming deriving terms
  /// are dividing each other. Mutates operands in place.
  static void simplifyDivInPlace(DivOperandInfo &numerator,
                                 DivOperandInfo &denominator) {
    SmallDenseMap<TypedAttr, size_t> &numeratorOperandOccurrences =
        numerator.symOccurrences;
    SmallDenseMap<TypedAttr, size_t> &denominatorOperandOccurrences =
        denominator.symOccurrences;

    // Emulate cancelling out shared operand(s) by decrementing their
    // occurrences. e.g., for
    //   `mul_no_wrap(D0, D2, D0)` with occurrence mapping `{ D0 : 2, D2 : 1 }`.
    //   `mul_no_wrap(D2, D0, D2)` with occurrence mapping `{ D0 : 1, D2 : 2 }`.
    // the new occurrence mappings are
    //   `{ D0 : 1, D2 : 0 }`.
    //   `{ D0 : 0, D2 : 1 }`.
    for (auto [numOpAttr, occurrences] : numeratorOperandOccurrences) {
      if (size_t denomOccurrences =
              denominatorOperandOccurrences.lookup(numOpAttr)) {
        size_t sharedOccurrences = std::min(occurrences, denomOccurrences);
        numeratorOperandOccurrences[numOpAttr] -= sharedOccurrences;
        denominatorOperandOccurrences[numOpAttr] -= sharedOccurrences;
      }
    }

    // Cancel out the constant terms
    if (numerator.constant == 0) {
      numerator.symOccurrences.clear();
    }
    if (denominator.constant == 0) {
      denominator.symOccurrences.clear();
    }
    if (numerator.constant != 0 && denominator.constant != 0) {
      bool isSigned = isSignedIntType(numerator.attrType);

      APInt gcdTerm = llvm::APIntOps::GreatestCommonDivisor(
          isSigned ? numerator.constant.abs() : numerator.constant,
          isSigned ? denominator.constant.abs() : denominator.constant);

      bool bothNegative = isSigned && numerator.constant.isNegative() &&
                          denominator.constant.isNegative();

      if (bothNegative)
        gcdTerm = -gcdTerm;

      numerator.constant = isSigned ? numerator.constant.sdiv(gcdTerm)
                                    : numerator.constant.udiv(gcdTerm);
      denominator.constant = isSigned ? denominator.constant.sdiv(gcdTerm)
                                      : denominator.constant.udiv(gcdTerm);
    }
  }
};

} // namespace

/// If the numerator is an `Add(t1, t2, ...)` and the denominator cleanly
/// divides every term, distribute the division across the sum and reduce the
/// denominator to 1. Returns true on success.
///
/// Without this, `(a*b*K + c*K) / K` gets stuck because `mul_no_wrap`
/// distributes over `Add` during simplification, leaving a sum whose shared
/// factor `K` can't be pulled back out of the `Div` folder.
static bool tryDistributeDivOverAdd(SmallVectorImpl<TypedAttr> &operands) {
  assert(operands.size() == 2 && "div has exactly two operands");

  ParamOperatorAttr addAttr = dyn_castPE(POC::Add, operands[0]);
  if (!addAttr)
    return false;

  TypedAttr denominatorAttr = operands[1];
  SmallVector<TypedAttr> quotients;
  quotients.reserve(addAttr.getOperands().size());
  for (TypedAttr term : addAttr.getOperands()) {
    DivOperandInfo numInfo(term);
    DivOperandInfo denomInfo(denominatorAttr);
    if (numInfo.isPoisoned || denomInfo.isPoisoned)
      return false;

    DivOperandInfo::simplifyDivInPlace(numInfo, denomInfo);

    // Only distribute if this term fully cancels the denominator.
    auto denomConst = sugarDynCast<IntegerAttr>(denomInfo.getExpression());
    if (!denomConst || !denomConst.getValue().isOne())
      return false;

    quotients.push_back(numInfo.getExpression());
  }

  operands[0] = ParamOperatorAttr::get(POC::Add, quotients);
  operands[1] = IntegerAttr::get(denominatorAttr.getType(), 1);
  return true;
}

/// Simplify division operands by cancelling out shared elements within
/// numerator and denominator products, e.g., `(a*b)/(b*b) --> a/b`
static void simplifyDivOperands(SmallVectorImpl<TypedAttr> &operands) {
  if (tryDistributeDivOverAdd(operands))
    return;

  TypedAttr &numeratorAttr = operands[0];
  TypedAttr &denominatorAttr = operands[1];

  // Build mapping from each MulNoWrap op operand to the number of its
  // occurrences, e.g., for `mul_no_wrap(D0, 42, D0)`, we build the mapping `{
  // D0 : 2}, constant: 42`
  DivOperandInfo numeratorInfo = DivOperandInfo(numeratorAttr);
  DivOperandInfo denominatorInfo = DivOperandInfo(denominatorAttr);

  // Poisoning: implies overflow in folding of constant @ precision of int64_t:
  //     e.g. mul_no_wrap(1e18, 1e18, D1) --> 1e90
  // Or numerator/denominator is is not a MulNoWrap or an IntegerAttr
  if (numeratorInfo.isPoisoned || denominatorInfo.isPoisoned)
    return;

  DivOperandInfo::simplifyDivInPlace(numeratorInfo, denominatorInfo);

  operands[0] = numeratorInfo.getExpression();
  operands[1] = denominatorInfo.getExpression();
}

static Attribute simplifyDiv(SmallVectorImpl<TypedAttr> &operands) {
  simplifyDivOperands(operands);

  // Implement support for identities like `x/1 = x` and guard against `x/0`
  if (auto rhs = sugarDynCast<IntegerAttr>(operands[1])) {
    if (rhs.getValue().isOne())
      return operands[0];
    if (rhs.getValue().isZero())
      return {};
  }

  return foldBinaryOp(
      operands, [](auto a, auto b) { return a.udiv(b); },
      [](auto a, auto b) { return a.sdiv(b); });
}

static llvm::APInt ceilSDiv(const llvm::APInt &a, const llvm::APInt &b) {
  assert(!b.isZero() && "Division by zero!");
  llvm::APInt q = a.sdiv(b);
  llvm::APInt r = a.srem(b);
  if (!r.isZero() && ((a.isNegative() == b.isNegative())))
    q += llvm::APInt(a.getBitWidth(), 1);
  return q;
}

static llvm::APInt ceilUDiv(const llvm::APInt &a, const llvm::APInt &b) {
  assert(!b.isZero() && "Division by zero!");
  return (a + b - llvm::APInt(a.getBitWidth(), 1)).udiv(b);
}

static Attribute simplifyCeilDiv(SmallVectorImpl<TypedAttr> &operands) {
  simplifyDivOperands(operands);

  // Implement support for identities like `x/1 = x` and guard against `x/0`
  if (auto rhs = sugarDynCast<IntegerAttr>(operands[1])) {
    if (rhs.getValue().isOne())
      return operands[0];
    if (rhs.getValue().isZero())
      return {};
  }

  return foldBinaryOp(
      operands, [](auto a, auto b) { return ceilUDiv(a, b); },
      [](auto a, auto b) { return ceilSDiv(a, b); });
}

static llvm::APInt floorSDiv(const llvm::APInt &a, const llvm::APInt &b) {
  assert(!b.isZero() && "Division by zero!");
  llvm::APInt q = a.sdiv(b);
  llvm::APInt r = a.srem(b);
  if (!r.isZero() && ((a.isNegative() != b.isNegative())))
    q -= llvm::APInt(a.getBitWidth(), 1);
  return q;
}

static Attribute simplifyFloorDiv(SmallVectorImpl<TypedAttr> &operands) {
  simplifyDivOperands(operands);

  // Implement support for identities like `x/1 = x` and guard against `x/0`
  if (auto rhs = sugarDynCast<IntegerAttr>(operands[1])) {
    if (rhs.getValue().isOne())
      return operands[0];
    if (rhs.getValue().isZero())
      return {};
  }

  return foldBinaryOp(
      operands, [](auto a, auto b) { return floorSDiv(a, b); },
      [](auto a, auto b) { return floorSDiv(a, b); });
}

static Attribute simplifyMod(SmallVectorImpl<TypedAttr> &operands) {
  TypedAttr lhs = operands[0];
  TypedAttr rhs = operands[1];

  // Check whether `x` is a multiple of `y`, only for simple cases.
  auto isMultipleOf = [](TypedAttr x, TypedAttr y) {
    if (x == y)
      return true;

    ArrayRef<TypedAttr> xProductOperands;
    if (auto mulAttr = dyn_castPE(POC::Mul, x))
      xProductOperands = mulAttr.getOperands();
    if (auto mulAttr = dyn_castPE(POC::MulNoWrap, x))
      xProductOperands = mulAttr.getOperands();
    return llvm::is_contained(xProductOperands, y);
  };

  // Add folding rule `(n * x) % x = 0` for `x` of integer type.
  if (lhs.getType().isIntOrIndex() && isMultipleOf(lhs, rhs))
    return IntegerAttr::get(rhs.getType(), 0);

  // Implement support for identities like `x%1 = 0`
  if (auto rhs = sugarDynCast<IntegerAttr>(operands[1])) {
    if (rhs.getValue().isOne())
      return IntegerAttr::get(rhs.getType(), 0);
    if (rhs.getValue().isZero())
      return {};
  }

  return foldBinaryOp(
      operands, [](auto a, auto b) { return a.urem(b); },
      [](auto a, auto b) { return a.srem(b); });
}

static Attribute simplifyEQ(SmallVectorImpl<TypedAttr> &operands) {
  // Make sure parameters are ordered correctly, which also matters if they
  // don't fold.
  llvm::stable_sort(operands, ParameterAttr::compare);

  return foldEquality(operands[0], operands[1]);
}

/// Simplify the < and <= operations.
static Attribute
simplifyRelationalCompare(POC opcode, SmallVectorImpl<TypedAttr> &operands) {
  auto rhs = sugarDynCast<IntegerAttr>(operands[1]);
  auto lhs = sugarDynCast<IntegerAttr>(operands[0]);

  if (rhs && !lhs) {
    // If this is a `(le x, RHS)` and RHS is a constant, canonicalize to `lt`.
    if (opcode == POC::LE) {
      if (intIsMaxValue(rhs.getType(), rhs.getValue())) // x <= 127 --> TRUE.
        return BoolAttr::get(rhs.getContext(), true);
      return ParamOperatorAttr::get(
          POC::LT, operands[0],
          IntegerAttr::get(rhs.getType(), rhs.getValue() + 1));
    }
    // If this is (x < MAXCST) canonicalize to (x != MAXCST).
    if (intIsMaxValue(rhs.getType(), rhs.getValue()))
      return ParamOperatorAttr::getNE(operands[0], rhs);
  }

  if (lhs && !rhs) {
    // (le cst, x) -> !(lt x, cst)
    if (opcode == POC::LE)
      return ParamOperatorAttr::getNot(
          ParamOperatorAttr::get(POC::LT, operands[1], operands[0]));
    // (lt cst, x) -> !(le x, cst)
    return ParamOperatorAttr::getNot(
        ParamOperatorAttr::get(POC::LE, operands[1], operands[0]));
  }

  if (opcode == POC::LT)
    return foldCompareOp(
        operands[0], operands[1], [](auto a, auto b) { return a.ult(b); },
        [](auto a, auto b) { return a.slt(b); });
  assert(opcode == POC::LE);
  return foldCompareOp(
      operands[0], operands[1], [](auto a, auto b) { return a.ule(b); },
      [](auto a, auto b) { return a.sle(b); });
}

static Attribute simplifyHasFeature(SmallVectorImpl<TypedAttr> &operands) {
  auto target = sugarDynCast<TargetParamAttr>(operands[0]);
  auto feature = sugarDynCast<StringAttr>(operands[1]);
  if (!target || !feature)
    return {};
  return Builder(target.getContext())
      .getBoolAttr(target.getTarget().hasFeature(feature));
}

static Attribute simplifyTargetGetField(SmallVectorImpl<TypedAttr> &operands,
                                        Type &resultType) {
  auto target = sugarDynCast<TargetParamAttr>(operands[0]);
  auto field = sugarDynCast<StringAttr>(operands[1]);
  if (!field)
    return {};

  Builder b(field.getContext());
  if (llvm::is_contained<StringRef>({"triple", "os", "arch", "endianness"},
                                    field))
    resultType = b.getType<StringType>();
  else
    resultType = b.getType<IndexType>();

  if (!target)
    return {};

  if (field.getValue() == "triple") {
    return StringAttr::get(target.getTarget().getTriple().getTriple(),
                           resultType);
  }
  if (field.getValue() == "os")
    return StringAttr::get(target.getTarget().getOS(), resultType);
  if (field.getValue() == "arch")
    return StringAttr::get(target.getTarget().getArch(), resultType);
  if (field.getValue() == "simd_bit_width")
    return b.getIndexAttr(target.getTarget().getSimdBitWidth());
  if (field.getValue() == "index_bit_width")
    return b.getIndexAttr(target.getTarget().resolveIndexBitWidth());
  if (field.getValue() == "endianness") {
    return StringAttr::get(
        target.getTarget().getTriple().isLittleEndian() ? "little" : "big",
        resultType);
  }
  return {};
}

/// Simplifies an `in` (also `in(:dtype`) operator.  We know the all the
/// operands have the same type.
static Attribute simplifyIn(SmallVectorImpl<TypedAttr> &operands) {
  TypedAttr lhs = operands[0];
  MutableArrayRef<TypedAttr> trailing =
      llvm::MutableArrayRef(operands).drop_front();

  Builder b(lhs.getContext());

  // If there are no trailing operands, fold to false.
  if (trailing.empty())
    return b.getBoolAttr(false);

  // If there is only one trailing operand, canonicalize to an `eq` operator.
  if (trailing.size() == 1)
    return ParamOperatorAttr::get(POC::EQ, operands);

  bool allKnownFalse = true;
  for (TypedAttr operand : trailing) {
    // Fold to true if a match was found by value.
    if (auto knownEq = foldEquality(lhs, operand)) {
      if (knownEq.getValue().isOne())
        return knownEq;
    } else if (lhs == operand) {
      // Fold to true if they match symbolically, like "x+1" and "x+1".
      return b.getBoolAttr(true);
    } else {
      // If this is a symbolic comparison like "x == 5", then we cannot fold the
      // non-containment case.
      allKnownFalse = false;
    }
  }

  // Ok, we know that LHS isn't known to equal any member of the set, but it or
  // they might be symbolic.  If we know for sure that LHS *isn't* equal to any
  // of the elements in the set then we can fold to false.
  if (allKnownFalse)
    return b.getBoolAttr(false);

  // Sort and unique the trailing operands.
  llvm::stable_sort(trailing, ParameterAttr::compare);
  SmallVector<TypedAttr> newOperands;
  newOperands.reserve(operands.size());
  newOperands.push_back(lhs);
  SmallPtrSet<Attribute, 4> seenTrailing;
  for (TypedAttr operand : trailing)
    if (seenTrailing.insert(operand).second)
      newOperands.push_back(operand);
  if (newOperands == operands)
    return {};
  return ParamOperatorAttr::get(POC::In, newOperands);
}

/// Simplifies a `get_sizeof` operator. Try to narrow the operand to a type
/// constant. If it does, query its data layout.
static Attribute simplifyGetSizeOf(SmallVectorImpl<TypedAttr> &operands,
                                   Type &resultType) {
  Builder b(operands[0].getContext());
  if (!resultType)
    resultType = b.getIndexType();

  auto typeCst = sugarDynCast<TypeParamAttr>(operands[0]);
  auto target = sugarDynCast<TargetParamAttr>(operands[1]);
  if (!typeCst || !target)
    return {};
  std::optional<int64_t> size = DataLayoutInterface::getTypeStoreSize(
      target.getTarget(), typeCst.getMlirType());
  if (!size)
    return {};

  return b.getIndexAttr(*size);
}

/// Simplifies a `get_alignof` operator. Try to narrow the operand to a type
/// constant. If it does, query its data layout.
static Attribute simplifyGetAlignOf(SmallVectorImpl<TypedAttr> &operands,
                                    Type &resultType) {
  Builder b(operands[0].getContext());
  if (!resultType)
    resultType = b.getIndexType();

  auto typeCst = sugarDynCast<TypeParamAttr>(operands[0]);
  auto target = sugarDynCast<TargetParamAttr>(operands[1]);
  if (!typeCst || !target)
    return {};
  std::optional<int64_t> size = DataLayoutInterface::getTypeABIAlign(
      target.getTarget(), typeCst.getMlirType());
  if (!size)
    return {};

  return b.getIndexAttr(*size);
}

// We want data_to_str to maintain a simple invariant without sugar getting in
// the way. Just fix up the sugar here so the stdlib doesn't have to deal with
// it.
static void simplifyDataToStr(MutableArrayRef<TypedAttr> operands) {
  assert(operands.size() == 2);
  auto op1Type = ParamListType::get(operands[0].getType());
  assert(isEqualCanon(operands[1].getType(), op1Type));
  if (operands[1].getType() != op1Type)
    operands[1] = ParamOperatorAttr::getRebind(operands[1], op1Type);
}

static Attribute simplifyApply(ArrayRef<TypedAttr> operands, Type &resultType) {
  TypedAttr func = operands.front();
  operands = operands.drop_front();
  // Take the result type.
  resultType = upbindApplyResult(cast<FuncTypeGeneratorType>(func.getType())
                                     .getBody()
                                     .getValues()
                                     .getResult(0));

  if (auto opExpr = dyn_cast<MLIROpAttr>(func)) {
    // Make the operation real by materializing it into a fake block.
    // HACK: Should we be materializing IR inside an attribute's constructor?
    // Maybe defer this to the interpreter.
    auto block = std::make_unique<Block>();
    SmallVector<Value> fakeOperands;
    auto loc = UnknownLoc::get(func.getContext());
    for (Type type : opExpr.getType().getBody().getArguments())
      fakeOperands.push_back(block->addArgument(type, loc));
    OwningOpRef<Operation *> op = Operation::create(
        loc, {opExpr.getName(), func.getContext()},
        opExpr.getType().getBody().getResults(), fakeOperands,
        opExpr.getAttrs(), /*properties=*/mlir::PropertyRef{});
    block->push_back(*op);
    // Verify the operation. Fail to fold if the operation is invalid. Silence
    // the error, since there is no way to report it.
    mlir::ScopedDiagnosticHandler handler(
        func.getContext(),
        [&](Diagnostic &diag) -> LogicalResult { return success(); });
    if (failed(mlir::verify(*op)))
      return {};
    SmallVector<OpFoldResult> results;
    SmallVector<Attribute> attrs;
    llvm::append_range(attrs, operands);
    if (failed((*op)->fold(attrs, results)))
      return {};
    assert(results.size() == 1 && "expected one operation result");
    if (auto result = results.front().dyn_cast<Attribute>())
      return cast<TypedAttr>(result);
    // If the fold hook returned an operation, just look up the corresponding
    // input.
    return operands[cast<BlockArgument>(cast<Value>(results.front()))
                        .getArgNumber()];
  }

  return {};
}

static TypedAttr simplifyRebind(ArrayRef<TypedAttr> operands, Type resultType) {
  assert(resultType && "rebind requires a result type");
  TypedAttr input = operands.front();
  if (input.getType() == resultType)
    return input;
  // Fold rebinds of an unbound.
  if (isa<UnboundAttr>(input))
    return UnboundAttr::get(resultType);
  if (isa<UnknownAttr>(input))
    return UnknownAttr::get(resultType);

  // If we're rebinding a sugared form to a de-sugared type, just strip sugar.
  // This ensures folding logic is never interrupted by de-sugaring rebinds.
  TypedAttr canonicalInput = getCanonicalAttr(input);
  if (canonicalInput.getType() == resultType)
    return canonicalInput;

  // Fold rebinds of a StructType. Unify metatypes so information is not lost.
  if (auto typeCst = sugarDynCast<TypeParamAttr>(input))
    return TypeParamAttr::get(typeCst.getTypeValue(), typeCst.getMlirType(),
                              resultType);
  // rebind(rebind(x)) => rebind(x)
  if (auto x = ParamOperatorAttr::stripRebind(input); x != input)
    return ParamOperatorAttr::get(POC::Rebind, x, resultType);

  return {};
}

// Returns the op with operands replaced with substitutions with actual values.
// substitutions are automatically added in conditionals with equals.
//
// E.g. given substitutions = {C: 3} calling this on
//  cond(eq(A == 2 and C == 3), f(A, C), g(A, C))
//
// Will become cond(eq(A == 2 and 3 == 3), f(A, 3), g(A, 3))
//
// We are able to apply the substitution `A == 2` for the "then" branch
// Note the substitution from the equal can only be applied to the then branch.
//
// This substitution and walking is done in a recursive manner. The depth
// of this recursion is bound by the initial `depth_left` parameter.
static TypedAttr cloneOperandsWithSubstitution(
    TypedAttr op, const DenseMap<TypedAttr, IntegerAttr> &substitutions,
    size_t depth_left) {
  if (substitutions.contains(op))
    return substitutions.at(op);

  if (depth_left <= 0)
    return op;

  auto opParamOperator = sugarDynCast<ParamOperatorAttr>(op);
  if (!opParamOperator)
    return op;

  // Constrain to IntegerAttr substitutions for now
  SmallVector<TypedAttr> newOperands;
  bool hasChanged = false;
  for (TypedAttr oldOperand : opParamOperator.getOperands()) {
    TypedAttr newOperand = cloneOperandsWithSubstitution(
        oldOperand, substitutions, depth_left - 1);
    newOperands.push_back(newOperand);
    hasChanged = hasChanged || (newOperand != oldOperand);
  }

  // No changes
  if (!hasChanged)
    return opParamOperator;

  auto result =
      ParamOperatorAttr::get(opParamOperator.getOpcode(), newOperands);
  return result;
}

static TypedAttr simplifyCond(ArrayRef<TypedAttr> operands) {
  TypedAttr condAttr = operands[0];
  TypedAttr thenAttr = operands[1];
  TypedAttr elseAttr = operands[2];
  if (thenAttr == elseAttr)
    return thenAttr;

  // cond(A != B, A, B) == A
  //
  // But A != B is represented as Xor(A == B, true)
  if (auto xorAttr = dyn_castPE(POC::Xor, condAttr)) {
    auto eqAttr = dyn_castPE(POC::EQ, xorAttr.getOperand(0));
    auto intAttr = sugarDynCast<IntegerAttr>(xorAttr.getOperand(1));
    if (eqAttr && intAttr && intAttr.getValue().isOne() &&
        eqAttr.getOperand(0) == thenAttr && eqAttr.getOperand(1) == elseAttr)
      return thenAttr;
  }

  // cond(A == B, B, A) == A
  if (auto eqAttr = dyn_castPE(POC::EQ, condAttr)) {
    auto lhsEq = eqAttr.getOperand(0);
    auto rhsEq = eqAttr.getOperand(1);
    if (thenAttr == rhsEq && elseAttr == lhsEq)
      return lhsEq;

    auto rhsEqAsIntegral = sugarDynCast<IntegerAttr>(rhsEq);
    auto lhsEqAsIntegral = sugarDynCast<IntegerAttr>(lhsEq);

    // If in form cond(A == 5, f(A, ...), ...)
    // Substitute all occurrences of A in the then branch with '5' up to
    // `MAX_RECURSION_DEPTH`
    const static size_t maxRecursionDepth = 3;
    if (rhsEqAsIntegral && !lhsEqAsIntegral) {
      DenseMap<TypedAttr, IntegerAttr> substitutions = {
          {lhsEq, rhsEqAsIntegral}};
      TypedAttr newThenAttr = cloneOperandsWithSubstitution(
          thenAttr, substitutions, maxRecursionDepth);
      if (newThenAttr != thenAttr)
        return ParamOperatorAttr::get(POC::Cond,
                                      {condAttr, newThenAttr, elseAttr});
    }
  }

  // cond(X, false, X) == X
  if (auto then = sugarDynCast<IntegerAttr>(thenAttr))
    if (then.getValue().isZero() && condAttr == elseAttr)
      return thenAttr;

  auto c = sugarDynCast<IntegerAttr>(condAttr);
  if (!c)
    return {};
  if (c.getValue().isOne())
    return thenAttr;
  if (c.getValue().isZero())
    return elseAttr;
  return {};
}

static TypedAttr simplifyPtrBitcast(ArrayRef<TypedAttr> operands,
                                    Type resultType) {
  if (operands.front().getType() == resultType)
    return operands.front();
  if (auto ptr = sugarDynCast<PointerAttr>(operands.front()))
    return PointerAttr::get(ptr.getAddr(), resultType);
  return {};
}

static TypedAttr simplifyLoadFromMem(ArrayRef<TypedAttr> operands,
                                     Type resultType) {
  TypedAttr ptrValue = operands.front();

  // If we get a PointerAttr, then it must not be mapped to any persistent
  // memory. There is nothing we can ever do with it. Return a UninitMemAttr
  // value.
  if (sugarIsa<PointerAttr>(ptrValue))
    return UninitMemAttr::get(resultType);

  // If the operand is an immediate store_to_mem, just return the value.
  if (auto storeToMem = sugarDynCast<StoreToMemAttr>(ptrValue))
    return storeToMem.getValue();

  return {};
}

static TypedAttr simplifyVariadicPtrMap(TypedAttr variadicOperand,
                                        TypedAttr addrSpaceOperand,
                                        Type resultType) {
  // Fold a concrete variadic list of types.
  auto variadic = sugarDynCast<ParamListAttr>(variadicOperand);
  if (!variadic)
    return {};

  auto resultEltType = cast<ParamListType>(resultType).getElementType();

  SmallVector<TypedAttr> results;
  // Map each type to PointerType of their type, retaining their metatype.
  for (auto elt : variadic.getValues()) {
    Type typeValue =
        PointerType::get(TypeValueType::get(elt), addrSpaceOperand);
    Type mlirType = PointerType::get(ParamType::get(elt), addrSpaceOperand);
    results.push_back(TypeParamAttr::get(typeValue, mlirType, resultEltType));
  }

  return ParamListAttr::get(results, cast<ParamListType>(resultType));
}

static TypedAttr simplifyVariadicPtrRemoveMap(TypedAttr variadicOperand,
                                              Type resultType) {
  // Fold a concrete variadic list of types.
  auto variadic = sugarDynCast<ParamListAttr>(variadicOperand);
  if (!variadic)
    return {};

  auto resultEltType = cast<ParamListType>(resultType).getElementType();

  SmallVector<TypedAttr> results;
  // Map each type from a PointerType of the element type.
  for (auto elt : variadic.getValues()) {
    auto eltCst = sugarDynCast<TypeParamAttr>(elt);
    if (!eltCst || !isa<PointerType>(eltCst.getMlirType()))
      return {};

    results.push_back(TypeParamAttr::get(
        cast<PointerType>(eltCst.getTypeValue()).getElementType(),
        cast<PointerType>(eltCst.getMlirType()).getElementType(),
        resultEltType));
  }
  return ParamListAttr::get(results, cast<ParamListType>(resultType));
}

static TypedAttr simplifyStrConcat(TypedAttr lhs, TypedAttr rhs) {
  auto lhsS = sugarDynCast<StringAttr>(lhs);
  if (!lhsS)
    return {};

  auto rhsS = sugarDynCast<StringAttr>(rhs);
  if (!rhsS)
    return {};
  SmallString<80> buffer;
  buffer.reserve(lhsS.size() + rhsS.size());
  buffer.append(lhsS.strref());
  buffer.append(rhsS.strref());
  return StringAttr::get(buffer, lhs.getType());
}

static TypedAttr simplifyFunctionGetArgTypes(MLIRContext *ctx,
                                             TypedAttr operand,
                                             Type resultType) {
  assert(resultType && "function_get_arg_types requires a result type");

  if (!::isa<ParamListType>(resultType))
    return {};
  auto variadicType = sugarDynCast<ParamListType>(resultType);
  auto traitType = variadicType.getElementType();

  Type mlirType;

  if (auto paramRef1 = sugarDynCast<ParamDeclRefAttr>(operand))
    return {};

  else if (auto typeConstAttr = sugarDynCast<TypeParamAttr>(operand)) {
    mlirType = typeConstAttr.getMlirType();
  } else if (auto paramIndexRef = sugarDynCast<ParamIndexRefAttr>(operand)) {
    mlirType = paramIndexRef.getType();
  } else if (auto symConstAttr = sugarDynCast<SymbolConstantAttr>(operand)) {
    mlirType = symConstAttr.getType();
  } else {
    return {};
  }

  ArrayRef<Type> argTypes;
  if (auto sigGen = sugarDynCast<FuncTypeGeneratorType>(mlirType))
    argTypes = sigGen.getBody().getArguments();
  else
    return {};

  SmallVector<TypedAttr> results;
  // TODO(MOCO-1106): Add a vtable here, see
  // https://www.notion.so/modularai/1571044d37bb80198d96f6772ebb1515
  for (Type type : argTypes)
    results.push_back(TypeParamAttr::get(type, traitType));
  return ParamListAttr::get(results, variadicType);
}

/// Construct a parameter operator attribute, folding it if possible.
static TypedAttr getParamOperator(MLIRContext *ctx, POC opcode,
                                  ArrayRef<TypedAttr> operandsIn,
                                  Type resultType) {
  SmallVector<TypedAttr, 4> operands(operandsIn.begin(), operandsIn.end());

  // Verify and canonicalize parameter expressions.
  Attribute result;
  switch (opcode) {
  case POC::Add:
    result = simplifyAdd(operands);
    break;
  case POC::Mul:
    [[fallthrough]];
  case POC::MulNoWrap:
    result = simplifyGenericMul(operands, opcode);
    break;
  case POC::And:
    result = simplifyAnd(operands);
    break;
  case POC::Or:
    result = simplifyOr(operands);
    break;
  case POC::Xor:
    result = simplifyXor(operands);
    break;
  case POC::Max:
    result = simplifyMax(operands);
    break;
  case POC::Min:
    result = simplifyMin(operands);
    break;
  case POC::Shl:
    result = simplifyShl(operands);
    break;
  case POC::Shr:
    result = simplifyShr(operands);
    break;
  case POC::Div:
    result = simplifyDiv(operands);
    break;
  case POC::DivS:
    result = simplifyDiv(operands);
    break;
  case POC::DivU:
    result = simplifyDiv(operands);
    break;
  case POC::CeilDivS:
    result = simplifyCeilDiv(operands);
    break;
  case POC::CeilDivU:
    result = simplifyCeilDiv(operands);
    break;
  case POC::FloorDivS:
    result = simplifyFloorDiv(operands);
    break;
  case POC::RemS:
    result = simplifyMod(operands);
    break;
  case POC::RemU:
    result = simplifyMod(operands);
    break;
  case POC::Mod:
    result = simplifyMod(operands);
    break;
  case POC::EQ:
    result = simplifyEQ(operands);
    resultType = IntegerType::get(ctx, 1);
    break;
  case POC::LT:
  case POC::LE:
    result = simplifyRelationalCompare(opcode, operands);
    resultType = IntegerType::get(ctx, 1);
    break;
  case POC::CurrentTarget:
    resultType = TargetType::get(ctx);
    break;
  case POC::TargetHasFeature:
    result = simplifyHasFeature(operands);
    resultType = IntegerType::get(ctx, 1);
    break;
  case POC::TargetGetField:
    result = simplifyTargetGetField(operands, resultType);
    break;
  case POC::CrossCompilation:
    resultType = IntegerType::get(ctx, 1);
    break;
  case POC::AcceleratorArch:
    resultType = StringType::get(ctx);
    break;
  case POC::DataToStr:
    simplifyDataToStr(operands);
    resultType = StringType::get(ctx);
    break;
  case POC::StringAddress: // Can't simplify.
    break;
  case POC::StrConcat:
    result = simplifyStrConcat(operands[0], operands[1]);
    resultType = StringType::get(ctx);
    break;
  case POC::FunctionGetArgTypes:
    result = simplifyFunctionGetArgTypes(ctx, operands[0], resultType);
    break;
  case POC::In:
    result = simplifyIn(operands);
    resultType = IntegerType::get(ctx, 1);
    break;
  case POC::GetSizeOf:
    result = simplifyGetSizeOf(operands, resultType);
    break;
  case POC::GetAlignOf:
    result = simplifyGetAlignOf(operands, resultType);
    break;
  case POC::Apply:
    result = simplifyApply(operands, resultType);
    break;
  case POC::Rebind:
    result = simplifyRebind(operands, resultType);
    break;
  case POC::Cond:
    result = simplifyCond(operands);
    break;
  case POC::ApplyResultSlot:
  case POC::GetEnv:
  case POC::AttrToStr:
    result = {};
    break;
  case POC::PtrBitcast:
    result = simplifyPtrBitcast(operands, resultType);
    break;
  case POC::LoadFromMem:
    result = simplifyLoadFromMem(operands, resultType);
    break;
  case POC::VariadicPtrMap:
    assert(operands.size() == 2 && "variadic_ptr_map always has 2 operands");
    result = simplifyVariadicPtrMap(operands[0], operands[1], resultType);
    break;
  case POC::VariadicPtrRemoveMap:
    assert(operands.size() == 1 && "variadic_ptrremove_map has 1 operand");
    result = simplifyVariadicPtrRemoveMap(operands[0], resultType);
    break;
  }

  // If we folded to an operand, return it.
  if (result)
    return cast<TypedAttr>(result);

  return ParamOperatorAttr::Base::get(ctx, opcode, operands, resultType);
}

TypedAttr ParamOperatorAttr::get(MLIRContext *context, POC opcode,
                                 ArrayRef<TypedAttr> operandsIn, Type type) {
  return getParamOperator(context, opcode, operandsIn, type);
}

TypedAttr
ParamOperatorAttr::getChecked(function_ref<InFlightDiagnostic()> emitError,
                              MLIRContext *context, POC opcode,
                              ArrayRef<TypedAttr> operandsIn, Type type) {
  if (failed(verify(emitError, opcode, operandsIn, type)))
    return {};
  return get(context, opcode, operandsIn, type);
}

ErrorOr<Type> inferParamOperatorResultType(POC opcode,
                                           ArrayRef<TypedAttr> operandsIn) {
  // All operands must have the same type.  The result type is usually the
  // same as the operands, but is i1 for comparisons (overridden below).
  Type resultType;
  if (opcode == POC::Cond)
    resultType = operandsIn[1].getType();
  else if (opcode != POC::GetSizeOf && opcode != POC::GetAlignOf)
    resultType = operandsIn.front().getType();
  // Raise error if operands do not have the same type for certain POCs.
  if (!llvm::is_contained({POC::Apply, POC::ApplyResultSlot, POC::DataToStr,
                           POC::TargetHasFeature, POC::TargetGetField,
                           POC::AcceleratorArch, POC::GetSizeOf,
                           POC::GetAlignOf, POC::GetEnv, POC::VariadicPtrMap,
                           POC::VariadicPtrRemoveMap, POC::StringAddress},
                          opcode) &&
      !llvm::all_of(operandsIn.drop_front(),
                    [&](auto op) { return op.getType() == resultType; })) {
    return Error(llvm::formatv(
        "POC opcode {}: Operands must have same type, got [{}]", opcode,
        llvm::join(llvm::map_range(operandsIn,
                                   [](TypedAttr attr) {
                                     return llvm::formatv("{}", attr);
                                   }),
                   ", ")));
  }
  return resultType;
}

TypedAttr
ParamOperatorAttr::getChecked(function_ref<InFlightDiagnostic()> emitError,
                              POC opcode, ArrayRef<TypedAttr> operandsIn) {
  auto resultTypeOr = inferParamOperatorResultType(opcode, operandsIn);
  if (failed(resultTypeOr)) {
    emitError() << resultTypeOr.takeError();
    return {};
  }
  if (failed(verify(emitError, opcode, operandsIn, *resultTypeOr)))
    return {};
  return get(opcode, operandsIn);
}

TypedAttr ParamOperatorAttr::get(POC opcode, ArrayRef<TypedAttr> operandsIn) {
  assert(!operandsIn.empty() && "Cannot have expr with no operands");
  auto resultTypeOr = inferParamOperatorResultType(opcode, operandsIn);
  assert(succeeded(resultTypeOr) && "failed to infer result type");
  return getParamOperator(operandsIn.front().getContext(), opcode, operandsIn,
                          *resultTypeOr);
}

/// Return (not x) which is the same as (xor x, true).  The `operand` value
/// must have type `i1`.
TypedAttr ParamOperatorAttr::getNot(TypedAttr operand) {
  TypedAttr one = BoolAttr::get(operand.getContext(), true);
  return ParamOperatorAttr::get(POC::Xor, {operand, one});
}

/// Return (neg x) which is the same as (mul x, -1).  The `operand` value
/// must have `index` type.
TypedAttr ParamOperatorAttr::getNeg(TypedAttr operand) {
  IntegerAttr minusOne = IntegerAttr::get(operand.getType(), -1);
  return ParamOperatorAttr::get(POC::Mul, operand, minusOne);
}

/// Return (x-y) which is the same as (add x, (neg y)).  The `operand` value
/// must have `index` type.
TypedAttr ParamOperatorAttr::getSub(TypedAttr lhs, TypedAttr rhs) {
  return get(POC::Add, lhs, getNeg(rhs));
}

/// If the specified attribute is a rebind, return its operand, otherwise
/// return the rebind itself.
TypedAttr ParamOperatorAttr::stripRebind(TypedAttr src) {
  if (auto rebind = dyn_castPE(POC::Rebind, src))
    return rebind.getOperand(0);
  return src;
}

/// Parameter operators are the basis of parameter expressions and are never
/// simple constants.
bool ParamOperatorAttr::isConstant() const { return false; }

/// Sort operators by opcode, then number of operands, then recursively sort by
/// operand values.
bool ParamOperatorAttr::isLessThan(Attribute rhs) const {
  auto op = ::cast<ParamOperatorAttr>(rhs);

  // Sort by string value of the opcode.
  if (getOpcode() != op.getOpcode())
    return stringifyPOC(getOpcode()) < stringifyPOC(op.getOpcode());

  // If they are the same opcode, sort by arity. More complex expressions are to
  // the left.
  if (getNumOperands() != op.getNumOperands())
    return getNumOperands() > op.getNumOperands();

  // We know the two subexpressions are different (they'd otherwise be pointer
  // equivalent) so just go compare all of the elements.
  for (auto [lhs, rhs] : llvm::zip(getOperands(), op.getOperands())) {
    if (ParameterAttr::compare(lhs, rhs))
      return true;
    if (ParameterAttr::compare(rhs, lhs))
      return false;
  }

  return false;
}

ErrorOrSuccess ParamOperatorAttr::validateForElaborator() const {
  // If this operator didn't fold, then it's a problem.
  // TODO: we could diagnose WHY it isn't folding more nicely now.
  return Error("could not simplify operator " + getParamAsString(*this));
}

//===----------------------------------------------------------------------===//
// MLIROpAttr
//===----------------------------------------------------------------------===//

LogicalResult MLIROpAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 StringAttr name, DictionaryAttr attrs,
                                 FuncTypeGeneratorType type) {
  if (type.getBody().getNumResults() != 1)
    return emitError()
           << "operation parameter expression must return one result";
  if (!type.isFullyBound())
    return emitError()
           << "operation parameter expression must be a concrete signature";
  return success();
}

/// An MLIR operation attribute is always a constant.
bool MLIROpAttr::isConstant() const { return true; }

TypedAttr KGEN::emitMLIROperationCall(
    StringRef opName,
    ArrayRef<std::pair<StringAttr (*)(mlir::OperationName), Attribute>> attrs,
    ArrayRef<TypedAttr> operands, Type resultType) {
  MLIRContext *ctx = resultType.getContext();
  mlir::OperationName name(opName, ctx);
  NamedAttrList attrList;
  for (auto [attrName, value] : attrs)
    attrList.append(attrName(name), value);
  SmallVector<Type> operandTypes;
  for (TypedAttr operand : operands)
    operandTypes.push_back(operand.getType());
  SmallVector<TypedAttr> applyOperands;
  applyOperands.push_back(
      MLIROpAttr::get(name.getIdentifier(), attrList.getDictionary(ctx),
                      FuncTypeGeneratorType::get(
                          /*inputParamTypes=*/{},
                          FunctionType::get(ctx, operandTypes, resultType))));
  llvm::append_range(applyOperands, operands);
  return ParamOperatorAttr::get(POC::Apply, applyOperands);
}

//===----------------------------------------------------------------------===//
// ClosureRefAttr
//===----------------------------------------------------------------------===//

static ParseResult parseClosureSymbolValue(
    AsmParser &p, SymbolRefAttr &symbol, StringAttr &nestedFunctionName,
    ClosureMethodAttr &method, SmallVector<TypedAttr> &paramValues) {
  if (p.parseAttribute(symbol) || p.parseComma() ||
      p.parseAttribute(nestedFunctionName) || p.parseComma() ||
      p.parseAttribute(method))
    return failure();
  if (succeeded(p.parseOptionalComma())) {
    if (parseParameterValues(p, paramValues))
      return failure();
  }
  return success();
}

static void printClosureSymbolValue(AsmPrinter &p, SymbolRefAttr symbol,
                                    StringAttr nestedFunctionName,
                                    ClosureMethodAttr method,
                                    ArrayRef<TypedAttr> paramValues) {
  p << symbol << ", " << nestedFunctionName << ", " << method;
  if (!paramValues.empty()) {
    p << ", ";
    printParameterValues(p, paramValues);
  }
}

//===----------------------------------------------------------------------===//
// ClosureAttr
//===----------------------------------------------------------------------===//

static ParseResult parseClosureAttr(AsmParser &p, Type &type) {
  SymbolRefAttr symbol;
  StringAttr name;
  if (p.parseAttribute(symbol) || p.parseAttribute(name))
    return failure();
  type = ParamClosureType::get(p.getContext(), symbol, name);
  return success();
}

static void printClosureAttr(AsmPrinter &p, Type type) {
  ParamClosureType closureType = cast<ParamClosureType>(type);
  p << closureType.getParentSymbol() << " " << closureType.getName();
}

//===----------------------------------------------------------------------===//
// MemSymbolTripleAttr
//===----------------------------------------------------------------------===//

static ParseResult parseMemSymbolTripleAttr(AsmParser &p, TypedAttr &copy,
                                            TypedAttr &move, TypedAttr &del,
                                            UnitAttr &isMove) {
  MemSymbolTripleParts parts;
  if (parseMemSymbolParts(p, parts))
    return failure();
  if (parts.isMove)
    isMove = UnitAttr::get(p.getContext());

  Type type;
  if (parts.requiresCaptureType() && (p.parseColon() || p.parseType(type)))
    return failure();

  auto materialize = [&](const MemSymbolTripleEntry &entry,
                         ArrayRef<ArgConvention> argConventions,
                         bool isConstructor = true) -> TypedAttr {
    if (entry.isTypedAttr())
      return entry.attr;
    if (!entry.isSymbolShorthand())
      return {};
    return makeSymbol(type, entry.symbolParts.callee,
                      entry.symbolParts.paramValues, argConventions,
                      isConstructor);
  };

  copy = materialize(parts.copy,
                     {ArgConvention::ReadMem, ArgConvention::ByRefResult});
  move = materialize(parts.move,
                     {ArgConvention::OwnedMem, ArgConvention::ByRefResult});
  del = materialize(parts.del, {ArgConvention::OwnedMem},
                    /*isConstructor=*/false);
  return success();
}

static void printMemSymbolTripleAttr(AsmPrinter &p, TypedAttr copy,
                                     TypedAttr move, TypedAttr del,
                                     UnitAttr isMove) {
  printMemSymbolTripleAttrWithoutType(p, copy, move, del);
  if (isMove)
    p << " " << MemSymbolTripleAttr::kIsMoveKeyword;
  if (auto symDel = dyn_cast<SymbolConstantAttr>(del))
    p << " : " << symDel.getType().getBody().getArguments().front();
}

//===----------------------------------------------------------------------===//
// DeferredAttr
//===----------------------------------------------------------------------===//

Type DeferredAttr::getType() const {
  return KGEN::DeferredType::get(getContext());
}

LogicalResult
DeferredAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                     Attribute attr) {
  if (::isa<TypedAttr>(attr))
    return emitError()
           << "`#kgen.deferred` can only be used for non-typed attributes";
  return success();
}

//===----------------------------------------------------------------------===//
// AttrCtorDeferredAttr
//===----------------------------------------------------------------------===//

Type AttrCtorDeferredAttr::getType() const {
  return KGEN::DeferredType::get(getContext());
}

LogicalResult AttrCtorDeferredAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    ArrayRef<TypedAttr> strings) {

  for (TypedAttr attr : strings) {
    if (!::isa<StringAttr, ToStringDeferredAttr>(attr))
      return emitError()
             << "`#kgen.attr_ctor_deferred` can only be used for 'StringAttr', "
                "or '#kgen.to_string_deferred', but got '"
             << attr << '\'';
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ToStringDeferredAttr
//===----------------------------------------------------------------------===//

Type ToStringDeferredAttr::getType() const {
  return StringAttr::get(getContext()).getType();
}

//===----------------------------------------------------------------------===//
// ConstraintAttr
//===----------------------------------------------------------------------===//

LogicalResult
ConstraintAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                       TypedAttr proposition, LocationAttr loc) {
  // Verify that the proposition has i1 type
  if (!proposition.getType().isSignlessInteger(1)) {
    return emitError() << "constraint proposition must have i1 type, but got '"
                       << proposition.getType() << "'";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// CastFromBuiltinAttr / CastToBuiltinAttr
//===----------------------------------------------------------------------===//

TypedAttr CastFromBuiltinAttr::get(MLIRContext *ctx, TypedAttr arg,
                                   SIMDType out_type) {
  if (auto fold = KGEN::foldCastFromBuiltin(arg, out_type))
    if (auto ret = dyn_cast<TypedAttr>(cast<Attribute>(fold)))
      return ret;

  return Base::get(ctx, arg, out_type);
}

TypedAttr
CastFromBuiltinAttr::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                MLIRContext *context, TypedAttr value,
                                SIMDType out_type) {
  if (failed(verify(emitError, value, out_type)))
    return {};
  return CastFromBuiltinAttr::get(context, value, out_type);
}

bool CastFromBuiltinAttr::isConstant() const { return false; }

LogicalResult
CastFromBuiltinAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                            TypedAttr value, SIMDType out_type) {
  return KGEN::verifyConversionCast(
      [emitError](StringRef msg) { return emitError() << msg; }, out_type,
      value.getType(), /*fromSimd=*/false);
}

TypedAttr CastToBuiltinAttr::get(MLIRContext *ctx, TypedAttr arg,
                                 Type out_type) {
  // If this is a known constant SIMD value, fold this directly to its
  // equivalent MLIR-typed value.
  if (auto fold = KGEN::foldCastToBuiltin(arg, out_type))
    if (auto ret = dyn_cast<TypedAttr>(cast<Attribute>(fold)))
      return ret;

  return Base::get(ctx, arg, out_type);
}

TypedAttr
CastToBuiltinAttr::getChecked(function_ref<InFlightDiagnostic()> emitError,
                              MLIRContext *context, TypedAttr value,
                              Type out_type) {
  if (failed(verify(emitError, value, out_type)))
    return {};
  return CastToBuiltinAttr::get(context, value, out_type);
}

bool CastToBuiltinAttr::isConstant() const { return false; }

LogicalResult
CastToBuiltinAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                          TypedAttr value, Type out_type) {
  if (!isa<SIMDType>(value.getType()))
    return emitError() << "Invalid operand type: must be SIMDType";
  return KGEN::verifyConversionCast(
      [emitError](StringRef msg) { return emitError() << msg; },
      cast<SIMDType>(value.getType()), out_type, /*fromSimd=*/true);
}

//===----------------------------------------------------------------------===//
// LLVMBitcodeLibAttr
//===----------------------------------------------------------------------===//

LogicalResult LLVMBitcodeLibAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, BoolAttr used,
    Attribute library) {
  // Verify that the library attribute is either StringAttr (for file paths)
  // or DenseResourceElementsAttr (for package bitcode).
  if (!isa<StringAttr>(library) && !isa<DenseResourceElementsAttr>(library)) {
    return emitError() << "library attribute must be either StringAttr "
                          "(for file paths) or DenseResourceElementsAttr "
                          "(for package bitcode)";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// LLVMBitcodeLibArrayAttr
//===----------------------------------------------------------------------===//

void LLVMBitcodeLibArrayAttr::externalize(
    llvm::SmallVector<std::pair<bool, mlir::Attribute>> &result) const {
  result.clear();
  for (auto libAttr : getValue())
    result.emplace_back(libAttr.getUsed().getValue(), libAttr.getLibrary());
}

//===----------------------------------------------------------------------===//
// SugarAttr
//===----------------------------------------------------------------------===//

mlir::OpAsmAliasResult SugarAttr::getAlias(raw_ostream &os) const {
  // If this is a ParamType wrapping a simple SugarAttr then use an alias based
  // on the alias name to significantly compact the type reference.
  if (getKind() == SugarKind::Alias)
    if (auto dre = dyn_cast<ParamDeclRefAttr>(getSugared())) {
      auto name = dre.getName().strref();
      // Remove mangling.
      name = name.take_front(name.find('`'));
      if (!name.empty()) {
        os << "alias_" << name;
        return mlir::OpAsmAliasResult::OverridableAlias;
      }
    }
  return mlir::OpAsmAliasResult::NoAlias;
}

/// Return the strongest form of sugar that the specified result should be
/// elided for.
///
/// This is designed to align with ASTPrinter, but needs to be here as part of
/// Sugar building, because we want this to simplify sugar when parameter values
/// are bound to concrete things.  For example we want to maintain "x+y" as
/// sugar, but fold it to "4" when 3 and 1 are substituted in.
static std::optional<SugarKind> canElideSugarFor(TypedAttr attr) {
  // If we folded this to a reference to another declaration, just use it.
  if (isa<ParamDeclRefAttr>(attr))
    return SugarKind::AlwaysInlineBuiltin;

  // Never sugar low-level MLIR integer literals.  They are internal
  // implementation details of low-level types, not something we want to expose
  // to Mojo programmers.
  if (isa<IntegerAttr>(attr))
    return SugarKind::AlwaysInlineBuiltin;

  // Anything that resolves into an UnknownAttr is a struct with no state.  All
  // the computation is happening in the type domain, as in the literal types.
  if (isa<UnknownAttr>(attr))
    return SugarKind::AlwaysInlineBuiltin;

  // Anything that resolves into an UnknownAttr is a struct with no state.  All
  // the computation is happening in the type domain, as in the literal types.
  if (isa<UnknownAttr>(attr))
    return SugarKind::AlwaysInlineBuiltin;

  // Otherwise, see if the LIT type knows how to elide itself.  LIT::StructType
  // knows how to print literals for Int, IntegerLiteral, etc.
  if (auto sugarItf = sugarDynCast<SugaredTypeInterface>(attr.getType()))
    return sugarItf.canElideSugarFor(attr);

  return {};
}

static ParseResult parseSugarAttr(AsmParser &p, SugarKind &kind,
                                  StringAttr &memberName, TypedAttr &sugared,
                                  TypedAttr &original, TypedAttr &canonical) {
  FailureOr<SugarKind> kindResult = mlir::FieldParser<SugarKind>::parse(p);
  if (failed(kindResult))
    return failure();
  kind = *kindResult;

  if (kind == SugarKind::MemberAlias) {
    if (p.parseComma() || p.parseAttribute(memberName))
      return failure();
  }

  Type type;
  if (p.parseComma() || p.parseType(type) || p.parseComma() ||
      parseParamValue(p, sugared, type) || p.parseComma() ||
      parseParamValue(p, original, type))
    return failure();

  canonical = getCanonicalAttr(original);
  return success();
}

static void printSugarAttr(AsmPrinter &p, SugarKind kind, StringAttr memberName,
                           TypedAttr sugared, TypedAttr original,
                           TypedAttr canonical) {
  p.printStrippedAttrOrType(kind);
  if (kind == SugarKind::MemberAlias) {
    p << ", " << memberName;
  }

  p << ", ";
  p.printType(sugared.getType());
  p << ", ";
  printParamValue(p, sugared);
  p << ", ";
  printParamValue(p, original);
  // The canonical value is rebuilt as needed.
}

TypedAttr SugarAttr::get(MLIRContext *context, SugarKind kind,
                         StringAttr memberName, TypedAttr sugared,
                         TypedAttr expanded, TypedAttr canonical) {
  // The two operand types must match otherwise this won't round-trip
  // correctly.
  assert(sugared.getType() == expanded.getType() &&
         "SugarAttr sugared and original types must match");
  assert((kind == SugarKind::MemberAlias && memberName) ||
         (kind != SugarKind::MemberAlias && !memberName) &&
             "memberName should be specified for MemberAlias only");

  // If we shouldn't maintain type sugar for this, then just return the
  // expanded. We strip sugar for always_inline builtin calls that simplify down
  // to something simple like "42". We don't want to maintain the call sugar
  // for things like 4+5 because it just gets in the way.
  //
  // This is also important for reducing the size of the IR, making it more
  // readable and the compiler faster for primitive things like origins.
  if (auto shouldElide = canElideSugarFor(expanded))
    if ((int)*shouldElide >= (int)kind)
      return expanded;

  // This method gets called by client doing general structural replacements,
  // e.g. a parameter with an arbitrary attribute.  This can turn canonical
  // forms to non-canonical and visa-versa, so always recompute the canonical
  // pointer.
  canonical = getCanonicalAttr(expanded);

  // We will /never/ use the type sugar in the expanded form of opaque
  // sugar kinds, so we can strip it all away to simplify things.
  if (kind <= SugarKind::Preserved)
    expanded = ParamOperatorAttr::getRebind(canonical, sugared.getType());

  return Base::get(sugared.getContext(), kind, memberName, sugared, expanded,
                   canonical);
}

Type SugarAttr::getMemberAliasType() const {
  return cast<TypeParamAttr>(getSugared()).getMlirType();
}

TypedAttr SugarAttr::getMemberAlias(Type type, StringAttr memberName,
                                    TypedAttr original) {
  return get(type.getContext(), SugarKind::MemberAlias, memberName,
             TypeParamAttr::get(type, original.getType()), original);
}

Type SugarAttr::getType() const { return getSugared().getType(); }

/// Remove any top-level sugar nodes from this type, but don't fully
/// canonicalize it.
TypedAttr SugarAttr::strip(TypedAttr value, bool keepOpaque) {
  if (!value)
    return {};

  while (auto sugar = dyn_cast<SugarAttr>(value)) {
    // Keep opaque sugar kinds if requested. These are implementation details
    // that should not produce "aka" notes in diagnostics.
    if (sugar.getKind() <= SugarKind::Preserved && keepOpaque)
      break;
    value = sugar.getExpanded();
  }
  return value;
}

Type SugarAttr::strip(Type type, bool keepOpaque) {
  if (!type)
    return {};
  // Sugar for a type will be wrapped in a ParamType converting the attr into
  // the type domain.
  if (auto paramRef = dyn_cast<ParamType>(type))
    if (isa<SugarAttr>(paramRef.getParam())) {
      // Strip the attr inside the ParamType if it is sugar.
      type = ParamType::get(strip(paramRef.getParam(), keepOpaque));
      // ParamType::get has some canonicalizations that can expose nested sugar.
      // Make sure to remove them as well.
      return strip(type);
    }
  return type;
}

Attribute SugarAttr::strip(Attribute value, bool keepOpaque) {
  if (auto typedAttr = dyn_cast<TypedAttr>(value))
    return SugarAttr::strip(typedAttr, keepOpaque);
  return value;
}

/// Return true if the specified value has top level SugarAttr.  Nested
/// sugar isn't processed here.
bool SugarAttr::hasTopLevelSugar(Type type) {
  if (auto paramRef = dyn_cast_if_present<ParamType>(type))
    return isa<SugarAttr>(paramRef.getParam());
  return false;
}

static Attribute getLocalCanonical(Attribute attr) {
  // SugarAttr maintains its canonical form directly so we don't need to walk.
  if (auto sugar = dyn_cast<SugarAttr>(attr))
    return sugar.getCanonical();
  return {};
}

static Type getLocalCanonical(Type type) {
  // FIXME: Why is this getting called with null types?
  if (!type)
    return {};

  // Otherwise, see if the LIT type knows how to elide itself.
  if (auto sugarItf = dyn_cast<KGEN::SugaredTypeInterface>(type))
    return sugarItf.getCachedCanonicalType(type);

  return {};
}

namespace {
class Canonicalizer : public ParameterReplacer<Canonicalizer> {
  template <typename T>
  std::conditional_t<std::is_base_of_v<Type, T>, Type, Attribute>
  doReplace(T value, size_t depth) {
    // If this type has a canonical cache pointer, use it.
    if (auto can = getLocalCanonical(value))
      return can;

    // Otherwise, recursively walk and rebuild the attribute with
    // canonicalized subelements.
    SmallVector<Attribute, 16> newAttrs;
    SmallVector<Type, 16> newTypes;
    bool changed = false;
    auto walkFn = [&](auto value, SmallVectorImpl<decltype(value)> &values) {
      auto newValue = this->replaceImpl(value, depth);
      changed |= newValue != value;
      values.push_back(newValue);
    };
    value.walkImmediateSubElements(
        [&](Attribute attr) { walkFn(attr, newAttrs); },
        [&](Type type) { walkFn(type, newTypes); });
    if (!changed)
      return value;
    return value.replaceImmediateSubElements(newAttrs, newTypes);
  }

  friend class ParameterReplacer<Canonicalizer>;
};
}; // end anonymous namespace

/// Given an attribute or type, return the "canonical" version of the attribute
/// with all type sugar removed.
TypedAttr KGEN::getCanonicalAttr(TypedAttr src) {
  // If this is locally and obviously canonical, then just return it.
  if (auto local = getLocalCanonical(src))
    return cast<TypedAttr>(local);
  return Canonicalizer().replace(src);
}

Attribute KGEN::getCanonicalAttr(Attribute src) {
  // If this is locally and obviously canonical, then just return it.
  if (auto local = getLocalCanonical(src))
    return local;
  return Canonicalizer().replace(src);
}

Type KGEN::getCanonicalType(Type src) {
  // If this is locally and obviously canonical, then just return it.
  if (auto local = getLocalCanonical(src))
    return local;
  return Canonicalizer().replace(src);
}

/// Return true if the specified types are canonically equal.
bool KGEN::isEqualCanon(Type t1, Type t2) {
  if (t1 == t2)
    return true;
  return getCanonicalType(t1) == getCanonicalType(t2);
}

bool KGEN::isEqualCanon(TypedAttr ta1, TypedAttr ta2) {
  if (ta1 == ta2)
    return true;
  return getCanonicalAttr(ta1) == getCanonicalAttr(ta2);
}

//===----------------------------------------------------------------------===//
// DTypeValue
//===----------------------------------------------------------------------===//

DTypeValue::DTypeValue(APInt data, KGENDType dtype)
    : data(std::move(data)), dtype(dtype) {
  assert(dtype.isAddress() || dtype.isIndex() || dtype.isUIndex() ||
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
  assert(dtype.isIndex() || dtype.isUIndex() || dtype.isAddress());
  return dtype.isIndex() ? data.getSExtValue() : data.getZExtValue();
}

namespace M::KGEN {
/// Provide the ability to hash values for attribute uniquing.
inline llvm::hash_code hash_value(const DTypeValue &value) {
  return hash_combine(value.getData(), value.getDType().getValue());
}
} // namespace M::KGEN

//===----------------------------------------------------------------------===//
// DTypeValue parser/printer helpers
//===----------------------------------------------------------------------===//

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
  assert(dtype.isIndex() || dtype.isUIndex() || dtype.isAddress());
  int64_t indexVal;
  if constexpr (optionalParse) {
    OptionalParseResult result;
    if (dtype.isIndex())
      result = p.parseOptionalInteger(indexVal);
    else
      result = p.parseOptionalInteger(reinterpret_cast<uint64_t &>(indexVal));
    if (!result.has_value())
      return std::nullopt;
    if (failed(*result))
      return failure();
  } else {
    if (dtype.isIndex()) {
      if (p.parseInteger(indexVal))
        return failure();
    } else {
      if (p.parseInteger(reinterpret_cast<uint64_t &>(indexVal)))
        return failure();
    }
  }
  return DTypeValue(indexVal, dtype);
}

//===----------------------------------------------------------------------===//
// SIMDAttrStorage / ODS Boilerplate
//===----------------------------------------------------------------------===//

namespace M::KGEN::detail {
/// Custom storage class that allocates and owns the `DTypeValue` instances in
/// an `OwningArrayRef`, because they are not POD.
struct SIMDAttrStorage : public mlir::AttributeStorage {
  using KeyTy = std::tuple<ArrayRef<DTypeValue>, SIMDType>;
  SIMDAttrStorage(SmallVector<DTypeValue> values, SIMDType type)
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
    return new (allocator.allocate<SIMDAttrStorage>()) SIMDAttrStorage(
        SmallVector<DTypeValue>(std::get<0>(key)), std::get<1>(key));
  }

  SmallVector<DTypeValue> values;
  SIMDType type;
};
} // namespace M::KGEN::detail

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
  DTypeValue scalarVal(std::move(apsVal), dtype);
  return SIMDAttr::get(scalarVal, type);
}

SIMDAttr SIMDAttr::getChecked(function_ref<InFlightDiagnostic()> emitError,
                              uint64_t intVal, SIMDType type) {
  DType dtype = *type.getResolvedDType();
  APInt apVal(dtype.getIntegerWidthInBits(), intVal);
  APSInt apsVal(std::move(apVal), /*isUnsigned=*/dtype.isUInt());
  DTypeValue scalarVal(std::move(apsVal), dtype);
  return SIMDAttr::getChecked(emitError, scalarVal, type);
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
    assert(dtype.isIndex() || dtype.isUIndex() || dtype.isAddress());
    if (dtype.isIndex())
      p << value.getIndexVal();
    else
      p << uint64_t(value.getIndexVal());
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
        dtype->isIndex() || dtype->isUIndex() || dtype->isAddress())
      return success();
    return p.emitError(
        p.getCurrentLocation(),
        "only integer, float, bool, and index dtype constants can be parsed");
  };

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

  if constexpr (optionalParse) {
    if (p.parseOptionalLess())
      return std::nullopt;
  } else {
    if (p.parseLess())
      return failure();
  }

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
  if (!values.empty() && llvm::all_equal(values)) {
    if constexpr (inAttr)
      p << ' ';
    printDTypeValue(p, values.front(), values.front().getDType());
    return;
  }

  p << '<';
  printDTypeValues(p, values, type);
  p << '>';
}

void KGEN::printDTypeValue(raw_ostream &os, const DTypeValue &value,
                           KGENDType dtype) {
  if (dtype.isInt())
    os << value.getIntVal();
  else if (dtype.isFloat()) {
    SmallString<64> s;
    value.getFloatVal().toString(s);
    os << s;
  } else if (dtype.isBool())
    os << (value.getBoolVal() ? "True" : "False");
  else if (dtype.isIndex())
    os << value.getIndexVal();
  else {
    assert(dtype.isUIndex() || dtype.isAddress());
    os << static_cast<uint64_t>(value.getIndexVal());
  }
}

void KGEN::printDTypeValues(raw_ostream &os, ArrayRef<DTypeValue> values,
                            KGENDType dtype) {
  if (values.size() == 1) {
    printDTypeValue(os, values[0], dtype);
  } else {
    os << "[";
    llvm::interleaveComma(values, os, [&](const DTypeValue &v) {
      printDTypeValue(os, v, dtype);
    });
    os << "]";
  }
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
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "KGEN/KGENDialect/KGENAttrs.cpp.inc"
