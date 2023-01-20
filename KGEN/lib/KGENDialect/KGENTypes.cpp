//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "Support/MDialect/MTypeInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// KGENDialect
//===----------------------------------------------------------------------===//

void KGENDialect::registerTypes() {
  // Register types.
  addTypes<
#define GET_TYPEDEF_LIST
#include "KGEN/KGENDialect/KGENTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ParamRefType
//===----------------------------------------------------------------------===//

Type ParamRefType::get(TypedAttr param) {
  // If the parameter is already resolved to a constant, fold this to the
  // indicated type.
  if (auto constant = param.dyn_cast<TypeConstantAttr>())
    return constant.getValue();

  // Otherwise, form the ParamRefType like normal.
  return Base::get(param.getContext(), param);
}

Type ParamRefType::replaceImmediateSubElements(ArrayRef<Attribute> replAttrs,
                                               ArrayRef<Type> replTypes) const {
  assert(replAttrs.size() == 1 && replTypes.empty());
  return ParamRefType::get(replAttrs[0]);
}

//===----------------------------------------------------------------------===//
// MLIRTypeType
//===----------------------------------------------------------------------===//

OptionalParseResult MLIRTypeType::parseValue(AsmParser &p,
                                             TypedAttr &value) const {
  Type type;
  OptionalParseResult result = p.parseOptionalType(type);
  if (!result.has_value())
    return {};
  if (failed(*result))
    return failure();
  value = TypeConstantAttr::get(type);
  return mlir::success();
}

LogicalResult MLIRTypeType::printValue(raw_ostream &os, TypedAttr value) const {
  auto type = ::dyn_cast<TypeConstantAttr>(value);
  if (!type)
    return failure();
  os << type.getValue();
  return success();
}

//===----------------------------------------------------------------------===//
// ListType
//===----------------------------------------------------------------------===//

ListType ListType::get(TypedAttr elementType, TypedAttr length) {
  return get(elementType.getContext(), elementType, length);
}

ListType ListType::get(Type elementType, int64_t length) {
  return get(TypeConstantAttr::get(elementType),
             Builder(elementType.getContext()).getIndexAttr(length));
}

ListType ListType::get(Type elementType, TypedAttr length) {
  return get(TypeConstantAttr::get(elementType), length);
}

LogicalResult ListType::verify(function_ref<InFlightDiagnostic()> emitError,
                               TypedAttr elementType, TypedAttr length) {
  if (!llvm::isa<MLIRTypeType>(elementType.getType()))
    return emitError()
           << "expected element type expression to be a '!kgen.mlirtype'";
  if (!llvm::isa<IndexType>(length.getType()))
    return emitError() << "expected length expression to be an 'index'";
  return success();
}

std::optional<int64_t> ListType::getResolvedLength() const {
  if (auto length = llvm::dyn_cast<IntegerAttr>(getLength()))
    return length.getInt();
  return {};
}

Type ListType::getResolvedElementType() const {
  if (auto type = llvm::dyn_cast<ConcreteTypeConstantAttr>(getElementType()))
    return type.getValue();
  return {};
}

//===----------------------------------------------------------------------===//
// SignatureType
//===----------------------------------------------------------------------===//

OptionalParseResult SignatureType::parseValue(AsmParser &p,
                                              TypedAttr &value) const {
  Attribute attr;
  OptionalParseResult result = p.parseOptionalAttribute(attr, *this);
  if (!result.has_value())
    return std::nullopt;

  // Parse a symbol reference as a signature type attribute.
  if (auto symbol = attr.dyn_cast<SymbolRefAttr>()) {
    // Parse any trailing parameter bindings.
    FailureOr<ParamBindArrayAttr> paramValues;
    if (parseOptionalParamBindSpec(p, paramValues))
      return failure();
    value = SymbolConstantAttr::get(symbol, *paramValues, *this);
  } else {
    value = attr.cast<TypedAttr>();
  }
  return mlir::success();
}

LogicalResult SignatureType::printValue(raw_ostream &os,
                                        TypedAttr value) const {
  auto symbolCst = ::dyn_cast<SymbolConstantAttr>(value);
  if (!symbolCst)
    return failure();
  os << symbolCst.getSymbol();
  printOptionalParamBindSpec(symbolCst.getParamValues(), os);
  return success();
}

static void getSignatureDefaults(ParamDeclArrayAttr &inputParams,
                                 TypeArrayAttr &resultParamTypes,
                                 FunctionType values,
                                 ConventionsAttr &conventions) {
  MLIRContext *ctx = values.getContext();
  if (!inputParams)
    inputParams = ParamDeclArrayAttr::get(ctx, {});
  if (!resultParamTypes)
    resultParamTypes = TypeArrayAttr::get(ctx, {});
  if (!conventions) {
    // Default valueConventions to zero.
    conventions = ConventionsAttr::get(
        ctx, SmallVector<ValueInputConvention>(values.getNumInputs()),
        FnEffects::None);
  }
}

SignatureType SignatureType::get(ParamDeclArrayAttr inputParams,
                                 TypeArrayAttr resultParamTypes,
                                 FunctionType values,
                                 ConventionsAttr conventions) {
  getSignatureDefaults(inputParams, resultParamTypes, values, conventions);
  return get(values.getContext(), inputParams, resultParamTypes, values,
             conventions);
}

SignatureType
SignatureType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                          ParamDeclArrayAttr inputParams,
                          TypeArrayAttr resultParamTypes, FunctionType values,
                          ConventionsAttr conventions) {
  getSignatureDefaults(inputParams, resultParamTypes, values, conventions);
  return getChecked(emitError, values.getContext(), inputParams,
                    resultParamTypes, values, conventions);
}

SignatureType SignatureType::get(ParamBindArrayAttr inputParams,
                                 ParamDeclArrayAttr resultParams,
                                 FunctionType values,
                                 ConventionsAttr conventions) {
  SmallVector<ParamDeclAttr> inputParamDecls;
  SmallVector<Type> resultParamTypes;
  for (ParamBindAttr inputParam : inputParams)
    inputParamDecls.push_back(inputParam.getDecl());
  for (ParamDeclAttr resultParam : resultParams)
    resultParamTypes.push_back(resultParam.getType());
  return get(ParamDeclArrayAttr::get(values.getContext(), inputParamDecls),
             TypeArrayAttr::get(values.getContext(), resultParamTypes), values,
             conventions);
}

SignatureType SignatureType::get(FunctionType values) {
  return get(ParamDeclArrayAttr(), {}, values, {});
}

SignatureType SignatureType::get(MLIRContext *ctx, TypeRange inputs,
                                 TypeRange results) {
  return get(FunctionType::get(ctx, inputs, results));
}

SignatureType SignatureType::setFnEffect(FnEffects effect) {
  return SignatureType::get(
      getInputParams(), getResultParamTypes(), getValues(),
      ConventionsAttr::get(getContext(), getValueInputConventions(),
                           bitEnumSet(getFnEffects(), effect)));
}

/// Return a signature with the specified parameter bindings substituted
/// into it as happens in a call.  The types specified in the parameter
/// bindings affects the type signature of the value input and outputs, and
/// also can remap the signature in the parameter list itself.
///
/// If an error occurs making the substitution, report it with emitErrorFn
/// and return null.
SignatureType SignatureType::getSpecializedSignature(
    ArrayRef<ParamBindAttr> inputParamValues,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitErrorFn) {
  // We need to substitute and simplify expressions that occur in the argument
  // list and parameter types, e.g.:
  //     kgen.generator @callee1<type: dtype>(%x: !pop.scalar<type>)
  //     kgen.generator @callee2<size>(%x: !pop.simd<size, f32>)
  // ... call @callee1<type: dtype = f32>(%arg1) : (!pop.scalar<f32>) -> ()
  // ... call @callee2<size=4>(%arg2) : (!pop.simd<4, f32>) -> ()
  //
  // This can also occur in parameter types, e.g. for region types (dt vs f32):
  //     kgen.generator @g<dt: dtype, region: () -> !pop.scalar<dt>>(...
  //     call @g<dt: dtype = f32, region: () -> !pop.scalar<f32>(...
  if (inputParamValues.size() != getInputParams().size()) {
    emitErrorFn() << "caller has " << inputParamValues.size()
                  << " input parameters but callee expects "
                  << getInputParams().size() << "; signature is " << *this;
    return SignatureType();
  }

  // If the signature isn't parameterized, then there are no substitutions to
  // perform.
  if (inputParamValues.empty())
    return *this;

  // We do this with with ParameterEvaluator which can do the remapping for us.
  ParameterEvaluator evaluator;
  auto remapType = [&](Type type) -> Type {
    return evaluator.getReboundType(type);
  };

  unsigned paramNo = 0;
  SmallVector<ParamDeclAttr> unboundDecls;
  for (auto [bind, decl] : llvm::zip(inputParamValues, getInputParams())) {
    if (bind.getName() != decl.getName()) {
      emitErrorFn() << "caller input parameter #" << paramNo << " has name "
                    << bind.getName() << " but callee expected name "
                    << decl.getName();
      return SignatureType();
    }

    // Bound parameters are allowed to refine the type of subsequent parameters,
    // e.g. in `<ty: type, fn: () -> !kgen.paramref<ty>>`, the expected type of
    // the second parameter will be refined when the first parameter is bound.
    auto remappedDeclType = remapType(decl.getType());
    if (bind.getType() != remappedDeclType) {
      emitErrorFn() << "caller input parameter #" << paramNo << " has type "
                    << bind.getType() << " but callee expected type "
                    << remappedDeclType;
      return SignatureType();
    }

    // If we're attempting to bind to an unknown attribute, we need to update
    // the decl, and keep it around so that we can continue to use it (as in a
    // partial bind).
    if (bind.getValue().isa<UnboundAttr>()) {
      unboundDecls.push_back(
          ParamDeclAttr::get(decl.getName(), remappedDeclType));
      // Set the binding to a declref of the thing itself - that will keep it
      // from becoming #kgen.unbound.
      evaluator.setParameterValue(
          bind.getDecl(),
          ParamDeclRefAttr::get(bind.getName(), bind.getType()));
    } else {
      evaluator.setParameterValue(bind.getDecl(), bind.getValue());
    }

    ++paramNo;
  }

  // Remap the parameter decls and result types.
  SmallVector<Type> newParamResultTypes;
  llvm::append_range(newParamResultTypes,
                     llvm::map_range(getResultParamTypes(), remapType));

  // Remap the value types.
  SmallVector<Type> inputTypes, resultTypes;
  llvm::append_range(inputTypes, llvm::map_range(getValueInputs(), remapType));
  llvm::append_range(resultTypes,
                     llvm::map_range(getValueResults(), remapType));

  return SignatureType::get(
      ParamDeclArrayAttr::get(getContext(), unboundDecls),
      TypeArrayAttr::get(getContext(), newParamResultTypes),
      FunctionType::get(getContext(), inputTypes, resultTypes),
      getConventions());
}

SignatureType SignatureType::getSpecializedSignature(
    ParamBindArrayAttr inputParams,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitErrorFn) {
  return getSpecializedSignature(inputParams.getValue(), emitErrorFn);
}

ArrayRef<Type> SignatureType::getValueInputs() {
  return getValues().getInputs();
}
ArrayRef<Type> SignatureType::getValueResults() {
  return getValues().getResults();
}

/// Return this signature type with the value signature replaced.
SignatureType SignatureType::getWithValuesReplaced(FunctionType fnType) {
  return SignatureType::get(getInputParams(), getResultParamTypes(), fnType,
                            getConventions());
}

SignatureType SignatureType::dropParamValues() {
  return get(ParamDeclArrayAttr::get(getContext(), {}), getResultParamTypes(),
             getValues(), getConventions());
}

bool SignatureType::isConcrete() {
  return getConventions().isDefault() && getInputParams().empty() &&
         getResultParamTypes().empty();
}

static ParseResult
parseValuesAndOptionalConventions(AsmParser &p,
                                  FailureOr<FunctionType> &valueFnSpec,
                                  FailureOr<ConventionsAttr> &conventions) {
  SmallVector<Type> inputs, outputs;
  ConventionsAttr conv;
  if (parseTypesWithConventions(p, inputs, outputs, conv))
    return failure();
  valueFnSpec = p.getBuilder().getFunctionType(inputs, outputs);
  conventions = conv;
  return success();
}

static void printValuesAndOptionalConventions(AsmPrinter &p,
                                              FunctionType valueFnSpec,
                                              ConventionsAttr conventions) {
  p << ' ';
  printTypesWithConventions(p.getStream(), valueFnSpec.getInputs(),
                            valueFnSpec.getResults(), conventions);
}

LogicalResult
SignatureType::verify(function_ref<InFlightDiagnostic()> emitError,
                      ParamDeclArrayAttr inputParams,
                      TypeArrayAttr resultParamTypes, FunctionType values,
                      ConventionsAttr conventions) {
  // Check we have the right number of conventions.
  if (conventions.getInputConventions().size() != values.getInputs().size())
    return emitError() << "incorrect # of input conventions specified";

  // If any signature parameters are force_inline, this signature must be as
  // well.
  for (ParamDeclAttr decl : inputParams) {
    auto sig = decl.getType().dyn_cast<SignatureType>();
    if (!sig)
      continue;

    // Found a signature, if it is force_inline then this must also be
    // force_inline.
    if (sig.isForceInline()) {
      if (!bitEnumContainsAny(conventions.getFnEffects(),
                              FnEffects::ForceInline)) {
        return emitError() << "signature input parameter " << decl.getName()
                           << " specified force_inline, and so expected "
                              "force_inline on this signature as well";
      }
    }
  }

  // If the function throws an error, make sure it has one result.
  if (bitEnumContainsAny(conventions.getFnEffects(), FnEffects::Throws) &&
      values.getNumResults() != 1)
    return emitError() << "a function that throws should have 1 result";
  return success();
}

//===----------------------------------------------------------------------===//
// DTypeType
//===----------------------------------------------------------------------===//

std::optional<int64_t> DTypeType::getTypeSize(TargetInfoAttr target) const {
  return sizeof(uint8_t);
}

std::optional<int64_t> DTypeType::getTypeAlign(TargetInfoAttr target) const {
  return 1;
}

//===----------------------------------------------------------------------===//
// DeclRefType
//===----------------------------------------------------------------------===//

DeclRefType DeclRefType::get(SymbolRefAttr name,
                             ParamBindArrayAttr paramValues) {
  return get(name.getContext(), name, paramValues);
}

DeclRefType DeclRefType::get(SymbolRefAttr name,
                             ArrayRef<ParamBindAttr> paramValues) {
  return get(name.getContext(), name,
             ParamBindArrayAttr::get(name.getContext(), paramValues));
}

DeclRefType DeclRefType::get(SymbolRefAttr name) {
  return get(name, ArrayRef<ParamBindAttr>());
}

//===----------------------------------------------------------------------===//
// StringType
//===----------------------------------------------------------------------===//

// A StringType is implemented as struct {char *address; size_t size;}.
// An index type as same alignment and size of a pointer type.
std::optional<int64_t>
KGEN::StringType::getTypeSize(TargetInfoAttr target) const {
  return 2 * target.getPointerSize();
}

std::optional<int64_t>
KGEN::StringType::getTypeAlign(TargetInfoAttr target) const {
  return target.getPointerSize();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

// Pull in the dialect definition.
#define GET_TYPEDEF_CLASSES
#include "KGEN/KGENDialect/KGENTypes.cpp.inc"
