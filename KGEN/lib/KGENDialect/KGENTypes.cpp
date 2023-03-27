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
#include "Support/TimeProfiler.h"
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

  // Register custom type parser and printers for KGEN types.
  registerPrettyType(
      "type", &MLIRTypeType::parse, TypeID::get<MLIRTypeType>(),
      +[](AsmPrinter &p, Type) { p << "type"; });
  registerMnemonicType<DTypeType>();
  registerMnemonicType<StringType>();
  registerMnemonicType<ListType>();
  registerMnemonicType<VariadicType>();
  registerMnemonicType<TargetType>();
  registerMnemonicType<BuildInfoType>();
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

LogicalResult MLIRTypeType::printValue(AsmPrinter &p, TypedAttr value) const {
  auto type = ::dyn_cast<TypeConstantAttr>(value);
  if (!type)
    return failure();
  p << type.getValue();
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
  // Parse a keyword or string as an MLIR operation attribute.
  std::string opName;
  llvm::SMLoc loc = p.getCurrentLocation();
  if (succeeded(p.parseOptionalString(&opName))) {
    NamedAttrList attrs;
    if (failed(p.parseOptionalAttrDict(attrs)))
      return failure();
    value = MLIROpAttr::getChecked([&] { return p.emitError(loc); },
                                   StringAttr::get(p.getContext(), opName),
                                   attrs.getDictionary(p.getContext()), *this);
    return mlir::success(!!value);
  }

  Attribute attr;
  OptionalParseResult result = p.parseOptionalAttribute(attr, *this);
  if (!result.has_value())
    return std::nullopt;
  if (failed(*result))
    return failure();

  // Parse a symbol reference as a signature type attribute.
  if (auto symbol = attr.dyn_cast<SymbolRefAttr>()) {
    // Parse any trailing parameter bindings.
    ParamBindArrayAttr paramValues;
    if (parseOptionalParamBindSpec(p, paramValues))
      return failure();
    value = SymbolConstantAttr::get(symbol, paramValues, *this);
  } else {
    value = attr.cast<TypedAttr>();
  }
  return mlir::success();
}

LogicalResult SignatureType::printValue(AsmPrinter &p, TypedAttr value) const {
  if (auto mlirOp = ::dyn_cast<MLIROpAttr>(value)) {
    p << mlirOp.getName();
    if (!mlirOp.getAttrs().empty())
      p << mlirOp.getAttrs();
    return success();
  }

  auto symbolCst = ::dyn_cast<SymbolConstantAttr>(value);
  if (!symbolCst)
    return failure();
  p << symbolCst.getSymbol();
  printOptionalParamBindSpec(p, symbolCst.getParamValues());
  return success();
}

static void getSignatureDefaults(ParamDeclArrayAttr &inputParams,
                                 ParamDeclArrayAttr &resultParams,
                                 FunctionType values, MetadataAttr &metadata) {
  MLIRContext *ctx = values.getContext();
  if (!inputParams)
    inputParams = ParamDeclArrayAttr::get(ctx, {});
  if (!resultParams)
    resultParams = ParamDeclArrayAttr::get(ctx, {});
  if (!metadata) {
    // Default value input conventions to take each argument by-value.
    metadata = MetadataAttr::get(ctx, values.getNumInputs());
  }
}

SignatureType SignatureType::get(ParamDeclArrayAttr inputParams,
                                 ParamDeclArrayAttr resultParams,
                                 FunctionType values, MetadataAttr metadata) {
  getSignatureDefaults(inputParams, resultParams, values, metadata);
  return get(values.getContext(), inputParams, resultParams, values, metadata);
}

SignatureType
SignatureType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                          ParamDeclArrayAttr inputParams,
                          ParamDeclArrayAttr resultParams, FunctionType values,
                          MetadataAttr metadata) {
  getSignatureDefaults(inputParams, resultParams, values, metadata);
  return getChecked(emitError, values.getContext(), inputParams, resultParams,
                    values, metadata);
}

SignatureType SignatureType::get(FunctionType values) {
  return get(ParamDeclArrayAttr(), {}, values, {});
}

SignatureType SignatureType::get(MLIRContext *ctx, TypeRange inputs,
                                 TypeRange results) {
  return get(FunctionType::get(ctx, inputs, results));
}

SignatureType SignatureType::getWithFnEffects(FnEffects effects) {
  return SignatureType::get(getInputParams(), getResultParams(), getValues(),
                            getMetadata().getWithFnEffects(effects));
}

bool SignatureType::isVararg(size_t index) {
  if (!bitEnumContainsAny(getFnEffects(), FnEffects::Vararg))
    return false;
  // If the function has keyword varargs, the vararg index is the second last.
  // Otherwise, it's the last.
  return (index + 1 +
          bitEnumContainsAny(getFnEffects(), FnEffects::KWVararg)) ==
         getValueInputs().size();
}

bool SignatureType::isKWVararg(size_t index) {
  if (!bitEnumContainsAny(getFnEffects(), FnEffects::KWVararg))
    return false;
  return index + 1 == getValueInputs().size();
}

/// Return true if this signature has a first argument is a result from the
/// function returned through memory.
bool SignatureType::hasMemoryOnlyResult() {
  auto conventions = getValueInputConventions();
  return conventions.size() >= 1 &&
         conventions[0] == ValueInputConvention::ByRefResult;
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
    function_ref<InFlightDiagnostic()> emitErrorFn) {
  TimeTraceScope<> traceScope("SignatureType::getSpecializedSignature");

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

  // We do this with with ParameterEvaluator which can do the remapping for us.
  ParameterEvaluator evaluator;

  auto remapType = [&](Type type) -> Type {
    return evaluator.getReboundType(type);
  };

  unsigned paramNo = 0;
  SmallVector<ParamDeclAttr, 16> unboundDecls;
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
      evaluator.setParameterValue(decl, ParamDeclRefAttr::get(decl));
    } else {
      evaluator.setParameterValue(bind.getName(), bind.getValue());
    }

    ++paramNo;
  }

  // FIXME: Signature typed attributes need to contain result parameter
  // declarations. For now, just bind them to themselves.
  for (ParamDeclAttr decl : getResultParams())
    evaluator.setParameterValue(decl, ParamDeclRefAttr::get(decl));

  // Remap the parameter decls and result parameter types.
  SmallVector<ParamDeclAttr, 16> newParamResults;
  llvm::append_range(
      newParamResults,
      llvm::map_range(getResultParams(), [&](ParamDeclAttr param) {
        return ParamDeclAttr::get(param.getName(), remapType(param.getType()));
      }));

  // Remap the value types.
  SmallVector<Type, 16> inputTypes, resultTypes;
  llvm::append_range(inputTypes, llvm::map_range(getValueInputs(), remapType));
  llvm::append_range(resultTypes,
                     llvm::map_range(getValueResults(), remapType));

  return SignatureType::get(
      ParamDeclArrayAttr::get(getContext(), unboundDecls),
      ParamDeclArrayAttr::get(getContext(), newParamResults),
      FunctionType::get(getContext(), inputTypes, resultTypes), getMetadata());
}

SignatureType SignatureType::getSpecializedSignature(
    ArrayRef<ParamBindAttr> inputParamValues,
    function_ref<InFlightDiagnostic()> emitErrorFn,
    ArrayRef<ParamDeclAttr> inputParams, ArrayRef<ParamDeclAttr> resultParams,
    FunctionType values, MetadataAttr metadata) {
  TimeTraceScope<> traceScope("SignatureType::getSpecializedSignature");

  // If the signature isn't parameterized, then there are no substitutions to
  // perform.
  MLIRContext *ctx = values.getContext();
  if (inputParamValues.empty())
    return SignatureType::get(ParamDeclArrayAttr::get(ctx, inputParams),
                              ParamDeclArrayAttr::get(ctx, resultParams),
                              values, metadata);

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

  // We do this with with ParameterEvaluator which can do the remapping for us.
  ParameterEvaluator evaluator;

  auto remapType = [&](Type type) -> Type {
    return evaluator.getReboundType(type);
  };

  unsigned paramNo = 0;
  SmallVector<ParamDeclAttr, 16> unboundDecls;
  for (auto [bind, decl] : llvm::zip(inputParamValues, inputParams)) {
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
      evaluator.setParameterValue(decl, ParamDeclRefAttr::get(decl));
    } else {
      evaluator.setParameterValue(bind.getName(), bind.getValue());
    }

    ++paramNo;
  }

  // FIXME: Signature typed attributes need to contain result parameter
  // declarations. For now, just bind them to themselves.
  for (ParamDeclAttr decl : resultParams)
    evaluator.setParameterValue(decl, ParamDeclRefAttr::get(decl));

  // Remap the parameter decls and result parameter types.
  SmallVector<ParamDeclAttr, 16> newParamResults;
  llvm::append_range(
      newParamResults, llvm::map_range(resultParams, [&](ParamDeclAttr param) {
        return ParamDeclAttr::get(param.getName(), remapType(param.getType()));
      }));

  // Remap the value types.
  SmallVector<Type, 16> inputTypes, resultTypes;
  llvm::append_range(inputTypes,
                     llvm::map_range(values.getInputs(), remapType));
  llvm::append_range(resultTypes,
                     llvm::map_range(values.getResults(), remapType));

  return SignatureType::get(
      ParamDeclArrayAttr::get(values.getContext(), unboundDecls),
      ParamDeclArrayAttr::get(values.getContext(), newParamResults),
      FunctionType::get(values.getContext(), inputTypes, resultTypes),
      metadata);
}

ArrayRef<Type> SignatureType::getValueInputs() const {
  return getValues().getInputs();
}
ArrayRef<Type> SignatureType::getValueResults() const {
  return getValues().getResults();
}

/// Return this signature type with the value signature replaced.
SignatureType SignatureType::getWithValuesReplaced(FunctionType fnType) {
  return SignatureType::get(getInputParams(), getResultParams(), fnType,
                            getMetadata());
}

SignatureType SignatureType::dropParamValues() {
  return get(ParamDeclArrayAttr::get(getContext(), {}), getResultParams(),
             getValues(), getMetadata());
}

bool SignatureType::isConcrete() {
  return getMetadata().isDefault() && getInputParams().empty() &&
         getResultParams().empty();
}

Type SignatureType::parse(AsmParser &p) {
  SignatureType signature;
  if (p.parseLess() || parseSignature(p, signature) || p.parseGreater())
    return {};
  return signature;
}

void SignatureType::print(AsmPrinter &p) const {
  p << '<';
  printSignature(p, *this);
  p << '>';
}

LogicalResult
SignatureType::verify(function_ref<InFlightDiagnostic()> emitError,
                      ParamDeclArrayAttr inputParams,
                      ParamDeclArrayAttr resultParams, FunctionType values,
                      MetadataAttr metadata) {
  return metadata.verifySignature(emitError, inputParams, resultParams, values);
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
  return 2 * llvm::alignTo(
                 llvm::divideCeil(target.getDataLayout().getPointerBitWidth(),
                                  CHAR_BIT),
                 target.getDataLayout().getPointerABIAlign());
}

std::optional<int64_t>
KGEN::StringType::getTypeAlign(TargetInfoAttr target) const {
  return target.getDataLayout().getPointerABIAlign();
}

//===----------------------------------------------------------------------===//
// VariadicType
//===----------------------------------------------------------------------===//

LogicalResult VariadicType::verify(function_ref<InFlightDiagnostic()> emitError,
                                   TypedAttr type) {
  assert(type && "type cannot be null");
  if (!type.getType().isa<MLIRTypeType>())
    return emitError() << "type parameter for pointer must be a !kgen.mlirtype";
  return success();
}

Type VariadicType::getResolvedElementType() const {
  if (auto typeCst = llvm::dyn_cast<TypeConstantAttr>(getElementType()))
    return typeCst.getValue();
  return nullptr;
}

VariadicType VariadicType::get(TypedAttr elementType) {
  return VariadicType::get(elementType.getContext(), elementType);
}

VariadicType VariadicType::get(Type elementType) {
  return VariadicType::get(TypeConstantAttr::get(elementType));
}

/// A variadic type is like an `llvm::ArrayRef`: a pointer to the start of the
/// contiguous sequence, and the size of that seqeunce. So, its size would be
/// the size of a pointer, plus the size of the size type (which has the same
/// size and alignment as a pointer type).
std::optional<int64_t> VariadicType::getTypeSize(TargetInfoAttr target) const {
  return 2 * llvm::alignTo(
                 llvm::divideCeil(target.getDataLayout().getPointerBitWidth(),
                                  CHAR_BIT),
                 target.getDataLayout().getPointerABIAlign());
}

/// The alignment of the variadic type is that its pointer and size.
std::optional<int64_t> VariadicType::getTypeAlign(TargetInfoAttr target) const {
  return target.getDataLayout().getPointerABIAlign();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

// Pull in the dialect definition.
#define GET_TYPEDEF_CLASSES
#include "KGEN/KGENDialect/KGENTypes.cpp.inc"
