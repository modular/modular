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

  // Register custom type parser and printers for KGEN types.
  registerPrettyType(
      "type", &MLIRTypeType::parse, TypeID::get<MLIRTypeType>(),
      +[](AsmPrinter &p, Type) { p << "type"; });
  registerMnemonicType<DTypeType>();
  registerMnemonicType<StringType>();
  registerMnemonicType<ListType>();
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

SignatureType SignatureType::setFnEffect(FnEffects effect) {
  return SignatureType::get(
      getInputParams(), getResultParams(), getValues(),
      MetadataAttr::get(getContext(), getValueInputConventions(),
                        getDefaultArguments(),
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
  SmallVector<ParamDeclAttr> newParamResults;
  llvm::append_range(
      newParamResults,
      llvm::map_range(getResultParams(), [&](ParamDeclAttr param) {
        return ParamDeclAttr::get(param.getName(), remapType(param.getType()));
      }));

  // Remap the value types.
  SmallVector<Type> inputTypes, resultTypes;
  llvm::append_range(inputTypes, llvm::map_range(getValueInputs(), remapType));
  llvm::append_range(resultTypes,
                     llvm::map_range(getValueResults(), remapType));

  return SignatureType::get(
      ParamDeclArrayAttr::get(getContext(), unboundDecls),
      ParamDeclArrayAttr::get(getContext(), newParamResults),
      FunctionType::get(getContext(), inputTypes, resultTypes), getMetadata());
}

SignatureType SignatureType::getSpecializedSignature(
    ParamBindArrayAttr inputParams,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitErrorFn) {
  return getSpecializedSignature(inputParams.getValue(), emitErrorFn);
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
  if (p.parseLess())
    return {};
  llvm::SMLoc loc = p.getCurrentLocation();
  ParamDeclArrayAttr inputParams, resultParams;
  if (parseOptionalParameterSpec(p, inputParams, resultParams))
    return {};
  SmallVector<Type> inputs, outputs;
  MetadataAttr metadata;
  if (parseTypesWithMetadata(p, inputs, outputs, metadata))
    return {};
  if (p.parseGreater())
    return {};
  return getChecked([&] { return p.emitError(loc); }, inputParams, resultParams,
                    p.getBuilder().getFunctionType(inputs, outputs), metadata);
}

void SignatureType::print(AsmPrinter &p) const {
  p << '<';
  printOptionalParameterSpec(p, getInputParams(), getResultParams());
  printTypesWithMetadata(p, getValueInputs(), getValueResults(), getMetadata());
  p << '>';
}

LogicalResult
SignatureType::verify(function_ref<InFlightDiagnostic()> emitError,
                      ParamDeclArrayAttr inputParams,
                      ParamDeclArrayAttr resultParams, FunctionType values,
                      MetadataAttr metadata) {
  // Check we have the right number of conventions.
  if (metadata.getInputConventions().size() != values.getInputs().size())
    return emitError() << "incorrect # of input conventions specified";

  DefaultArgumentsAttr defaults = metadata.getDefaultArguments();
  if (defaults) {
    for (auto [defaultsIndex, value] : llvm::enumerate(defaults.getValues())) {
      size_t index = values.getInputs().size() - defaults.getValues().size() +
                     defaultsIndex;
      Type expected = values.getInputs()[index];
      if (value.getType() != expected) {
        return emitError() << "argument #" << index << " has type " << expected
                           << " but default argument has type "
                           << value.getType();
      }
    }
  }

  // If the function throws an error, make sure it has one result.
  if (bitEnumContainsAny(metadata.getFnEffects(), FnEffects::Throws) &&
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
