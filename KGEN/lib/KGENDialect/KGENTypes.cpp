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
  registerMnemonicType<PointerType>();
  registerMnemonicType<StringType>();
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
  OptionalParseResult result = parseOptionalKGENType(p, type);
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
  printKGENType(p, type.getValue());
  return success();
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
    ParameterExprArrayAttr paramValues;
    if (parseParameterValues(p, paramValues))
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
  printParameterValues(p, symbolCst.getParamValues());
  return success();
}

static void getSignatureDefaults(TypeArrayAttr &inputParamTypes,
                                 TypeArrayAttr &resultParamTypes,
                                 FunctionType values,
                                 FnMetadataAttr &metadata) {
  MLIRContext *ctx = values.getContext();
  if (!inputParamTypes)
    inputParamTypes = TypeArrayAttr::get(ctx, {});
  if (!resultParamTypes)
    resultParamTypes = TypeArrayAttr::get(ctx, {});
  if (!metadata) {
    // Default value input conventions to take each argument by-value.
    metadata = FnMetadataAttr::get(ctx, values.getNumInputs());
  }
}

SignatureType SignatureType::get(TypeArrayAttr inputParamTypes,
                                 TypeArrayAttr resultParamTypes,
                                 FunctionType values, FnMetadataAttr metadata) {
  getSignatureDefaults(inputParamTypes, resultParamTypes, values, metadata);
  return get(values.getContext(), inputParamTypes, resultParamTypes, values,
             metadata);
}

SignatureType
SignatureType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                          TypeArrayAttr inputParamTypes,
                          TypeArrayAttr resultParamTypes, FunctionType values,
                          FnMetadataAttr metadata) {
  getSignatureDefaults(inputParamTypes, resultParamTypes, values, metadata);
  return getChecked(emitError, values.getContext(), inputParamTypes,
                    resultParamTypes, values, metadata);
}

SignatureType SignatureType::get(FunctionType values) {
  return get(TypeArrayAttr(), {}, values, {});
}

SignatureType SignatureType::get(MLIRContext *ctx, TypeRange inputs,
                                 TypeRange results) {
  return get(FunctionType::get(ctx, inputs, results));
}

SignatureType SignatureType::getWithFnEffects(FnEffects effects) {
  return SignatureType::get(getInputParamTypes(), getResultParamTypes(),
                            getValues(),
                            getMetadata().getWithFnEffects(effects));
}

static bool isVarargKind(FnEffects effects, size_t numInputs, size_t index,
                         FnEffects kind) {
  if (!bitEnumContainsAny(effects, kind))
    return false;
  // If the function has keyword varargs, the vararg index is the second last.
  // Otherwise, it's the last.
  return (index + 1 + bitEnumContainsAny(effects, FnEffects::KWVararg)) ==
         numInputs;
}

bool SignatureType::isVararg(size_t index) {
  return isVarargKind(getFnEffects(), getNumInputs(), index, FnEffects::Vararg);
}

bool SignatureType::isPackVararg(size_t index) {
  return isVarargKind(getFnEffects(), getNumInputs(), index,
                      FnEffects::PackVararg);
}

bool SignatureType::isKWVararg(size_t index) {
  if (!bitEnumContainsAny(getFnEffects(), FnEffects::KWVararg))
    return false;
  return index + 1 == getNumInputs();
}

bool SignatureType::hasMemoryOnlyResult() {
  auto conventions = getValueInputConventions();
  return conventions.size() >= 1 &&
         conventions[0] == ValueInputConvention::ByRefResult;
}

bool SignatureType::hasInitSelfResult() {
  auto conventions = getValueInputConventions();
  return conventions.size() >= 1 &&
         conventions[0] == ValueInputConvention::InitSelf;
}

/// Return a signature with the specified parameter bindings substituted
/// into it as happens in a call.  The types specified in the parameter
/// bindings affects the type signature of the value input and outputs, and
/// also can remap the signature in the parameter list itself.
///
/// If an error occurs making the substitution, report it with emitErrorFn
/// and return null.
SignatureType SignatureType::getSpecializedSignature(
    ArrayRef<TypedAttr> inputParamValues,
    function_ref<InFlightDiagnostic()> emitErrorFn) {
  if (inputParamValues.empty())
    return *this;
  return getSpecializedSignature(inputParamValues, emitErrorFn,
                                 getInputParamTypes(), getResultParamTypes(),
                                 getValues(), getMetadata());
}

SignatureType SignatureType::getSpecializedSignature(
    ArrayRef<TypedAttr> inputParamValues,
    function_ref<InFlightDiagnostic()> emitErrorFn,
    ArrayRef<Type> inputParamTypes, ArrayRef<Type> resultParamTypes,
    FunctionType values, FnMetadataAttr metadata) {
  TimeTraceScope<> traceScope("SignatureType::getSpecializedSignature");

  // If the signature isn't parameterized, then there are no substitutions to
  // perform.
  MLIRContext *ctx = values.getContext();
  if (inputParamValues.empty())
    return SignatureType::get(TypeArrayAttr::get(ctx, inputParamTypes),
                              TypeArrayAttr::get(ctx, resultParamTypes), values,
                              metadata);

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
  evaluator.setInputDepth(1);

  auto remapType = [&](Type type) -> Type {
    return evaluator.getReboundType(type);
  };

  unsigned paramNo = 0;
  SmallVector<Type, 16> unboundParamTypes;
  for (auto [value, type] : llvm::zip(inputParamValues, inputParamTypes)) {
    // Bound parameters are allowed to refine the type of subsequent parameters,
    // e.g. in `<ty: type, fn: () -> !kgen.paramref<ty>>`, the expected type of
    // the second parameter will be refined when the first parameter is bound.
    auto remappedDeclType = remapType(type);
    if (value.getType() != remappedDeclType) {
      emitErrorFn() << "caller input parameter #" << paramNo << " has type "
                    << value.getType() << " but callee expected type "
                    << remappedDeclType;
      return SignatureType();
    }

    // If we're attempting to bind to an unknown attribute, we need to update
    // the decl, and keep it around so that we can continue to use it (as in a
    // partial bind).
    if (value.isa<UnboundAttr>()) {
      // Set the binding to a declref of the thing itself - that will keep it
      // from becoming #kgen.unbound.
      auto value =
          ParamIndexRefAttr::get(/*depth=*/-1, /*isResult=*/false,
                                 unboundParamTypes.size(), remappedDeclType);
      unboundParamTypes.push_back(remappedDeclType);
      evaluator.addInputValue(value);
    } else {
      evaluator.addInputValue(value);
    }

    ++paramNo;
  }

  // FIXME: Signature typed attributes need to contain result parameter
  // declarations. For now, just bind them to themselves.
  for (auto [idx, type] : llvm::enumerate(resultParamTypes)) {
    evaluator.addResultValue(
        ParamIndexRefAttr::get(/*depth=*/-1, /*isResult=*/true, idx, type));
  }

  // Remap the parameter decls and result parameter types.
  SmallVector<Type, 16> newParamResultTypes;
  llvm::append_range(newParamResultTypes,
                     llvm::map_range(resultParamTypes, remapType));

  // Remap the value types.
  SmallVector<Type, 16> inputTypes, resultTypes;
  llvm::append_range(inputTypes,
                     llvm::map_range(values.getInputs(), remapType));
  llvm::append_range(resultTypes,
                     llvm::map_range(values.getResults(), remapType));

  return SignatureType::get(
      TypeArrayAttr::get(values.getContext(), unboundParamTypes),
      TypeArrayAttr::get(values.getContext(), newParamResultTypes),
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
  return SignatureType::get(getInputParamTypes(), getResultParamTypes(), fnType,
                            getMetadata());
}

SignatureType SignatureType::dropParamValues() {
  return get(TypeArrayAttr::get(getContext(), {}), getResultParamTypes(),
             getValues(), getMetadata());
}

bool SignatureType::isConcrete() {
  return getMetadata().isDefault() && getInputParamTypes().empty() &&
         getResultParamTypes().empty();
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
                      TypeArrayAttr inputParams, TypeArrayAttr resultParams,
                      FunctionType values, FnMetadataAttr metadata) {
  if (!inputParams || !resultParams || !values || !metadata)
    return emitError() << "signature type parameters cannot be null";

  // Check we have the right number of conventions.
  if (metadata.getInputConventions().size() != values.getInputs().size())
    return emitError() << "incorrect # of input conventions specified";

  bool hasVararg = bitEnumContainsAny(
      metadata.getFnEffects(), FnEffects::Vararg | FnEffects::PackVararg);
  unsigned minNumArgs = hasVararg + bitEnumContainsAny(metadata.getFnEffects(),
                                                       FnEffects::KWVararg);
  if (values.getNumInputs() < minNumArgs) {
    return emitError()
           << "function has varargs and/or kwvarargs but signature only has "
           << values.getNumInputs() << " arguments";
  }

  // Verify input convention and argument types.
  for (auto [i, argType, conv] :
       llvm::enumerate(values.getInputs(), metadata.getInputConventions())) {
    Type type = argType;
    // Verify variadics.
    if (isVarargKind(metadata.getFnEffects(), values.getNumInputs(), i,
                     FnEffects::Vararg)) {
      auto variadic = ::dyn_cast<VariadicType>(type);
      if (!variadic) {
        return emitError() << "argument #" << i
                           << " in signature with varargs should be a "
                              "`!kgen.variadic` but got: "
                           << type;
      }
      type = variadic.getElementAsType();
    }
    // Verify argument conventions.
    switch (conv) {
    case ValueInputConvention::BorrowedInMem:
    case ValueInputConvention::ByRef:
    case ValueInputConvention::ByRefResult:
    case ValueInputConvention::InitSelf:
    case ValueInputConvention::OwnedInMem:
      if (::isa<PointerType>(type))
        break;
      if (type.getDialect().getNamespace() == "lit")
        break; // lit.ref is also ok, but we can't check it directly.
      return emitError()
             << "argument #" << i << " with convention '" << stringifyEnum(conv)
             << "' in signature type should be a `!kgen.pointer` but got: "
             << type;
    default:
      break;
    }
  }

  ArrayRef<TypedAttr> defaults = metadata.getDefaultArguments();
  for (auto [defaultsIndex, value] : llvm::enumerate(defaults)) {
    size_t index = values.getInputs().size() - defaults.size() + defaultsIndex;
    Type expected = values.getInputs()[index];
    if (value.getType() != expected) {
      return emitError() << "argument #" << index << " has type " << expected
                         << " but default argument has type "
                         << value.getType();
    }
  }
  // If the function throws an error, make sure it has one variant result.
  if (bitEnumContainsAny(metadata.getFnEffects(), FnEffects::Throws) &&
      values.getNumResults() != 1)
    return emitError() << "a function that throws should have 1 result";

  return success();
}

std::optional<int64_t> SignatureType::getTypeSize(TargetInfoAttr target) const {
  // Non-capturing closures are function pointers. Capturing closures contain
  // a function pointer and a capture state pointer.
  return (isCapturing() ? 2 : 1) * target.getDataLayout().getPointerSize();
}

std::optional<int64_t>
SignatureType::getTypeAlign(TargetInfoAttr target) const {
  return target.getDataLayout().getPointerABIAlign();
}

//===----------------------------------------------------------------------===//
// IndexRefRemapper
//===----------------------------------------------------------------------===//

template <typename T>
auto IndexRefRemapper::normalizeSignatureWalk(T value, size_t depth)
    -> std::conditional_t<std::is_base_of_v<Type, T>, Type, Attribute> {
  if constexpr (std::is_base_of_v<Attribute, T>) {
    if (!mapping.empty()) {
      if (auto ref = dyn_cast<ParamDeclRefAttr>(value)) {
        auto it = mapping.find(ref.getName());
        if (it == mapping.end())
          return ref;
        auto [idx, isResult] = it->second;
        return ParamIndexRefAttr::get(
            depth, isResult, idx, normalizeSignatureWalk(ref.getType(), depth));
      }
    }
    if (offset != 0) {
      if (auto ref = dyn_cast<ParamIndexRefAttr>(value)) {
        if (ref.getDepth() != depth)
          return ref;
        return ParamIndexRefAttr::get(
            depth, ref.getIsResult(), ref.getIndex() + offset,
            normalizeSignatureWalk(ref.getType(), depth));
      }
    }
  }
  if constexpr (std::is_base_of_v<Type, T>) {
    if (isa<SignatureType>(value))
      ++depth;
  }
  SmallVector<Attribute, 16> newAttrs;
  SmallVector<Type, 16> newTypes;
  bool changed = false;
  value.walkImmediateSubElements(
      [&](Attribute attr) {
        Attribute newAttr = normalizeSignatureWalk(attr, depth);
        changed |= newAttr != attr;
        newAttrs.push_back(newAttr);
      },
      [&](Type type) {
        Type newType = normalizeSignatureWalk(type, depth);
        changed |= newType != type;
        newTypes.push_back(newType);
      });
  if (!changed)
    return value;
  return value.replaceImmediateSubElements(newAttrs, newTypes);
}

IndexRefRemapper::IndexRefRemapper(ArrayRef<ParamDeclAttr> inputParams,
                                   ArrayRef<ParamDeclAttr> resultParams,
                                   size_t offset)
    : offset(offset) {
  auto mapIndices = [&](ArrayRef<ParamDeclAttr> params, bool isResult) {
    for (auto [idx, param] : llvm::enumerate(params))
      mapping.try_emplace(param.getName(), std::make_pair(idx, isResult));
  };
  mapIndices(inputParams, /*isResult=*/false);
  mapIndices(resultParams, /*isResult=*/true);
}

Attribute IndexRefRemapper::remapAttrImpl(Attribute attr) {
  return normalizeSignatureWalk(attr);
}

Type IndexRefRemapper::remapTypeImpl(Type type) {
  return normalizeSignatureWalk(type);
}

SignatureType IndexRefRemapper::remapToSignature(
    ArrayRef<ParamDeclAttr> inputParams, ArrayRef<ParamDeclAttr> resultParams,
    FunctionType functionType, FnMetadataAttr metadata,
    function_ref<InFlightDiagnostic()> emitError) {
  IndexRefRemapper remapper(inputParams, resultParams);
  SmallVector<Type> inputParamTypes, resultParamTypes;
  for (ParamDeclAttr param : inputParams)
    inputParamTypes.push_back(remapper.remap(param.getType()));
  for (ParamDeclAttr param : resultParams)
    resultParamTypes.push_back(remapper.remap(param.getType()));

  if (!emitError) {
    emitError = []() -> InFlightDiagnostic {
      llvm_unreachable("invalid signature");
    };
  }

  MLIRContext *ctx = functionType.getContext();
  return SignatureType::getChecked(
      emitError, TypeArrayAttr::get(ctx, inputParamTypes),
      TypeArrayAttr::get(ctx, resultParamTypes), remapper.remap(functionType),
      metadata ? remapper.remap(metadata) : nullptr);
}

SignatureType
IndexRefRemapper::prependParams(SignatureType sig,
                                ArrayRef<ParamDeclAttr> parentParams) {
  IndexRefRemapper remapper(parentParams, {}, parentParams.size());
  SmallVector<Type> inputParamTypes;
  for (ParamDeclAttr param : parentParams)
    inputParamTypes.push_back(remapper.remap(param.getType()));
  for (Type type : sig.getInputParamTypes())
    inputParamTypes.push_back(remapper.remap(type));
  return SignatureType::get(
      TypeArrayAttr::get(sig.getContext(), inputParamTypes),
      remapper.remap(sig.getResultParamTypes()),
      remapper.remap(sig.getValues()), remapper.remap(sig.getMetadata()));
}

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

LogicalResult PointerType::verify(function_ref<InFlightDiagnostic()> emitError,
                                  TypedAttr type, TypedAttr addressSpace) {
  if (type && !::isa<MLIRTypeType>(type.getType()))
    return emitError() << "type parameter for pointer must be a !kgen.mlirtype";
  if (addressSpace && !addressSpace.getType().isIndex())
    return emitError() << "address space parameter `" << addressSpace
                       << "` must be an index type";
  return success();
}

Type PointerType::getElementAsType() const {
  TypedAttr elemType = getElementType();
  if (auto typeCst = ::dyn_cast<TypeConstantAttr>(elemType))
    return typeCst.getValue();
  assert(::isa<MLIRTypeType>(elemType.getType()) &&
         "parameter expr must have metatype type");
  return ParamRefType::get(elemType);
}

PointerType PointerType::get(TypedAttr elementType, unsigned addressSpace) {
  MLIRContext *ctx = elementType.getContext();
  return PointerType::get(ctx, elementType,
                          IntegerAttr::get(IndexType::get(ctx), addressSpace));
}

PointerType PointerType::get(Type elementType, unsigned addressSpace) {
  return get(TypeConstantAttr::get(elementType), addressSpace);
}

PointerType PointerType::get(TypedAttr elementType, TypedAttr addressSpace) {
  return get(addressSpace.getContext(), elementType, addressSpace);
}

PointerType PointerType::get(Type elementType, TypedAttr addressSpace) {
  return get(TypeConstantAttr::get(elementType), addressSpace);
}

std::optional<int64_t> PointerType::getTypeSize(TargetInfoAttr target) const {
  return target.getDataLayout().getPointerSize();
}

std::optional<int64_t> PointerType::getTypeAlign(TargetInfoAttr target) const {
  return target.getDataLayout().getPointerABIAlign();
}

ErrorOrSuccess PointerType::writeTo(TypedAttr value, int64_t addr,
                                    InterpreterState &state) const {
  int64_t size = *getTypeSize(state.getTarget());
  ErrorOr<void *> mem =
      state.getWritableMemory(addr, size, /*writePointer=*/true);
  if (mem.isError())
    return mem.takeError();
  // The pointer size of the target is variable.
  APInt intVal(size * CHAR_BIT, value.cast<PointerAttr>().getAddr());
  llvm::StoreIntToMemory(intVal, reinterpret_cast<uint8_t *>(*mem), size);
  return success();
}

ErrorOr<TypedAttr> PointerType::readFrom(int64_t addr,
                                         InterpreterState &state) const {
  int64_t size = *getTypeSize(state.getTarget());
  ErrorOr<const void *> mem = state.getReadableMemory(addr, size);
  if (mem.isError())
    return mem.takeError();
  APInt intVal(size * CHAR_BIT, 0);
  llvm::LoadIntFromMemory(intVal, (const uint8_t *)*mem, size);
  return PointerAttr::get(intVal.getLimitedValue(), *this);
}

OptionalParseResult PointerType::parseValue(AsmParser &p,
                                            TypedAttr &value) const {
  int64_t addr;
  // Parse an integer as a raw pointer attribute.
  if (OptionalParseResult result = p.parseOptionalInteger(addr);
      result.has_value()) {
    if (failed(*result))
      return failure();
    value = PointerAttr::get(addr, *this);
    return mlir::success();
  }

  // Parse a `store_to_mem` directive.
  if (succeeded(p.parseOptionalKeyword("store_to_mem"))) {
    TypedAttr memValue;
    if (p.parseLParen() || parseParamValue(p, memValue, getElementAsType()) ||
        p.parseRParen())
      return failure();
    value = StoreToMemAttr::get(memValue, *this);
    return mlir::success();
  }

  return {};
}

LogicalResult PointerType::printValue(AsmPrinter &p, TypedAttr value) const {
  // Print a raw pointer attribute as an integer.
  if (auto ptrAttr = ::dyn_cast<PointerAttr>(value)) {
    p << ptrAttr.getAddr();
    return success();
  }

  // Print a `store_to_mem` directive.
  if (auto memAttr = ::dyn_cast<StoreToMemAttr>(value)) {
    p << "store_to_mem(";
    printParamValue(p, memAttr.getValue());
    p << ')';
    return success();
  }

  return failure();
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

std::optional<StringRef> DeclRefType::getAliasName() {
  // Don't alias types with parameter references.
  if (!getParamValues().empty())
    return {};
  StringRef rootName = getSymbol().getRootReference().getValue();

  // Alias declref types that have mangled names.
  if (llvm::all_of(rootName, [](char c) { return std::isalnum(c); }))
    return {};

  // Use the leaf name as the alias name.
  StringRef leaf = getSymbol().getLeafReference().getValue();
  unsigned offset = leaf.size();
  while (offset > 0 && std::isalnum(leaf[offset - 1]))
    --offset;
  if (offset == leaf.size())
    return {};
  return leaf.substr(offset);
}

//===----------------------------------------------------------------------===//
// StringType
//===----------------------------------------------------------------------===//

// A StringType is implemented as struct {char *address; size_t size;}.
// An index type as same alignment and size of a pointer type.
std::optional<int64_t>
KGEN::StringType::getTypeSize(TargetInfoAttr target) const {
  return 2 * llvm::alignTo(target.getDataLayout().getPointerSize(),
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

Type VariadicType::getElementAsType() const {
  TypedAttr eltType = getElementType();
  if (auto typeCst = llvm::dyn_cast<TypeConstantAttr>(eltType))
    return typeCst.getValue();
  assert(::isa<MLIRTypeType>(eltType.getType()) &&
         "parameter expr must have metatype type");
  return ParamRefType::get(eltType);
}

VariadicType VariadicType::get(TypedAttr elementType) {
  return VariadicType::get(elementType.getContext(), elementType);
}

VariadicType VariadicType::get(Type elementType) {
  return VariadicType::get(TypeConstantAttr::get(elementType));
}

/// A variadic type is like an `llvm::ArrayRef`: a pointer to the start of the
/// contiguous sequence, and the size of that sequence. So, its size would be
/// the size of a pointer, plus the size of the size type (which has the same
/// size and alignment as a pointer type).
std::optional<int64_t> VariadicType::getTypeSize(TargetInfoAttr target) const {
  return 2 * llvm::alignTo(target.getDataLayout().getPointerSize(),
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
