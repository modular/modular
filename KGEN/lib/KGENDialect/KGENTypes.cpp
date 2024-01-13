//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/KGENDType.h"
#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "Support/MDialect/MTypeInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// FnEffects
//===----------------------------------------------------------------------===//

namespace M::KGEN {
static llvm::hash_code hash_value(FnEffects effects) {
  return llvm::hash_value(static_cast<uint16_t>(effects.getImpl()));
}
} // namespace M::KGEN

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
      "regtype", &AnyRegTypeType::parse, TypeID::get<AnyRegTypeType>(),
      +[](AsmPrinter &p, Type) { p << "regtype"; });
  registerMnemonicType<DTypeType>();
  registerMnemonicType<PointerType>();
  registerMnemonicType<NoneType>();
  registerMnemonicType<StringType>();
  registerMnemonicType<VariadicType>();
  registerMnemonicType<TargetType>();
  registerMnemonicType<BuildInfoType>();
  registerMnemonicType<StructType>();
  registerMnemonicType<VariantType>();
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
// AnyRegTypeType
//===----------------------------------------------------------------------===//

/// Implementation of the parsing logic for sugar types (e.g. !kgen.anytype).
static OptionalParseResult parseSugaredTypeValue(AsmParser &p, TypedAttr &value,
                                                 Type type) {
  Type typeValue;
  bool parsingVTable = succeeded(p.parseOptionalLSquare());
  auto vtable = VTableAttr::get(p.getContext(), {});

  OptionalParseResult result = parseOptionalKGENType(p, typeValue);
  if (!result.has_value()) {
    // If a '[' was seen, require a type to be present.
    if (parsingVTable)
      return p.emitError(p.getCurrentLocation(), "expected a type");
    return {};
  }
  if (failed(*result))
    return failure();

  // Parse the vtable if a '[' was seen.
  if (parsingVTable) {
    if (p.parseComma() || p.parseLBrace() ||
        (p.parseOptionalRBrace() &&
         (!(vtable = cast_or_null<VTableAttr>(VTableAttr::parse(p, {}))) ||
          p.parseRBrace())) ||
        p.parseRSquare())
      return failure();
  }

  value = TypeConstantAttr::get(typeValue, type, vtable);
  return mlir::success();
}

/// Implementation of the printing logic for sugar types (e.g. !kgen.anytype).
static LogicalResult printSugaredTypeValue(AsmPrinter &p, TypedAttr value) {
  auto type = dyn_cast<TypeConstantAttr>(value);
  if (!type)
    return failure();

  VTableAttr vtable = type.getVTable();
  if (!vtable.getEntries().empty())
    p << '[';
  printKGENType(p, type.getValue());
  if (!vtable.getEntries().empty()) {
    p << ", {";
    vtable.print(p);
    p << "}]";
  }
  return success();
}

OptionalParseResult AnyRegTypeType::parseValue(AsmParser &p,
                                               TypedAttr &value) const {
  return parseSugaredTypeValue(p, value, *this);
}

LogicalResult AnyRegTypeType::printValue(AsmPrinter &p, TypedAttr value) const {
  return printSugaredTypeValue(p, value);
}

std::optional<int64_t>
AnyRegTypeType::getTypeSize(TargetInfoAttr target) const {
  // TODO: Types don't have a runtime representation yet! But one can imagine it
  // would contain a type ID, and a pointer to the witness table.
  return target.getDataLayout().getPointerSize() * 2;
}

std::optional<int64_t>
AnyRegTypeType::getTypeAlign(TargetInfoAttr target) const {
  return target.getDataLayout().getPointerABIAlign();
}

ErrorOrSuccess AnyRegTypeType::writeTo(TypedAttr value, int64_t addr,
                                       InterpreterState &state) const {
  ErrorOr<void *> mem =
      state.getWritableMemory(addr, *getTypeSize(state.getTarget()));
  if (mem)
    return mem.takeError();

  // Without a concrete runtime representation, just make sure the value can be
  // roundtripped.
  unsigned ptrSize = state.getTarget().getDataLayout().getPointerSize();
  llvm::StoreIntToMemory(
      APInt(ptrSize * 8, (uint64_t)value.getAsOpaquePointer()), (uint8_t *)*mem,
      ptrSize);
  return success();
}

ErrorOr<TypedAttr> AnyRegTypeType::readFrom(int64_t addr,
                                            InterpreterState &state) const {
  ErrorOr<const void *> mem =
      state.getReadableMemory(addr, *getTypeSize(state.getTarget()));
  if (mem)
    return mem.takeError();

  // Without a concrete runtime representation, just make sure the value can be
  // roundtripped.
  unsigned ptrSize = state.getTarget().getDataLayout().getPointerSize();
  APInt opaque(ptrSize * 8, 0);
  llvm::LoadIntFromMemory(opaque, (const uint8_t *)*mem, ptrSize);
  return ::cast<TypedAttr>(
      Attribute::getFromOpaquePointer((const void *)opaque.getLimitedValue()));
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

SignatureType SignatureType::get(MLIRContext *context, TypeRange inputs,
                                 TypeRange results) {
  return get(FunctionType::get(context, inputs, results));
}

SignatureType SignatureType::getWithFnEffects(FnEffects effects) {
  return SignatureType::get(getValues(), getInputParamTypes(),
                            getResultParamTypes(), getInputConventions(),
                            effects, getMetadata());
}

bool SignatureType::isVarArg(size_t index) {
  if (!getFnEffects().hasVarArgs())
    return false;
  return getFnEffects().isVarArg(getNumInputs(), index);
}

bool SignatureType::isPackVarArg(size_t index) {
  if (!getFnEffects().hasPackVarArgs())
    return false;
  return getFnEffects().isVarArg(getNumInputs(), index);
}

bool SignatureType::isKWVarArg(size_t index) {
  if (!getFnEffects().hasKWVarArgs())
    return false;
  return index + 1 == getNumInputs();
}

bool SignatureType::hasMemoryOnlyResult() {
  ArrayRef<ValueInputConvention> conventions = getInputConventions();
  return !conventions.empty() &&
         conventions[0] == ValueInputConvention::ByRefResult;
}

bool SignatureType::hasInitSelfResult() {
  ArrayRef<ValueInputConvention> conventions = getInputConventions();
  return !conventions.empty() &&
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
                                 getValues(), getInputConventions(),
                                 getFnEffects(), getMetadata());
}

SignatureType
SignatureType::getSpecializedSignature(ArrayRef<TypedAttr> inputParamValues,
                                       Location location) {
  return getSpecializedSignature(inputParamValues, [&]() -> InFlightDiagnostic {
    return emitError(location);
  });
}

SignatureType SignatureType::getSpecializedSignature(
    ArrayRef<TypedAttr> inputParamValues,
    function_ref<InFlightDiagnostic()> emitErrorFn,
    ArrayRef<Type> inputParamTypes, ArrayRef<Type> resultParamTypes,
    FunctionType values, ArrayRef<ValueInputConvention> inputConventions,
    FnEffects effects, FnMetadataAttrInterface metadata) {
  CompilerTimeTraceScope traceScope("SignatureType::getSpecializedSignature");

  // If the signature isn't parameterized, then there are no substitutions to
  // perform.
  if (inputParamValues.empty()) {
    return SignatureType::get(values, inputParamTypes, resultParamTypes,
                              inputConventions, effects, metadata);
  }

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

  SmallVector<Type, 16> unboundParamTypes;
  llvm::BitVector boundParams(inputParamTypes.size());
  for (auto [paramNo, value, type] :
       llvm::enumerate(inputParamValues, inputParamTypes)) {
    // Bound parameters are allowed to refine the type of subsequent parameters,
    // e.g. in `<ty: type, fn: () -> !kgen.paramref<ty>>`, the expected type of
    // the second parameter will be refined when the first parameter is bound.
    auto remappedDeclType = remapType(type);
    if (value.getType() != remappedDeclType) {
      assert(emitErrorFn && "unexpected invalid signature");
      emitErrorFn() << "caller input parameter #" << paramNo << " has type "
                    << value.getType() << " but callee expected type "
                    << remappedDeclType;
      return SignatureType();
    }

    // If we're attempting to bind to an unknown attribute, we need to update
    // the decl, and keep it around so that we can continue to use it (as in a
    // partial bind).
    if (::isa<UnboundAttr>(value)) {
      // Set the binding to a declref of the thing itself - that will keep it
      // from becoming #kgen.unbound.
      auto value =
          ParamIndexRefAttr::get(/*depth=*/-1, /*isResult=*/false,
                                 unboundParamTypes.size(), remappedDeclType);
      unboundParamTypes.push_back(remappedDeclType);
      evaluator.addInputValue(value);
    } else {
      if (::isa<VariadicType>(type))
        effects.setParamVarArgs(false);
      evaluator.addInputValue(value);
      boundParams.set(paramNo);
    }
  }

  // FIXME: Signature typed attributes need to contain result parameter
  // declarations. For now, just bind them to themselves.
  for (auto [idx, type] : llvm::enumerate(resultParamTypes)) {
    evaluator.addResultValue(
        ParamIndexRefAttr::get(/*depth=*/-1, /*isResult=*/true, idx, type));
  }

  // Remap the result parameter types, and input/result argument types. The size
  // of the SmallVector here has been determined by manual microoptimizations.
  SmallVector<Type, 16> newParamResultTypes, inputTypes, resultTypes;
  llvm::append_range(newParamResultTypes,
                     llvm::map_range(resultParamTypes, remapType));
  llvm::append_range(inputTypes,
                     llvm::map_range(values.getInputs(), remapType));
  llvm::append_range(resultTypes,
                     llvm::map_range(values.getResults(), remapType));

  if (metadata) {
    // Rebind input parameter references in the metadata.
    metadata = ::cast<FnMetadataAttrInterface>(
        evaluator.getReboundAttribute(metadata));
    // Tell the metadata which input parameters have been bound.
    metadata = metadata.getWithBoundParams(boundParams);
  }
  return SignatureType::get(
      FunctionType::get(values.getContext(), inputTypes, resultTypes),
      unboundParamTypes, newParamResultTypes, inputConventions, effects,
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
  return SignatureType::get(fnType, getInputParamTypes(), getResultParamTypes(),
                            getInputConventions(), getFnEffects(),
                            getMetadata());
}

bool SignatureType::isConcrete() {
  return getInputParamTypes().empty() && getResultParamTypes().empty();
}

Type SignatureType::parse(AsmParser &parser) {
  SignatureType signature;
  if (parser.parseLess() || parseSignature(parser, signature) ||
      parser.parseGreater())
    return {};
  return signature;
}

void SignatureType::print(AsmPrinter &printer) const {
  printer << '<';
  printSignature(printer, *this);
  printer << '>';
}

bool SignatureType::hasAddress(ValueInputConvention conv) {
  switch (conv) {
  case ValueInputConvention::None:
  case ValueInputConvention::OwnedInReg:
  case ValueInputConvention::BorrowedInReg:
    return false;
  case ValueInputConvention::OwnedInMem:
  case ValueInputConvention::BorrowedInMem:
  case ValueInputConvention::ByRef:
  case ValueInputConvention::ByRefResult:
  case ValueInputConvention::InitSelf:
    return true;
  }
  llvm_unreachable("invalid value input convention");
}

LogicalResult
SignatureType::verify(function_ref<InFlightDiagnostic()> emitError,
                      ArrayRef<Type> inputParamTypes,
                      ArrayRef<Type> resultParamTypes, FunctionType values,
                      ArrayRef<ValueInputConvention> inputConventions,
                      FnEffects effects, FnMetadataAttrInterface metadata) {
  // Check we have the right number of conventions.
  if (inputConventions.size() != values.getInputs().size())
    return emitError() << "incorrect # of input conventions specified";

  unsigned minNumArgs = effects.hasAnyVarArgs() + effects.hasKWVarArgs();
  if (values.getNumInputs() < minNumArgs) {
    return emitError()
           << "function has varargs and/or kwvarargs but signature only has "
           << values.getNumInputs() << " arguments";
  }

  // If the signature has metadata, defer to it for further verification.
  // Otherwise, run the standard KGEN signature verification.
  if (metadata) {
    return metadata.verifySignature(emitError, inputParamTypes,
                                    resultParamTypes, values, inputConventions,
                                    effects);
  }

  // Verify input convention and argument types.
  for (auto [i, argType, conv] :
       llvm::enumerate(values.getInputs(), inputConventions)) {
    Type type = argType;
    // Verify variadics.
    if (effects.hasVarArgs() && effects.isVarArg(values.getNumInputs(), i)) {
      auto variadic = ::dyn_cast<VariadicType>(type);
      if (!variadic) {
        return emitError() << "argument #" << i
                           << " in signature with varargs should be a "
                              "`!kgen.variadic` but got: "
                           << type;
      }
      type = variadic.getElementType();
    }
    // Verify argument conventions.  Before lit lowering, they need to be
    // !lit.ref type, after lowering, they should have !kgen.pointer type.
    if (hasAddress(conv)) {
      if (::isa<PointerType>(type))
        break;
      // TODO: During LowerLIT, we strip off the metadata, but later we lower
      // references to pointers.  This means that LowerLIT needs a
      // kgen.signature (without LIT attribute) with references.  Accept
      // !lit.ref until we can sort this out.
      if (type.getDialect().getNamespace() == "lit")
        break;

      return emitError()
             << "argument #" << i << " with convention '" << stringifyEnum(conv)
             << "' in signature type should be a `!kgen.pointer` but got: "
             << type;
    }
  }

  // If the function throws an error, make sure it has one variant result.
  size_t numResults = values.getNumResults();
  if (effects.isThrows())
    if (numResults != 1 || !::isa<VariantType>(values.getResult(0)))
      return emitError() << "a throwing function should have 1 variant result";

  if (!inputConventions.empty() &&
      inputConventions[0] == ValueInputConvention::ByRefResult) {
    if (effects.isThrows()) {
      // We already checked this above
      auto variantTy = ::cast<VariantType>(values.getResult(0));
      if (variantTy.getNumTypes() != 2) {
        return emitError() << "a throwing function with byref_result must have "
                              "a variant result of 2 types";
      }
      auto typeConst = ::dyn_cast<TypeConstantAttr>(variantTy.getTypes()[1]);
      if (!typeConst || !::isa<KGEN::NoneType>(typeConst.getValue())) {
        return emitError() << "a throwing function with byref_result must have "
                              "a variant result with none as the second type";
      }
    } else {
      if (numResults != 1 || !::isa<KGEN::NoneType>(values.getResult(0))) {
        return emitError() << "a non-throwing function with byref_result must "
                              "have 1 none result";
      }
    }
  }

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

SignatureType SignatureType::remapToSignature(
    ArrayRef<ParamDeclAttr> inputParams, ArrayRef<ParamDeclAttr> resultParams,
    FunctionType functionType, ArrayRef<ValueInputConvention> inputConventions,
    FnEffects effects, Attribute metadata,
    function_ref<InFlightDiagnostic()> emitError) {
  IndexRefRemapper remapper(inputParams, resultParams);
  SmallVector<Type> inputParamTypes, resultParamTypes;
  for (ParamDeclAttr param : inputParams)
    inputParamTypes.push_back(remapper.replace(param.getType()));
  for (ParamDeclAttr param : resultParams)
    resultParamTypes.push_back(remapper.replace(param.getType()));

  if (!emitError) {
    emitError = []() -> InFlightDiagnostic {
      llvm_unreachable("invalid signature");
    };
  }

  return SignatureType::getChecked(
      emitError, remapper.replace(functionType), inputParamTypes,
      resultParamTypes, inputConventions, effects,
      metadata ? remapper.replace(metadata) : nullptr);
}

SignatureType
SignatureType::prependParams(SignatureType sig,
                             ArrayRef<ParamDeclAttr> parentParams) {
  IndexRefRemapper remapper(parentParams, /*resultParams=*/{},
                            parentParams.size());
  SmallVector<Type> inputParamTypes;
  for (ParamDeclAttr param : parentParams)
    inputParamTypes.push_back(remapper.replace(param.getType()));
  for (Type type : sig.getInputParamTypes())
    inputParamTypes.push_back(remapper.replace(type));

  FnMetadataAttrInterface metadata = sig.getMetadata();
  if (metadata) {
    metadata = remapper.replace(
        sig.getMetadata().prependPosParams(parentParams.size()));
  }

  return SignatureType::get(remapper.replace(sig.getValues()), inputParamTypes,
                            remapper.replace(sig.getResultParamTypes()),
                            sig.getInputConventions(), sig.getFnEffects(),
                            metadata);
}

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

LogicalResult PointerType::verify(function_ref<InFlightDiagnostic()> emitError,
                                  Type type, TypedAttr addressSpace) {
  if (!addressSpace || addressSpace.getType().isIndex())
    return success();
  return emitError() << "address space parameter `" << addressSpace
                     << "` must be an index type";
}

PointerType PointerType::get(TypedAttr elementType, unsigned addressSpace) {
  return PointerType::get(ParamRefType::get(elementType), addressSpace);
}

PointerType PointerType::get(Type elementType, unsigned addressSpace) {
  Builder b(elementType.getContext());
  return get(elementType, b.getIndexAttr(addressSpace));
}

PointerType PointerType::get(Type elementType, TypedAttr addressSpace) {
  return get(elementType.getContext(), elementType, addressSpace);
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
    if (p.parseLParen() || parseParamValue(p, memValue, getElementType()) ||
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

ErrorOrSuccess DTypeType::writeTo(TypedAttr value, int64_t addr,
                                  InterpreterState &state) const {
  // DType is one byte.
  ErrorOr<void *> mem = state.getWritableMemory(addr, 1);
  if (mem.isError())
    return mem.takeError();
  *(uint8_t *)*mem = ::cast<DTypeConstantAttr>(value).getDType().getValue();
  return success();
}

ErrorOr<TypedAttr> DTypeType::readFrom(int64_t addr,
                                       InterpreterState &state) const {
  ErrorOr<const void *> mem = state.getReadableMemory(addr, 1);
  if (mem.isError())
    return mem.takeError();
  return DTypeConstantAttr::get(getContext(),
                                KGENDType(*(const uint8_t *)*mem));
}

//===----------------------------------------------------------------------===//
// NoneType
//===----------------------------------------------------------------------===//

std::optional<int64_t>
KGEN::NoneType::getTypeSize(TargetInfoAttr target) const {
  return 0;
}

std::optional<int64_t>
KGEN::NoneType::getTypeAlign(TargetInfoAttr target) const {
  return 1;
}

//===----------------------------------------------------------------------===//
// DeclRefType
//===----------------------------------------------------------------------===//

DeclRefType DeclRefType::get(SymbolRefAttr name,
                             ArrayRef<TypedAttr> paramValues, Type metatype) {
  return get(name.getContext(), name, paramValues, metatype);
}

DeclRefType DeclRefType::get(SymbolRefAttr name, Type metatype) {
  return get(name, {}, metatype);
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
// IntLiteralType
//===----------------------------------------------------------------------===//

OptionalParseResult IntLiteralType::parseValue(AsmParser &p,
                                               TypedAttr &value) const {
  APInt resultAP;
  OptionalParseResult parseResult = p.parseOptionalInteger(resultAP);
  if (!parseResult.has_value())
    return {};
  if (failed(*parseResult))
    return failure();
  value = IntLiteralAttr::get(p.getContext(), IPInt(resultAP));
  return mlir::success();
}

LogicalResult IntLiteralType::printValue(AsmPrinter &p, TypedAttr value) const {
  auto v = ::dyn_cast<IntLiteralAttr>(value);
  if (!v)
    return failure();
  p.getStream() << v.getValue();
  return success();
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

/// Helper to write a data pointer plus size type, both of which are pointer
/// width, and the pointer comes first.
static ErrorOrSuccess writePointerAndSize(int64_t writeAddr, int64_t ptr,
                                          int64_t size,
                                          InterpreterState &state) {
  unsigned ptrSize = state.getTarget().getDataLayout().getPointerSize();
  ErrorOr<void *> mem = state.getWritableMemory(writeAddr, ptrSize * 2);
  if (mem.isError())
    return mem.takeError();
  // Store the pointer address, and then advance a pointer width and store the
  // size.
  llvm::StoreIntToMemory(APInt(ptrSize * 8, ptr), (uint8_t *)*mem, ptrSize);
  llvm::StoreIntToMemory(APInt(ptrSize * 8, size), (uint8_t *)*mem + ptrSize,
                         ptrSize);
  return success();
}

/// Helper to read a data pointer and size type, both of which are pointer
/// width, and the pointer comes first.
static ErrorOr<std::pair<int64_t, int64_t>>
readPointerAndSize(int64_t readAddr, InterpreterState &state) {
  unsigned ptrSize = state.getTarget().getDataLayout().getPointerSize();
  ErrorOr<const void *> mem = state.getReadableMemory(readAddr, ptrSize * 2);
  if (mem.isError())
    return mem.takeError();
  APInt ptrVal(ptrSize * 8, 0);
  APInt sizeVal(ptrSize * 8, 0);
  llvm::LoadIntFromMemory(ptrVal, (const uint8_t *)*mem, ptrSize);
  llvm::LoadIntFromMemory(sizeVal, (const uint8_t *)*mem + ptrSize, ptrSize);
  return std::make_pair(ptrVal.getLimitedValue(), sizeVal.getLimitedValue());
}

ErrorOrSuccess StringType::writeTo(TypedAttr value, int64_t addr,
                                   InterpreterState &state) const {
  DialectResourceManager &mgr = MemoryHandle::getManagerInterface(getContext());
  // Ensure the string is null-terminated. This is safe because `StringAttr`
  // always stores a null terminator.
  auto strAttr = ::cast<StringAttr>(value);
  StringRef str(strAttr.data(), strAttr.size() + 1);
  if (strAttr.getValue().empty())
    str = "\0";
  MemoryHandle hdl = mgr.getOrAddStringResource(str);
  ErrorOr<int64_t> strAddr = state.mapConstGlobalMemory(hdl);
  if (strAddr.isError())
    return strAddr.takeError();

  // Store a pointer and a size.
  return writePointerAndSize(addr, *strAddr, strAttr.size(), state);
}

ErrorOr<TypedAttr> StringType::readFrom(int64_t addr,
                                        InterpreterState &state) const {
  // Load a pointer and size.
  ErrorOr<std::pair<int64_t, int64_t>> ptrSize =
      readPointerAndSize(addr, state);
  if (ptrSize)
    return ptrSize.takeError();
  auto [strAddr, strSize] = *ptrSize;

  // Read back the string.
  ErrorOr<const void *> strMem = state.getReadableMemory(strAddr, strSize);
  if (strMem.isError())
    return strMem.takeError();

  return StringAttr::get(StringRef((const char *)*strMem, strSize), *this);
}

//===----------------------------------------------------------------------===//
// VariadicType
//===----------------------------------------------------------------------===//

static void printVariadicConvention(AsmPrinter &p, ValueInputConvention conv) {
  // Default to borrowed_in_reg
  if (conv != ValueInputConvention::BorrowedInReg)
    p << ", " << stringifyValueInputConvention(conv);
}

static ParseResult parseVariadicConvention(AsmParser &p,
                                           ValueInputConvention &conv) {
  // Default to borrowed_in_reg
  if (!succeeded(p.parseOptionalComma())) {
    conv = ValueInputConvention::BorrowedInReg;
    return success();
  }

  StringRef name;
  llvm::SMLoc loc = p.getCurrentLocation();
  if (p.parseKeyword(&name))
    return failure();
  auto convVal = symbolizeValueInputConvention(name);
  if (!convVal.has_value()) {
    p.emitError(loc, "expected convention");
    return failure();
  }
  conv = *convVal;
  return success();
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

ErrorOrSuccess VariadicType::writeTo(TypedAttr value, int64_t addr,
                                     InterpreterState &state) const {
  // A variadic is a pointer and a size, where the pointer refers to
  // stack-allocated memory.
  ArrayRef<TypedAttr> values = ::cast<VariadicAttr>(value).getValues();
  TargetInfoAttr target = state.getTarget();

  // Query the size and alignment of the element type.
  Type elemType = getElementType();
  std::optional<int64_t> typeAlign =
      DataLayoutInterface::getTypeABIAlign(target, elemType);
  std::optional<int64_t> allocSize =
      DataLayoutInterface::getTypeAllocSize(target, elemType);
  if (!typeAlign || !allocSize)
    return Error("could not query element type size or alignment");

  // Allocate stack memory for the elements.
  ErrorOr<int64_t> valuesAddr =
      state.allocateStackMemory(*allocSize * values.size(), *typeAlign);
  if (valuesAddr.isError())
    return valuesAddr.takeError();
  int64_t baseAddr = *valuesAddr;

  // Now write all the elements to the stack memory.
  for (auto [i, value] : llvm::enumerate(values)) {
    if (ErrorOrSuccess err =
            state.writeAttributeToMemory(baseAddr + i * *allocSize, value))
      return err.takeError();
  }

  // And now write the pointer and size.
  return writePointerAndSize(addr, baseAddr, values.size(), state);
}

ErrorOr<TypedAttr> VariadicType::readFrom(int64_t addr,
                                          InterpreterState &state) const {
  // Read the pointer and size.
  ErrorOr<std::pair<int64_t, int64_t>> ptrSize =
      readPointerAndSize(addr, state);
  if (ptrSize)
    return ptrSize.takeError();
  auto [baseAddr, numElems] = *ptrSize;

  // Query the size and alignment of the element type.
  TargetInfoAttr target = state.getTarget();
  Type elemType = getElementType();
  std::optional<int64_t> allocSize =
      DataLayoutInterface::getTypeAllocSize(target, elemType);
  if (!allocSize)
    return Error("could not query element type size or alignment");

  // Now read the variadic elements off the stack.
  SmallVector<TypedAttr> values;
  for (unsigned i = 0; i != numElems; ++i) {
    ErrorOr<TypedAttr> value =
        state.readAttributeFromMemory(baseAddr + i * *allocSize, elemType);
    if (value)
      return value.takeError();
    values.push_back(value.takeValue());
  }

  return VariadicAttr::get(values, *this);
}

//===----------------------------------------------------------------------===//
// StructType
//===----------------------------------------------------------------------===//

static void printIsMemoryOnly(AsmPrinter &p, bool isMemoryOnly) {
  if (isMemoryOnly)
    p << " memoryOnly";
}

static ParseResult parseIsMemoryOnly(AsmParser &p, bool &isMemoryOnly) {
  if (succeeded(p.parseOptionalKeyword("memoryOnly")))
    isMemoryOnly = true;
  return success();
}

/// Try to narrow all the given type expressions to MLIR types.
static LogicalResult resolveTypes(ArrayRef<TypedAttr> types,
                                  SmallVectorImpl<Type> &resolvedTypes) {
  for (const TypedAttr &type : types) {
    if (auto constant = llvm::dyn_cast<TypeConstantAttr>(type))
      resolvedTypes.push_back(constant.getValue());
    else
      return failure();
  }
  return success();
}

static std::optional<int64_t> getPackedElementsTypeSize(ArrayRef<Type> types,
                                                        TargetInfoAttr target) {
  int64_t size = 0;
  int64_t strictest = 1;
  for (Type type : types) {
    std::optional<int64_t> typeAlign =
        DataLayoutInterface::getTypeABIAlign(target, type);
    std::optional<int64_t> typeSize =
        DataLayoutInterface::getTypeAllocSize(target, type);
    if (!typeAlign || !typeSize)
      return {};
    size = llvm::alignTo(size, *typeAlign) + *typeSize;
    strictest = std::max(strictest, *typeAlign);
  }
  return llvm::alignTo(size, strictest);
}

std::optional<int64_t> StructType::getTypeSize(TargetInfoAttr target) const {
  return getPackedElementsTypeSize(getElementTypes(), target);
}

static std::optional<int64_t>
getPackedElementsTypeAlign(ArrayRef<Type> types, TargetInfoAttr target) {
  int64_t strictest = 1;
  for (Type type : types) {
    std::optional<int64_t> typeAlign =
        DataLayoutInterface::getTypeABIAlign(target, type);
    if (!typeAlign)
      return {};
    strictest = std::max(strictest, *typeAlign);
  }
  return strictest;
}

std::optional<int64_t> StructType::getTypeAlign(TargetInfoAttr target) const {
  return getPackedElementsTypeAlign(getElementTypes(), target);
}

ErrorOrSuccess StructType::writeTo(TypedAttr value, int64_t addr,
                                   InterpreterState &state) const {
  int64_t offset = 0;
  for (TypedAttr value : ::cast<StructAttr>(value).getValues()) {
    auto dl = ::cast<DataLayoutInterface>(value.getType());
    // Store each element spaced apart by padding according to its alignment.
    offset = llvm::alignTo(offset, *dl.getTypeAlign(state.getTarget()));
    // Ignore unknown values. Just leave the memory as-is.
    if (!::isa<UnknownAttr>(value)) {
      ErrorOrSuccess result =
          state.writeAttributeToMemory(addr + offset, value);
      if (result.isError())
        return result.takeError();
    }
    offset += *dl.getTypeSize(state.getTarget());
  }
  return success();
}

ErrorOr<TypedAttr> StructType::readFrom(int64_t addr,
                                        InterpreterState &state) const {
  SmallVector<TypedAttr> values;
  int64_t offset = 0;
  for (Type elType : getElementTypes()) {
    auto dl = elType.cast<DataLayoutInterface>();
    offset = llvm::alignTo(offset, *dl.getTypeAlign(state.getTarget()));
    ErrorOr<TypedAttr> value =
        state.readAttributeFromMemory(addr + offset, elType);
    if (value.isError())
      return value.takeError();
    values.push_back(value.takeValue());
    offset += *dl.getTypeSize(state.getTarget());
  }
  return StructAttr::get(values, *this);
}

//===----------------------------------------------------------------------===//
// PackType
//===----------------------------------------------------------------------===//

static void printPackType(AsmPrinter &p, TypedAttr value) {
  if (auto variadic = dyn_cast<VariadicType>(value.getType())) {
    if (isa<AnyRegTypeType>(variadic.getElementType())) {
      printParamValue(p, value);
      return;
    }
  }
  printColonTypeParamValue(p, value);
}

static ParseResult parsePackType(AsmParser &p, TypedAttr &value) {
  auto anyRegTypeType = AnyRegTypeType::get(p.getContext());
  Type type =
      VariadicType::get(anyRegTypeType, ValueInputConvention::BorrowedInReg);
  if (succeeded(p.parseOptionalColon()))
    if (parseKGENType(p, type))
      return failure();
  return parseParamValue(p, value, type);
}

/// Verify that the element type of the variadic attribute or expression is a
/// type expression.
LogicalResult PackType::verify(function_ref<InFlightDiagnostic()> emitError,
                               TypedAttr variadic) {
  if (::isa<VariadicType>(variadic.getType()))
    return success();
  return emitError() << "expected an operand of variadic type, but got "
                     << variadic.getType();
}

std::optional<int64_t> PackType::getTypeSize(TargetInfoAttr target) const {
  // A pack backed by an attribute has a size equivalent to a struct composed
  // of the elements in the sequence.
  if (VariadicAttr attr = getVariadicAttr()) {
    SmallVector<Type> types;
    if (failed(resolveTypes(attr.getValues(), types)))
      return {};
    return getPackedElementsTypeSize(types, target);
  }

  // We can't know the size of a variadic expression, since we don't know how
  // many elements are in the backing sequence.
  return {};
}

std::optional<int64_t> PackType::getTypeAlign(TargetInfoAttr target) const {
  TypedAttr variadic = getVariadic();

  // A pack backed by an attribute has alignment equivalent to a struct
  // composed of the elements in the sequence.
  if (auto attr = ::dyn_cast<VariadicAttr>(variadic)) {
    SmallVector<Type> types;
    if (failed(resolveTypes(attr.getValues(), types)))
      return {};
    return getPackedElementsTypeAlign(types, target);
  }

  // A pack backed by an expression has alignment equivalent to the variadic
  // type's element type.
  auto variadicType = ::dyn_cast<VariadicType>(variadic.getType());
  if (!variadicType)
    return {};
  Type type = variadicType.getElementType();
  return DataLayoutInterface::getTypeABIAlign(target, type);
}

bool PackType::isEmpty() const {
  VariadicAttr attr = getVariadicAttr();
  return attr && attr.getValues().empty();
}

VariadicAttr PackType::getVariadicAttr() const {
  return ::dyn_cast<VariadicAttr>(getVariadic());
}

//===----------------------------------------------------------------------===//
// VariantType
//===----------------------------------------------------------------------===//

VariantType VariantType::get(ArrayRef<Type> types, Type metaType) {
  assert(!types.empty());
  SmallVector<TypedAttr> typeExprs;
  for (Type type : types)
    typeExprs.push_back(TypeConstantAttr::get(type, metaType));
  return get(types.front().getContext(), typeExprs);
}

/// Return the number of types in the variant.
size_t VariantType::getNumTypes() { return getTypes().size(); }

SmallVector<Type> VariantType::getParameterizedElementTypes() const {
  SmallVector<Type> types;
  types.reserve(getTypes().size());
  for (TypedAttr type : getTypes())
    types.push_back(ParamRefType::get(type));
  return types;
}

Type VariantType::getType(unsigned index) {
  return ParamRefType::get(getTypes()[index]);
}

/// Compute the size in bytes of just the content section of a variant. The
/// content field is the biggest element size rounded up to the nearest
/// multiple of the content element type size, which is i64.
static std::optional<int64_t> computeVariantContentSize(VariantType type,
                                                        TargetInfoAttr target) {
  int64_t maxSize = 0;
  for (TypedAttr typeExpr : type.getTypes()) {
    auto typeCst = llvm::dyn_cast<ConcreteTypeConstantAttr>(typeExpr);
    if (!typeCst)
      return {};
    std::optional<int64_t> typeSize =
        DataLayoutInterface::getTypeAllocSize(target, typeCst.getValue());
    if (!typeSize)
      return {};
    maxSize = std::max(maxSize, *typeSize);
  }
  return llvm::alignTo(maxSize, *type.getTypeAlign(target));
}

/// Get bitwidth of the integer used to represent the discriminator. The
/// discriminator field is the smallest integer type whose maximum value is
/// greater than the number of possible subtypes, but which is at least `i1`.
static int64_t getVariantDiscrSizeInBits(VariantType type) {
  return std::max(1u, llvm::Log2_32_Ceil(type.getTypes().size()));
}

/// Get the width of the integer used to represent the discriminator in bytes.
/// This returns at least 1, because the bitwidth of the discriminator is at
/// least 1.
static int64_t getVariantDiscrSize(VariantType type) {
  return llvm::divideCeil(getVariantDiscrSizeInBits(type), CHAR_BIT);
}

std::optional<int64_t> VariantType::getTypeSize(TargetInfoAttr target) const {
  // A variant is lowered to a struct that consists of a content field and a
  // discriminator field.
  std::optional<int64_t> contentSize = computeVariantContentSize(*this, target);
  if (!contentSize)
    return {};
  // Align to the content array element alignment. We don't expect the
  // discriminator to exceed it in size (at least a 32-bit integer).
  return llvm::alignTo(*contentSize + getVariantDiscrSize(*this),
                       *getTypeAlign(target));
}

std::optional<int64_t> VariantType::getTypeAlign(TargetInfoAttr target) const {
  // The alignment of the variant type is the alignment of the integer type
  // equal to the pointer width.
  // FIXME: This is incorrect but the LLVM lowering needs to be fixed.
  return target.getDataLayout().getIntegerABIAlign(
      target.getDataLayout().getPointerBitWidth());
}

ErrorOrSuccess VariantType::writeTo(TypedAttr value, int64_t addr,
                                    InterpreterState &state) const {
  // Just write the value to the address and then the discriminator.
  auto variant = ::cast<VariantAttr>(value);
  TypedAttr typeValue = variant.getValue();
  ErrorOrSuccess result = state.writeAttributeToMemory(addr, typeValue);
  if (result.isError())
    return result.takeError();
  addr += *computeVariantContentSize(*this, state.getTarget());

  unsigned discrSize = getVariantDiscrSize(*this);
  ErrorOr<void *> mem = state.getWritableMemory(addr, discrSize);
  if (mem.isError())
    return mem.takeError();
  APInt discrVal(discrSize * CHAR_BIT, variant.getIndex());
  llvm::StoreIntToMemory(discrVal, reinterpret_cast<uint8_t *>(*mem),
                         discrSize);
  return success();
}

ErrorOr<TypedAttr> VariantType::readFrom(int64_t addr,
                                         InterpreterState &state) const {
  // Read the discriminator first so we know what type to read.
  unsigned discrSize = getVariantDiscrSize(*this);
  ErrorOr<const void *> mem = state.getReadableMemory(
      addr + *computeVariantContentSize(*this, state.getTarget()), discrSize);
  if (mem.isError())
    return mem.takeError();
  APInt discrVal(discrSize * CHAR_BIT, 0);
  llvm::LoadIntFromMemory(discrVal, reinterpret_cast<const uint8_t *>(*mem),
                          discrSize);

  unsigned index = discrVal.getZExtValue();
  TypedAttr type = getTypes()[index];
  ErrorOr<TypedAttr> result = state.readAttributeFromMemory(
      addr, type.cast<ConcreteTypeConstantAttr>().getValue());
  if (result.isError())
    return result.takeError();
  return VariantAttr::get(result.takeValue(), index, *this);
}

//===----------------------------------------------------------------------===//
// CaptureListType
//===----------------------------------------------------------------------===//

/// Implementation of parsing for CaptureListType.
static ParseResult parseCaptureListType(AsmParser &p,
                                        TypedAttr &capturingFunc) {
  Type type;
  if (parseKGENType(p, type) || p.parseColon() ||
      parseParamValue(p, capturingFunc, type))
    return failure();
  if (!isa<SignatureType>(type))
    return failure();
  return mlir::success();
}

/// Implementation of printing for CaptureListType.
static void printCaptureListType(AsmPrinter &p, TypedAttr capturingFunc) {
  printKGENType(p, capturingFunc.getType());
  p << " : ";
  printParamValue(p, capturingFunc, capturingFunc.getType());
}

std::optional<int64_t>
CaptureListType::getTypeSize(TargetInfoAttr target) const {
  return target.getDataLayout().getPointerSize();
}

std::optional<int64_t>
CaptureListType::getTypeAlign(TargetInfoAttr target) const {
  return target.getDataLayout().getPointerABIAlign();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

// Pull in the dialect definition.
#define GET_TYPEDEF_CLASSES
#include "KGEN/KGENDialect/KGENTypes.cpp.inc"
