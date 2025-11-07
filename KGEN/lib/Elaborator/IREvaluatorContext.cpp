//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "IREvaluatorContext.h"
#include "KGEN/Interpreter/InterpreterState.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/TransformUtils/ManglingUtils.h"
#include "Support/StringExtras.h"
#include "llvm/ADT/ScopeExit.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// ImplNodeBase
//===----------------------------------------------------------------------===//

void ImplNodeBase::initialize(InstantiatedOpInterface inst,
                              ParameterUseDefGraph &&graph) {
  this->inst = inst;
  this->paramGraph = std::move(graph);
}

//===----------------------------------------------------------------------===//
// ParamNodeBase
//===----------------------------------------------------------------------===//

void ParamNodeBase::emplace() {
  if (done.exchange(DoneState::DONE) == DoneState::NOT_DONE)
    paramCh.copy().emplace();
}

AsyncValueRef<Chain> ParamNodeBase::copy() const { return paramCh.copy(); }

void ParamNodeBase::setToError() {
  if (done.exchange(DoneState::ERROR) == DoneState::NOT_DONE)
    paramCh.copy().emplace();
}

StringAttr ParamNodeBase::getMangledName() {
  // Check cached result.
  if (const void *namePtr = mangledName.load())
    return StringAttr::getFromOpaquePointer(namePtr);

  // Bind all parameter values in this scope.
  ArrayRef<TypedAttr> inputParamValues = inputParams.getValue();
  [[maybe_unused]] ArrayRef<ParamDeclAttr> inputParamDecls =
      gen.getInputParams();
  assert(inputParamValues.size() == inputParamDecls.size() &&
         "incorrect # input parameter values");
  std::string baseName = mangleParameterValues(gen, inputParamValues,
                                               [](StringRef) { return ""; });
  StringAttr name = StringAttr::get(gen->getContext(), baseName);

  const void *existing = nullptr;
  if (mangledName.compare_exchange_strong(existing, name.getAsOpaquePointer()))
    return name;
  return StringAttr::getFromOpaquePointer(existing);
}

/// Print a single SIMD value.
static void printSIMDValue(raw_ostream &os, const POP::DTypeValue &value,
                           KGENDType dtype) {
  if (dtype.isInt()) {
    os << value.getIntVal();
  } else if (dtype.isFloat()) {
    SmallString<256> strVal;
    value.getFloatVal().toString(strVal);
    os << StringRef(strVal.data(), strVal.size());
  } else if (dtype.isBool()) {
    os << (value.getBoolVal() ? "True" : "False");
  } else {
    assert(dtype.isIndex() || dtype.isAddress());
    os << value.getIndexVal();
  }
}

/// Print a KGENDType, following the naming scheme in the Mojo DType struct.
/// NOTE: It would be better to have custom type name printing that can be
/// implemented on the struct directly.
static void printDType(raw_ostream &os, KGENDType dtype, bool qualified) {
  if (qualified)
    os << "stdlib.builtin.dtype.";
  os << "DType." << dtype.getAsString(/*libForm=*/true);
}

void IREvaluatorContext::printParamValue(raw_ostream &os, ParamDeclAttr decl,
                                         TypedAttr value,
                                         bool qualifiedBuiltins) {
  TypeSwitch<TypedAttr>(value)
      .Case<DTypeConstantAttr>([&](auto dtypeConstant) {
        printDType(os, dtypeConstant.getDType(), qualifiedBuiltins);
      })
      .Case<IntegerAttr>([&](auto intAttr) {
        // Print booleans nicely.
        if (intAttr.getType().isSignlessInteger(1))
          os << (intAttr.getValue().isZero() ? "False" : "True");
        else
          intAttr.print(os, /*elideType=*/true);
      })
      .Case<NoneAttr>([&](auto noneAttr) { os << "None"; })
      .Case<UnboundAttr>([&](auto unboundAttr) { os << "?"; })
      .Case<TypeParamAttr>([&](auto typeAttr) {
        if (auto typeValue = dyn_cast<TypeValueType>(typeAttr.getTypeValue())) {
          auto instanceRef =
              cast<TypeInstanceRefAttr>(typeValue.getTypeValue());
          os << stringifyTypeInstanceRef(instanceRef, qualifiedBuiltins);
          return;
        }

        // We print a placeholder for anything we don't know how to print.
        // NOTE: We could consider just printing the mlir for anything else. For
        // now, this is a more conservative approach, since it prevents leaking
        // IR details.
        os << "<unprintable>";
      })
      .Case<StructAttr>([&](auto structAttr) {
        os << "{";
        llvm::interleaveComma(structAttr.getValues(), os, [&](TypedAttr value) {
          printParamValue(os, decl, value, qualifiedBuiltins);
        });
        os << "}";
      })
      .Case<MemRefAttr>([&](auto memRefAttr) {
        MemoryBlobAttr memory =
            memRefAttr.getModel().getMemory()[memRefAttr.getIndex()];
        if (MemoryHandleAttr handle = memory.getHandle(); handle.isString()) {
          // NOTE: these strings should be null terminated, but let's be safe.
          os << '"' << StringRef(handle.getData(), handle.getSize() - 1) << '"';
          return;
        }

        os << "<unprintable>";
      })
      .Case<POP::SIMDAttr>([&](auto simdAttr) {
        ArrayRef<POP::DTypeValue> values = simdAttr.getValues();
        KGENDType dType = *simdAttr.getType().getResolvedDType();
        if (values.size() == 1) {
          // We handle scalars specially for improved readability.
          printSIMDValue(os, values[0], dType);
        } else {
          os << "[";
          llvm::interleaveComma(values, os, [&](const POP::DTypeValue &value) {
            printSIMDValue(os, value, dType);
          });
          os << "]";
        }
        os << " : ";
        if (qualifiedBuiltins)
          os << "stdlib.builtin.simd.";
        os << "SIMD[";
        printDType(os, dType, qualifiedBuiltins);
        os << ", " << values.size() << "]";
      })
      .Default([&](auto value) { os << "<unprintable>"; });
}

std::string
IREvaluatorContext::stringifyTypeInstanceRef(TypeInstanceRefAttr instanceRef,
                                             bool qualifiedBuiltins) {
  ParamNodeBase *genNode = lookupParamNodeBase(instanceRef.getSymbol());
  StructGeneratorOp genOp = cast<StructGeneratorOp>(genNode->gen);

  // Print the type name first. A few common types can be printed more tersely.
  /// NOTE: It would be better to have custom type name printing that can be
  /// implemented on the struct directly.
  std::string name = genOp.getSymName().str();
  if (!qualifiedBuiltins && name.starts_with("stdlib::")) {
    if (name == "stdlib::builtin::simd::SIMD")
      name = "SIMD";
    else if (name == "stdlib::builtin::int::Int")
      name = "Int";
    else if (name == "stdlib::builtin::uint::UInt")
      name = "UInt";
    else if (name == "stdlib::builtin::bool::Bool")
      name = "Bool";
    else if (name == "stdlib::collections::string::string::String")
      name = "String";
  }
  replaceAll(name, "::", ".");

  ArrayRef<TypedAttr> paramValues = genNode->inputParams.getValue();
  if (!paramValues.empty()) {
    std::string paramValuesStr;
    llvm::raw_string_ostream os(paramValuesStr);
    auto paramDecls = genOp.getInputParams();

    // If the type is parameterized, print the parameter values.
    llvm::interleaveComma(llvm::zip(paramDecls, paramValues), os,
                          [&](auto pair) {
                            auto [decl, value] = pair;
                            printParamValue(os, decl, value, qualifiedBuiltins);
                          });

    name += "[" + paramValuesStr + "]";
  }
  return name;
}

//===----------------------------------------------------------------------===//
// IREvaluatorContext
//===----------------------------------------------------------------------===//

IREvaluatorContext::IREvaluatorContext(EnvAttr env, MLIRContext *mlirCtx,
                                       InterpreterState *state)
    : env(env), state(state), mlirCtx(mlirCtx) {}

FailureOr<TypedAttr> IREvaluatorContext::evaluateGetEnv(ParamOperatorAttr op) {
  // Grab the module from the elaborator. This is a read operation of memory
  // that is not modified during elaboration, so no synchronization is needed.
  auto name = dyn_cast<StringAttr>(op.getOperands().front());
  if (!name) {
    emitError({*errorLoc, "'get_env' name did not narrow to a constant"});
    return failure();
  }

  // Get the `StringRef` out of the `StringAttr` because the latter comes with
  // a `StringType` type that makes pointer comparisons fails.
  ErrorOr<TypedAttr> result = env.queryValue(name.getValue(), op.getType());

  if (result.isError()) {
    emitError({*errorLoc, result.getError()});
    return failure();
  }

  return result.get();
}

// See if we can decode the first 'numBytes' of the memory blob into a
// StringAttr.
static StringAttr getBytesOf(MemoryBlobAttr value, size_t numBytes) {
  // We don't bother handling these.
  if (!value.getPointerRegions().empty() || !value.getSymbolRegions().empty())
    return {};

  if (numBytes <= value.getHandle().getSize()) {
    return StringAttr::get(StringRef(value.getHandle().getData(), numBytes),
                           StringType::get(value.getContext()));
  }
  return {};
}

/// Extract a value of type `struct<(pointer<none>, index)>` into a StringAttr.
FailureOr<StringAttr> IREvaluatorContext::evaluateStringPart(TypedAttr part,
                                                             bool reset) {
  // Get the two parts of the struct, StructExtract will fold.
  auto lengthAttr = dyn_cast<IntegerAttr>(StructExtractAttr::get(part, 1));
  if (!lengthAttr) {
    emitError({*errorLoc, "'data_to_str' length didn't resolve to a constant"});
    return failure();
  }
  size_t numBytes = lengthAttr.getInt();
  if (!numBytes)
    return {StringAttr::get("", StringType::get(mlirCtx))};

  MemRefAttr pointerAttr =
      dyn_cast<MemRefAttr>(StructExtractAttr::get(part, 0));
  if (!pointerAttr) {
    emitError({*errorLoc, "'data_to_str' did not narrow to a constant"});
    return failure();
  }

  // Check to see if we have a memref(interp.memory_handle(...)) because
  // we can just immediately fold it in common cases without materializing the
  // memory.
  // We don't handle index/offset yet.
  if (auto result =
          getBytesOf(pointerAttr.getModel().getMemory()[pointerAttr.getIndex()],
                     numBytes)) {
    if (pointerAttr.getOffset() == 0)
      return result;
  }

  // Reset memory upon exit.
  auto resetState = llvm::make_scope_exit([&] {
    if (reset)
      state->reset();
  });

  if (ErrorOrSuccess err = state->internalizeMemory(pointerAttr)) {
    emitError({*errorLoc, "'data_to_str' failed to read data"});
    return failure();
  }

  size_t address = cast<PointerAttr>(pointerAttr).getAddr();
  Type byteType = IntegerType::get(mlirCtx, 8);

  // Read each of the bytes into 'result' one at a time.  If any fail,
  // just bail out.
  std::string result;
  while (numBytes) {
    ErrorOr<TypedAttr> attrOr =
        state->readAttributeFromMemory(address, byteType);
    if (attrOr.isError() || !isa<IntegerAttr>(attrOr.get())) {
      emitError({*errorLoc, "'data_to_str' failed to read data"});
      return failure();
    }
    result.push_back((char)cast<IntegerAttr>(attrOr.get()).getInt());
    ++address;
    --numBytes;
  }

  // Success!
  return {StringAttr::get(result, StringType::get(mlirCtx))};
}

/// Evaluate POC::DataToStr "data_to_str" operator.
FailureOr<TypedAttr> IREvaluatorContext::evaluateDataToStr(ParamOperatorAttr op,
                                                           bool reset) {
  FailureOr<StringAttr> result = evaluateStringPart(op.getOperand(0), reset);
  if (failed(result))
    return failure();

  // Extra string parts, which will be a VariadicAttr of type
  // !kgen.variadic<>
  VariadicAttr extrasAttr = dyn_cast<VariadicAttr>(op.getOperand(1));
  if (!extrasAttr) {
    emitError(
        {*errorLoc, "'data_to_str' did not narrow to a variadic constant"});
    return failure();
  }

  // If there are no extra parts then we're done.
  if (extrasAttr.getValues().empty())
    return TypedAttr(*result);

  // Otherwise, we need to evaluate the extra parts and concatenate them.
  std::string concatStr = result->str();
  for (TypedAttr extra : extrasAttr.getValues()) {
    FailureOr<StringAttr> extraResult = evaluateStringPart(extra, reset);
    if (failed(extraResult))
      return failure();
    concatStr += extraResult->str();
  }
  return TypedAttr(StringAttr::get(concatStr, StringType::get(mlirCtx)));
}
