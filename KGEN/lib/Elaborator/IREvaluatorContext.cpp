//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "IREvaluatorContext.h"
#include "KGEN/Interpreter/InterpreterState.h"

using namespace M;
using namespace KGEN;

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
FailureOr<StringAttr> IREvaluatorContext::evaluateStringPart(TypedAttr part) {
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
FailureOr<TypedAttr>
IREvaluatorContext::evaluateDataToStr(ParamOperatorAttr op) {
  FailureOr<StringAttr> result = evaluateStringPart(op.getOperand(0));
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
    FailureOr<StringAttr> extraResult = evaluateStringPart(extra);
    if (failed(extraResult))
      return failure();
    concatStr += extraResult->str();
  }
  return TypedAttr(StringAttr::get(concatStr, StringType::get(mlirCtx)));
}
