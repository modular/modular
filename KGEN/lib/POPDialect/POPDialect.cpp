//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the POP dialect.
//
//===----------------------------------------------------------------------===//

#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/Compiler/Bytecode.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/ML/DType.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
using namespace POP;

//===----------------------------------------------------------------------===//
// POPDialectBytecodeInterface
//===----------------------------------------------------------------------===//

namespace {
using mlir::DialectBytecodeReader;
using mlir::DialectBytecodeWriter;
using mlir::get;
using POP::ArrayAttr;
using POP::ArrayType;

using OptionalIPRational = std::optional<IPRational>;

static LogicalResult readIPInt(DialectBytecodeReader &reader, IPInt &value) {
  uint64_t width;
  if (failed(reader.readVarInt(width)))
    return failure();
  FailureOr<APInt> result = reader.readAPIntWithKnownWidth(width);
  if (failed(result))
    return failure();
  value = IPInt(std::move(*result));
  return success();
}

static LogicalResult readOptionalIPInt(DialectBytecodeReader &reader,
                                       std::optional<IPInt> &value) {
  bool hasValue;
  uint64_t width;
  if (failed(reader.readVarIntWithFlag(width, hasValue)))
    return failure();
  if (!hasValue)
    return success();

  FailureOr<APInt> result = reader.readAPIntWithKnownWidth(width);
  if (failed(result))
    return failure();
  value = IPInt(std::move(*result));
  return success();
}

static LogicalResult readOptionalIPRational(DialectBytecodeReader &reader,
                                            std::optional<IPRational> &value) {
  std::optional<IPInt> numerator;
  if (failed(readOptionalIPInt(reader, numerator)))
    return failure();
  if (!numerator)
    return success();

  IPInt denominator;
  if (failed(readIPInt(reader, denominator)))
    return failure();
  value = IPRational(*numerator, denominator);
  return success();
}

static void writeOptionalIPInt(DialectBytecodeWriter &writer,
                               const std::optional<IPInt> &value) {
  if (!value)
    return writer.writeVarIntWithFlag(0, /*flag=*/false);

  uint64_t width = value->getAPInt().getSignificantBits();
  writer.writeVarIntWithFlag(width, /*flag=*/true);
  writer.writeAPIntWithKnownWidth(value->getAPInt().trunc(width));
}

static void writeIPInt(DialectBytecodeWriter &writer, const IPInt &value) {
  uint64_t width = value.getAPInt().getSignificantBits();
  writer.writeVarInt(width);
  writer.writeAPIntWithKnownWidth(value.getAPInt().trunc(width));
}

static void writeOptionalIPRational(DialectBytecodeWriter &writer,
                                    const std::optional<IPRational> &value) {
  if (!value)
    return writeOptionalIPInt(writer, std::nullopt);
  writeOptionalIPInt(writer, value->getNumerator());
  writeIPInt(writer, value->getDenominator());
}

static LogicalResult readDTypeValues(DialectBytecodeReader &reader,
                                     SmallVectorImpl<DTypeValue> &values) {
  uint64_t size;
  if (failed(reader.readVarInt(size)))
    return failure();
  values.reserve(size);
  for (unsigned i = 0; i < size; ++i) {
    uint64_t kind, width;
    if (failed(reader.readVarInt(kind)) || failed(reader.readVarInt(width)))
      return failure();
    FailureOr<APInt> value = reader.readAPIntWithKnownWidth(width);
    if (failed(value))
      return failure();
    values.emplace_back(std::move(*value), static_cast<KGENDType>(kind));
  }
  return success();
}

static void writeDTypeValues(DialectBytecodeWriter &writer,
                             ArrayRef<DTypeValue> values) {
  writer.writeVarInt(values.size());
  for (const DTypeValue &value : values) {
    writer.writeVarInt(value.getDType().getValue());
    writer.writeVarInt(value.getData().getBitWidth());
    writer.writeAPIntWithKnownWidth(value.getData());
  }
}

#include "KGEN/POPDialect/POPDialectBytecode.cpp.inc"

struct POPDialectBytecodeInterface : public mlir::BytecodeDialectInterface {
  POPDialectBytecodeInterface(Dialect *dialect)
      : BytecodeDialectInterface(dialect) {}

  Attribute readAttribute(DialectBytecodeReader &reader) const override {
    return ::readAttribute(getContext(), reader);
  }

  LogicalResult writeAttribute(Attribute attr,
                               DialectBytecodeWriter &writer) const override {
    return ::writeAttribute(attr, writer);
  }

  Type readType(DialectBytecodeReader &reader) const override {
    return ::readType(getContext(), reader);
  }

  LogicalResult writeType(Type type,
                          DialectBytecodeWriter &writer) const override {
    return ::writeType(type, writer);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

// Pull in the dialect definition.
#include "KGEN/POPDialect/POPDialect.cpp.inc"

// Register operations.
void POPDialect::initialize() {
  registerAttributes();
  registerTypes();

  addOperations<
#define GET_OP_LIST
#include "KGEN/POPDialect/POP.cpp.inc"
      >();

  addInterface<POPDialectBytecodeInterface>();
}
