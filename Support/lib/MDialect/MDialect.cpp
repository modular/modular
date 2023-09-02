//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/MDialect/MDialect.h"
#include "Support/AlignedAlloc.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;

//===----------------------------------------------------------------------===//
// MDialectBytecodeInterface
//===----------------------------------------------------------------------===//

using mlir::DialectBytecodeReader;
using mlir::DialectBytecodeWriter;
using mlir::get;

static LogicalResult parseTriple(DialectBytecodeReader &reader,
                                 llvm::Triple &triple) {
  StringRef tripleStr;
  if (failed(reader.readString(tripleStr)))
    return failure();
  triple = llvm::Triple(tripleStr);
  return success();
}

static void printTriple(MLIRContext *ctx, DialectBytecodeWriter &writer,
                        const llvm::Triple &triple) {
  writer.writeOwnedString(StringAttr::get(ctx, triple.str()));
}

static LogicalResult parseDataLayout(DialectBytecodeReader &reader,
                                     DataLayout &dl) {
  StringRef dlStr;
  if (failed(reader.readString(dlStr)))
    return failure();
  ErrorOr<DataLayout> dlOr = DataLayout::parse(dlStr);
  if (dlOr.isError())
    return reader.emitError(dlOr.getError());
  dl = dlOr.takeValue();
  return success();
}

static void printDataLayout(MLIRContext *ctx, DialectBytecodeWriter &writer,
                            const DataLayout &dl) {
  writer.writeOwnedString(dl.toString());
}

namespace {
#include "Support/MDialect/MDialectBytecode.cpp.inc"

struct MDialectBytecodeInterface : public mlir::BytecodeDialectInterface {
  MDialectBytecodeInterface(Dialect *dialect)
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
// MDialect
//===----------------------------------------------------------------------===//

void MDialect::initialize() {
  registerAttributes();
  registerTypes();
  injectTypeInterfaces();
  injectAttrInterfaces();

  addInterface<MDialectBytecodeInterface>();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "Support/MDialect/MDialect.cpp.inc"
