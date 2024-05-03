//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CODialect/CODialect.h"
#include "KGEN/CODialect/COTypes.h"
#include "Support/Compiler/Bytecode.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
using namespace CO;

//===----------------------------------------------------------------------===//
// CODialectBytecodeInterface
//===----------------------------------------------------------------------===//

namespace {
using mlir::DialectBytecodeReader;
using mlir::DialectBytecodeWriter;
using mlir::get;

#include "KGEN/CODialect/CODialectBytecode.cpp.inc"

struct CODialectBytecodeInterface : public mlir::BytecodeDialectInterface {
  CODialectBytecodeInterface(Dialect *dialect)
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
// CODialect
//===----------------------------------------------------------------------===//

void CODialect::initialize() {
  registerTypes();
  registerOperations();
  addInterface<CODialectBytecodeInterface>();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "KGEN/CODialect/CODialect.cpp.inc"
