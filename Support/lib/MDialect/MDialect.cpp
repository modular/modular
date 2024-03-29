//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/MDialect/MDialect.h"
#include "LLCL/CompilerSupport/LLVMThreadPool.h"
#include "LLCL/Runtime/Runtime.h"
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

static LogicalResult readDataLayout(DialectBytecodeReader &reader,
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

static void writeDataLayout(DialectBytecodeWriter &writer,
                            const DataLayout &dl) {
  writer.writeOwnedString(dl.toString());
}

static LogicalResult parseRelocModel(DialectBytecodeReader &reader,
                                     llvm::Reloc::Model &model) {
  StringRef modelStr;
  if (failed(reader.readString(modelStr)))
    return failure();

  std::optional<llvm::Reloc::Model> result = symbolizeRelocationModel(modelStr);
  if (!result)
    return failure();

  model = *result;
  return success();
}

static void printRelocModel(MLIRContext *ctx, DialectBytecodeWriter &writer,
                            llvm::Reloc::Model model) {
  writer.writeOwnedString(
      StringAttr::get(ctx, stringifyRelocationModel(model)));
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

void M::registerContext(mlir::DialectRegistry &registry, ContextRef &ref) {
  std::function<void(MLIRContext * ctx, MDialect * dialect)> fn =
      [ref = ref.copy()](MLIRContext *ctx, MDialect *dialect) {
        dialect->setInternal(ref.copy());
      };
  registry.addExtension<MDialect>(std::move(fn));
}

void M::registerContext(mlir::MLIRContext &ctx, ContextRef &ref) {
  // In any execution setting where an LLCL runtime may be available, do not
  // allow MLIR contexts to have their own threading enabled -- it must go
  // through LLCL.
  if (LLVM_UNLIKELY(ctx.isMultithreadingEnabled())) {
    llvm::report_fatal_error(
        "default MLIR threading must be disabled; please construct the "
        "MLIRContext with MLIRContext::Threading::DISABLED");
  }

  DialectRegistry registry;
  registerContext(registry, ref);
  ctx.appendDialectRegistry(registry);

  // This function should be called once per MLIR context, but may be called
  // multiple times per Modular context. Guard against that by checking for an
  // existing thread pool.
  LLCL::LLVMThreadPool *tp = ref->get<LLCL::LLVMThreadPool>();
  if (!tp) {
    if (LLCL::Runtime *runtime = ref->get<LLCL::Runtime>())
      tp = &ref->emplace<LLCL::LLVMThreadPool>(*runtime);
  }
  // If the runtime is available, enable threading in MLIR with it.
  if (tp)
    ctx.setThreadPool(*tp);
}

ContextRef M::loadContext(mlir::MLIRContext *ctx) {
  StringRef name = MDialect::getDialectNamespace();
  return static_cast<MDialect *>(ctx->getOrLoadDialect(name))->getInteral();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "Support/MDialect/MDialect.cpp.inc"
