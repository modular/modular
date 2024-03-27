//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ToolCommon/InitAllDialects.h"

#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/Interpreter/InterpreterDialect.h"
#include "KGEN/Interpreter/InterpreterInterface.h"
#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/LITDialect/LITDialect.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/DialectRegistry.h"

using namespace M;
using namespace KGEN;

namespace {

//===----------------------------------------------------------------------===//
// InterpreterDialectExtension
//===----------------------------------------------------------------------===//

struct ValueOpInterpreterInterface
    : public InterpreterOpInterface::ExternalModel<ValueOpInterpreterInterface,
                                                   DebugInfo::ValueOp> {
  /// Implement the interpret hook for this operation. Since the operation has
  /// no results, we cannot use the fold hook.
  ErrorTreeOrSuccess interpret(Operation *op, ArrayRef<Attribute> operands,
                               InterpreterState &state) const {
    return success();
  }
};

struct KillOpInterpreterInterface
    : public InterpreterOpInterface::ExternalModel<ValueOpInterpreterInterface,
                                                   DebugInfo::KillOp> {
  /// Implement the interpret hook for this operation. Since the operation has
  /// no results, we cannot use the fold hook.
  ErrorTreeOrSuccess interpret(Operation *op, ArrayRef<Attribute> operands,
                               InterpreterState &state) const {
    return success();
  }
};

/// This dialect extension implements interpreter hooks for non-KGEN dialects.
class InterpreterDialectExtension : public mlir::DialectExtensionBase {
public:
  explicit InterpreterDialectExtension()
      : DialectExtensionBase(
            DebugInfo::DebugInfoDialect::getDialectNamespace()) {}

  /// Apply the extension by injecting the operation interfaces.
  void apply(MLIRContext *ctx,
             MutableArrayRef<Dialect *> dialects) const override {
    DebugInfo::ValueOp::attachInterface<ValueOpInterpreterInterface>(*ctx);
    DebugInfo::KillOp::attachInterface<KillOpInterpreterInterface>(*ctx);
  }

  /// Clone the extension.
  std::unique_ptr<mlir::DialectExtensionBase> clone() const override {
    return std::make_unique<InterpreterDialectExtension>();
  }
};

//===----------------------------------------------------------------------===//
// ParameterPrettyFormatExtension
//===----------------------------------------------------------------------===//

/// This dialect extension ensures dialects that plug into KGEN's pretty
/// parameter printing are loaded.
class ParameterPrettyFormatExtension : public mlir::DialectExtensionBase {
public:
  /// Apply this extension once the KGEN dialect is loaded.
  explicit ParameterPrettyFormatExtension()
      : DialectExtensionBase(KGENDialect::getDialectNamespace()) {}

  /// Apply the extension by loading all other dialects with pretty printing.
  void apply(MLIRContext *ctx,
             MutableArrayRef<Dialect *> dialects) const override {
    ctx->loadDialect<POP::POPDialect>();
    ctx->loadDialect<LIT::LITDialect>();
  }

  /// Clone the extension.
  std::unique_ptr<mlir::DialectExtensionBase> clone() const override {
    return std::make_unique<ParameterPrettyFormatExtension>();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// registerAllKGENDialects
//===----------------------------------------------------------------------===//

void M::registerAllKGENDialects(mlir::DialectRegistry &registry) {
  registry.insert<
      // clang-format off
      InterpreterDialect,
      HLCF::HLCFDialect,
      KGENDialect,
      LIT::LITDialect,
      POP::POPDialect,
      MDialect,
      DebugInfo::DebugInfoDialect,
      mlir::index::IndexDialect,
      mlir::LLVM::LLVMDialect,
      mlir::NVVM::NVVMDialect
      // clang-format on
      >();

  registry.addExtensions<InterpreterDialectExtension,
                         ParameterPrettyFormatExtension>();
}

void M::preloadAllKGENDialects(MLIRContext *ctx) {
  ctx->loadDialect<
      // clang-format off
      InterpreterDialect,
      HLCF::HLCFDialect,
      KGENDialect,
      LIT::LITDialect,
      POP::POPDialect,
      MDialect,
      DebugInfo::DebugInfoDialect,
      mlir::index::IndexDialect,
      mlir::LLVM::LLVMDialect,
      mlir::NVVM::NVVMDialect
      // clang-format on
      >();
}
