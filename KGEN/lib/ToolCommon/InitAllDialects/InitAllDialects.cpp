//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ToolCommon/InitAllDialects.h"

#include "KGEN/CODialect/CODialect.h"
#include "KGEN/CustomDialect/CustomDialect.h"
#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/Interpreter/InterpreterDialect.h"
#include "KGEN/Interpreter/InterpreterInterface.h"
#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/KGENDialect/KGENInterfaces.h"
#include "KGEN/LITDialect/LITDialect.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/IR/DialectRegistry.h"

using namespace M;
using namespace KGEN;

namespace {

//===----------------------------------------------------------------------===//
// InterpreterDialectExtension
//===----------------------------------------------------------------------===//

template <typename Concrete>
struct DebugOpInterpreterInterface
    : public InterpreterOpInterface::ExternalModel<
          DebugOpInterpreterInterface<Concrete>, Concrete> {
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
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InterpreterDialectExtension)

  explicit InterpreterDialectExtension()
      : DialectExtensionBase(
            DebugInfo::DebugInfoDialect::getDialectNamespace()) {}

  /// Apply the extension by injecting the operation interfaces.
  void apply(MLIRContext *ctx,
             MutableArrayRef<Dialect *> dialects) const override {
    DebugInfo::ValueOp::attachInterface<
        DebugOpInterpreterInterface<DebugInfo::ValueOp>>(*ctx);
    DebugInfo::KillOp::attachInterface<
        DebugOpInterpreterInterface<DebugInfo::KillOp>>(*ctx);
    DebugInfo::LineTableLocOp::attachInterface<
        DebugOpInterpreterInterface<DebugInfo::LineTableLocOp>>(*ctx);
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
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ParameterPrettyFormatExtension)

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

//===----------------------------------------------------------------------===//
// IndexDialectExtension
//===----------------------------------------------------------------------===//

/// Implement ComputeOpInterface hooks for index dialect.
struct AddOpComputeOpInterface
    : public ComputeOpInterface::ExternalModel<AddOpComputeOpInterface,
                                               mlir::index::AddOp> {
  ComputeKind getComputeKind(Operation *op) const {
    return ComputeKind::Addition;
  }
};

struct SubOpComputeOpInterface
    : public ComputeOpInterface::ExternalModel<SubOpComputeOpInterface,
                                               mlir::index::SubOp> {
  ComputeKind getComputeKind(Operation *op) const {
    return ComputeKind::Addition;
  }
};

struct MulOpComputeOpInterface
    : public ComputeOpInterface::ExternalModel<MulOpComputeOpInterface,
                                               mlir::index::MulOp> {
  ComputeKind getComputeKind(Operation *op) const {
    return ComputeKind::Multiplication;
  }
};

struct CmpOpComputeOpInterface
    : public ComputeOpInterface::ExternalModel<CmpOpComputeOpInterface,
                                               mlir::index::CmpOp> {
  ComputeKind getComputeKind(Operation *op) const {
    return ComputeKind::Comparison;
  }
};
struct DivSOpComputeOpInterface
    : public ComputeOpInterface::ExternalModel<DivSOpComputeOpInterface,
                                               mlir::index::DivSOp> {
  ComputeKind getComputeKind(Operation *op) const {
    return ComputeKind::Division;
  }
};

struct DivUOpComputeOpInterface
    : public ComputeOpInterface::ExternalModel<DivUOpComputeOpInterface,
                                               mlir::index::DivUOp> {
  ComputeKind getComputeKind(Operation *op) const {
    return ComputeKind::Division;
  }
};

struct CeilDivSOpComputeOpInterface
    : public ComputeOpInterface::ExternalModel<CeilDivSOpComputeOpInterface,
                                               mlir::index::CeilDivSOp> {
  ComputeKind getComputeKind(Operation *op) const {
    return ComputeKind::Division;
  }
};

struct CeilDivUOpComputeOpInterface
    : public ComputeOpInterface::ExternalModel<CeilDivUOpComputeOpInterface,
                                               mlir::index::CeilDivUOp> {
  ComputeKind getComputeKind(Operation *op) const {
    return ComputeKind::Division;
  }
};

struct FloorDivSOpComputeOpInterface
    : public ComputeOpInterface::ExternalModel<FloorDivSOpComputeOpInterface,
                                               mlir::index::FloorDivSOp> {
  ComputeKind getComputeKind(Operation *op) const {
    return ComputeKind::Division;
  }
};

struct RemSOpComputeOpInterface
    : public ComputeOpInterface::ExternalModel<RemSOpComputeOpInterface,
                                               mlir::index::RemSOp> {
  ComputeKind getComputeKind(Operation *op) const {
    return ComputeKind::Division;
  }
};

struct RemUOpComputeOpInterface
    : public ComputeOpInterface::ExternalModel<RemUOpComputeOpInterface,
                                               mlir::index::RemUOp> {
  ComputeKind getComputeKind(Operation *op) const {
    return ComputeKind::Division;
  }
};

// Extend KGEN dialect to inject ComputeOpInterface hooks into index
// dialect.
class KGENDialectExtension : public mlir::DialectExtensionBase {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(KGENDialectExtension)

  explicit KGENDialectExtension()
      : DialectExtensionBase(mlir::index::IndexDialect::getDialectNamespace()) {
  }

  /// Apply the extension by injecting the operation interfaces.
  void apply(MLIRContext *ctx,
             MutableArrayRef<Dialect *> dialects) const override {
    mlir::index::AddOp::attachInterface<AddOpComputeOpInterface>(*ctx);
    mlir::index::SubOp::attachInterface<SubOpComputeOpInterface>(*ctx);
    mlir::index::MulOp::attachInterface<MulOpComputeOpInterface>(*ctx);
    mlir::index::CmpOp::attachInterface<CmpOpComputeOpInterface>(*ctx);
    mlir::index::DivSOp::attachInterface<DivSOpComputeOpInterface>(*ctx);
    mlir::index::DivUOp::attachInterface<DivUOpComputeOpInterface>(*ctx);
    mlir::index::CeilDivSOp::attachInterface<CeilDivSOpComputeOpInterface>(
        *ctx);
    mlir::index::CeilDivUOp::attachInterface<CeilDivUOpComputeOpInterface>(
        *ctx);
    mlir::index::FloorDivSOp::attachInterface<FloorDivSOpComputeOpInterface>(
        *ctx);
    mlir::index::RemSOp::attachInterface<RemSOpComputeOpInterface>(*ctx);
    mlir::index::RemUOp::attachInterface<RemUOpComputeOpInterface>(*ctx);
  }

  /// Clone the extension.
  std::unique_ptr<mlir::DialectExtensionBase> clone() const override {
    return std::make_unique<KGENDialectExtension>();
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
      CO::CODialect,
      Custom::CustomDialect,
      MDialect,
      DebugInfo::DebugInfoDialect,
      mlir::index::IndexDialect,
      mlir::LLVM::LLVMDialect,
      mlir::NVVM::NVVMDialect
      // clang-format on
      >();

  registry
      .addExtensions<InterpreterDialectExtension,
                     ParameterPrettyFormatExtension, KGENDialectExtension>();
  mlir::LLVM::registerInlinerInterface(registry);
}

void M::preloadAllKGENDialects(MLIRContext *ctx) {
  ctx->loadDialect<
      // clang-format off
      InterpreterDialect,
      HLCF::HLCFDialect,
      KGENDialect,
      LIT::LITDialect,
      POP::POPDialect,
      Custom::CustomDialect,
      MDialect,
      DebugInfo::DebugInfoDialect,
      mlir::index::IndexDialect,
      mlir::LLVM::LLVMDialect,
      mlir::NVVM::NVVMDialect
      // clang-format on
      >();
}
