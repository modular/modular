//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Context.h"
#include "MLRT/AsyncRT/CompilerSupport/Context.h"
#include "MLRT/AsyncRT/CompilerSupport/LLVMThreadPool.h"
#include "MLRT/AsyncRT/Runtime/Runtime.h"
#include "Support/MDialect/MDialect.h"

using namespace M;

//===---------------------------------------------------------------------===//
// MContextExtension
//===---------------------------------------------------------------------===//

namespace {

/// Dialect extension to inject an RCRef<M::Context> into MDialect.
class MContextExtension : public mlir::DialectExtensionBase {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MContextExtension)

  /// Apply this extension once MDialect is loaded.
  explicit MContextExtension(ContextRef ref, bool enableThreadPool)
      : DialectExtensionBase(MDialect::getDialectNamespace()),
        ctxRef(std::move(ref)), enableThreadPool(enableThreadPool) {}

  /// Apply this extension to the given context and the required dialects.
  void apply(MLIRContext *context,
             MutableArrayRef<Dialect *> dialects) const final {
    auto *dialect = cast<MDialect>(dialects.front());

    dialect->setInternal(ctxRef.copy());
    if (context->isMultithreadingEnabled() || !enableThreadPool)
      return;
    MLRT::LLVMThreadPool *tp = ctxRef->get<MLRT::LLVMThreadPool>();
    if (!tp) {
      if (MLRT::CPUDevice *cpuDevice = ctxRef->get<MLRT::CPUDevice>())
        tp = &ctxRef->emplace<MLRT::LLVMThreadPool>(*cpuDevice);
    }

    // If the cpuDevice is available, enable threading in MLIR with it.
    if (tp)
      context->setThreadPool(*tp);
  }

  /// Return a copy of this extension.
  std::unique_ptr<DialectExtensionBase> clone() const final {
    return std::make_unique<MContextExtension>(ctxRef.copy(), enableThreadPool);
  }

private:
  ContextRef ctxRef;
  bool enableThreadPool;
};

} // namespace

void M::registerContext(mlir::DialectRegistry &registry, ContextRef &ref,
                        bool enableThreadPool) {
  std::unique_ptr<mlir::DialectExtensionBase> ctxExtension =
      std::make_unique<MContextExtension>(ref.copy(), enableThreadPool);
  registry.addExtension(mlir::TypeID::get<MContextExtension>(),
                        std::move(ctxExtension));
}

void M::registerContext(mlir::MLIRContext &ctx, ContextRef &ref,
                        bool enableThreadPool) {
  DialectRegistry registry;
  registerContext(registry, ref, enableThreadPool);
  ctx.appendDialectRegistry(registry);
}

ContextRef M::loadContext(mlir::MLIRContext *ctx) {
  StringRef name = MDialect::getDialectNamespace();
  return static_cast<MDialect *>(ctx->getOrLoadDialect(name))->getInternal();
}
