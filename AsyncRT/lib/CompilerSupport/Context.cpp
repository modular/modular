//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Context.h"
#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/CompilerSupport/LLVMThreadPool.h"
#include "AsyncRT/Runtime/Runtime.h"
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
  explicit MContextExtension(ContextRef ref)
      : DialectExtensionBase(MDialect::getDialectNamespace()),
        ctxRef(std::move(ref)) {}

  /// Apply this extension to the given context and the required dialects.
  void apply(MLIRContext *context,
             MutableArrayRef<Dialect *> dialects) const final {
    auto *dialect = cast<MDialect>(dialects.front());

    dialect->setInternal(ctxRef.copy());
    if (context->isMultithreadingEnabled())
      return;
    AsyncRT::LLVMThreadPool *tp = ctxRef->get<AsyncRT::LLVMThreadPool>();
    if (!tp) {
      if (AsyncRT::Runtime *runtime = ctxRef->get<AsyncRT::Runtime>())
        tp = &ctxRef->emplace<AsyncRT::LLVMThreadPool>(*runtime);
    }

    // If the runtime is available, enable threading in MLIR with it.
    if (tp)
      context->setThreadPool(*tp);
  }

  /// Return a copy of this extension.
  std::unique_ptr<DialectExtensionBase> clone() const final {
    return std::make_unique<MContextExtension>(ctxRef.copy());
  }

private:
  ContextRef ctxRef;
};

} // namespace

void M::registerContext(mlir::DialectRegistry &registry, ContextRef &ref) {
  std::unique_ptr<mlir::DialectExtensionBase> ctxExtension =
      std::make_unique<MContextExtension>(ref.copy());
  registry.addExtension(mlir::TypeID::get<MContextExtension>(),
                        std::move(ctxExtension));
}

void M::registerContext(mlir::MLIRContext &ctx, ContextRef &ref) {
  DialectRegistry registry;
  registerContext(registry, ref);
  ctx.appendDialectRegistry(registry);
}

ContextRef M::loadContext(mlir::MLIRContext *ctx) {
  StringRef name = MDialect::getDialectNamespace();
  return static_cast<MDialect *>(ctx->getOrLoadDialect(name))->getInteral();
}
