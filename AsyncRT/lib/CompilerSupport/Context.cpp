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

void M::registerContext(mlir::DialectRegistry &registry, ContextRef &ref) {
  std::function<void(MLIRContext * ctx, MDialect * dialect)> fn =
      [ref = ref.copy()](MLIRContext *ctx, MDialect *dialect) {
        dialect->setInternal(ref.copy());
        if (ctx->isMultithreadingEnabled())
          return;
        AsyncRT::LLVMThreadPool *tp = ref->get<AsyncRT::LLVMThreadPool>();
        if (!tp) {
          if (AsyncRT::Runtime *runtime = ref->get<AsyncRT::Runtime>())
            tp = &ref->emplace<AsyncRT::LLVMThreadPool>(*runtime);
        }
        // If the runtime is available, enable threading in MLIR with it.
        if (tp)
          ctx->setThreadPool(*tp);
      };
  registry.addExtension<MDialect>(std::move(fn));
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
