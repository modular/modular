//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLVMServer.h"
#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/Runtime/Runtime.h"
#include "Init/Init.h"
#include "KGEN/Compiler/ObjectCompiler.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "Support/Context.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/lsp-server-support/Logging.h"
#include "llvm/Support/Base64.h"
#include <optional>

using namespace M;
using namespace mlir;
namespace CSP = M::KGEN::CSP;
using namespace CSP;

//===----------------------------------------------------------------------===//
// LLVMServer::Impl
//===----------------------------------------------------------------------===//

struct LLVMServer::Impl {
  Impl(ContextRef ctx) : mlirCtx(), globalCtx(ctx.copy()) {
    initMLIRContext(mlirCtx);
  }

private:
  void initMLIRContext(MLIRContext &ctx) {
    DialectRegistry registry;
    registerAllKGENDialects(registry);
    registerKGENToLLVMTranslation(registry);
    registerContext(registry, globalCtx, /*enableThreadPool=*/true);
    mlirCtx.appendDialectRegistry(registry);
    mlirCtx.allowUnregisteredDialects(true);
  }

public:
  /// MLIR Context
  MLIRContext mlirCtx;
  /// Global context
  ContextRef globalCtx;
};

//===----------------------------------------------------------------------===//
// LLVMServer
//===----------------------------------------------------------------------===//

LLVMServer::LLVMServer(std::unique_ptr<Impl> &&impl) : impl(std::move(impl)) {}
LLVMServer::LLVMServer(LLVMServer &&) = default;

LLVMServer::~LLVMServer() = default;

ErrorOr<LLVMServer> LLVMServer::create(bool singleThreaded) {
  ErrorOr<ContextRef> ctxOr = Init::createContext(
      "compilation-server",
      Init::Options().withRuntimeOptions(AsyncRT::RuntimeOptions()
                                             .withSingleThreaded(singleThreaded)
                                             .withMainWillNotDonate()));
  if (ctxOr.isError())
    return ctxOr.takeError();
  auto impl = std::make_unique<Impl>(ctxOr->copy());
  LLVMServer server(std::move(impl));
  return server;
}

std::string LLVMServer::echoMLIR(mlir::StringRef module) {
  // Parse MLIR module into ModuleOp
  OwningOpRef<ModuleOp> moduleOp =
      parseSourceString<ModuleOp>(module, mlir::ParserConfig(&impl->mlirCtx));

  if (!moduleOp)
    return "Error: cannot parse MLIR module";

  // Print MLIR module
  std::string str;
  llvm::raw_string_ostream strStream(str);
  strStream << *moduleOp;
  return str;
}

std::string LLVMServer::emitArchive(const EmitArchiveParams &params) {
  // Parse MLIR module into ModuleOp.
  OwningOpRef<ModuleOp> moduleOp = parseSourceString<ModuleOp>(
      params.module, mlir::ParserConfig(&impl->mlirCtx));
  if (!moduleOp)
    return "Error: cannot parse MLIR module";

  // Create object compiler.
  auto compilerOr = ObjectCompiler::create(
      ".mojo_cache", params.compilationOptions, params.isJIT, impl->mlirCtx);
  if (failed(compilerOr))
    return "Error: cannot create object compiler";
  ObjectCompiler &objCompiler = **compilerOr;

  // Emit archive.
  ErrorOr<BufferRef> archiveOr = objCompiler.emitArchive(*moduleOp);
  if (failed(archiveOr))
    return "Error: cannot execute emitArchive";

  // Return emitted archive encoded as text string.
  BufferRef archive = archiveOr.takeValue();
  StringRef buffer = archive->getBuffer();

  return llvm::encodeBase64(buffer);
}
