//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLVMServer.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/lsp-server-support/Logging.h"
#include <optional>

using namespace M;
using namespace mlir;
namespace CSP = M::KGEN::CSP;

//===----------------------------------------------------------------------===//
// LLVMServer::Impl
//===----------------------------------------------------------------------===//

struct CSP::LLVMServer::Impl {
  Impl() : mlirCtx() { initMLIRContext(mlirCtx); }

private:
  void initMLIRContext(MLIRContext &ctx) {
    DialectRegistry registry;
    registerAllKGENDialects(registry);
    mlirCtx.appendDialectRegistry(registry);
    mlirCtx.allowUnregisteredDialects(true);
  }

public:
  /// MLIR Context
  MLIRContext mlirCtx;
};

//===----------------------------------------------------------------------===//
// LLVMServer
//===----------------------------------------------------------------------===//

CSP::LLVMServer::LLVMServer() : impl(std::make_unique<Impl>()) {}

CSP::LLVMServer::~LLVMServer() = default;

std::string CSP::LLVMServer::emitArchive(StringRef mlirModule) {
  // Parse MLIR module into ModuleOp
  MLIRContext mlirCtx;

  OwningOpRef<ModuleOp> moduleOp = parseSourceString<ModuleOp>(
      mlirModule, mlir::ParserConfig(&impl->mlirCtx));

  // TEMPORARY: For testing purposes return either "error" or textual
  // representation of the module.
  if (!moduleOp)
    return "error";

  std::string str;
  llvm::raw_string_ostream strStream(str);
  strStream << *moduleOp;
  return str;

  // Create ObjectCompiler

  // Call EmitArchive
}
