//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Protocol.h"

namespace CSP = M::KGEN::CSP;

//===----------------------------------------------------------------------===//
// CompileLLVMModuleParams
//===----------------------------------------------------------------------===//

bool CSP::fromJSON(const llvm::json::Value &value,
                   CompileLLVMModuleParams &result, llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("bitcode", result.bitcode);
}

llvm::json::Value CSP::toJSON(const CompileLLVMModuleParams &value) {
  return llvm::json::Object{{"bitcode", value.bitcode}};
}
