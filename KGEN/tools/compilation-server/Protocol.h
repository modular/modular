//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements structs for Compile Server Protocol.
//
// Each struct has a toJSON and fromJSON function, that converts between
// the struct and a JSON representation. (See JSON.h)
//
// Some structs also have operator<< serialization. This is for debugging and
// tests, and is not generally machine-readable.

//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_CS_PROTOCOL_H
#define KGEN_TOOLS_CS_PROTOCOL_H

#include "Support/LLVMForwardDecls.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <string>

namespace M::KGEN::CSP {

//===----------------------------------------------------------------------===//
// CompileLLVMModuleParams
//===----------------------------------------------------------------------===//

struct CompileLLVMModuleParams {
  /// LLVM bitcode encoded as string.
  std::string bitcode;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, CompileLLVMModuleParams &result,
              llvm::json::Path path);
llvm::json::Value toJSON(const CompileLLVMModuleParams &value);

} // namespace M::KGEN::CSP

#endif // KGEN_TOOLS_CS_PROTOCOL_H
