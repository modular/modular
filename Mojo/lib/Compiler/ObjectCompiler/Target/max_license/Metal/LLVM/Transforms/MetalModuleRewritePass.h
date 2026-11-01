//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILER_LLVMIR_TRANSFORMS_MODULEREWRITEPASS_H
#define KGEN_COMPILER_LLVMIR_TRANSFORMS_MODULEREWRITEPASS_H

#include "llvm/IR/PassManager.h"

namespace M::KGEN {

/// Module-level rewrite pass for the Metal backend. Runs before the
/// per-function InstructionRewritePass and performs two transforms that
/// require a module-wide view:
///
///   - Inline static_string_* globals into per-function stack allocas, because
///     Metal's GPU JIT compiler can't always resolve global constant symbols.
///   - Replace __mojo_metal_os_log_64 sentinel calls with air.os_log calls.
class MetalModuleRewritePass
    : public llvm::RequiredPassInfoMixin<MetalModuleRewritePass> {
public:
  llvm::PreservedAnalyses run(llvm::Module &module,
                              llvm::ModuleAnalysisManager &mam);

  static llvm::StringRef name() { return "MetalModuleRewritePass"; }
};

} // namespace M::KGEN

#endif // KGEN_COMPILER_LLVMIR_TRANSFORMS_MODULEREWRITEPASS_H
