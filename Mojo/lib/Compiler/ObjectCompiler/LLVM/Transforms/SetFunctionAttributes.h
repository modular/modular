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

#ifndef KGEN_COMPILER_LLVMIR_TRANSFORMS_SETFUNCTIONATTRIBUTES_H
#define KGEN_COMPILER_LLVMIR_TRANSFORMS_SETFUNCTIONATTRIBUTES_H

#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/IR/PassManager.h"
#include <optional>

namespace llvm {
class Module;
} // namespace llvm

namespace M::KGEN {

/// Pass to set some function attributes that are needed for compilation.
class SetFunctionAttributes
    : public llvm::RequiredPassInfoMixin<SetFunctionAttributes> {
public:
  /// `denormalFP32Mode` sets the f32 denormal mode of every function; unset
  /// leaves the functions' denormal attributes as they are.
  explicit SetFunctionAttributes(
      std::optional<llvm::DenormalMode::DenormalModeKind> denormalFP32Mode =
          std::nullopt)
      : denormalFP32Mode(denormalFP32Mode) {}

  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);

  static llvm::StringRef name() { return "KGEN::SetFunctionAttributes"; }

private:
  std::optional<llvm::DenormalMode::DenormalModeKind> denormalFP32Mode;
};

} // namespace M::KGEN

#endif // KGEN_COMPILER_LLVMIR_TRANSFORMS_SETFUNCTIONATTRIBUTES_H
