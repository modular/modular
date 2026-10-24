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
//
// Metal AIR Pass - Transform LLVM IR for Metal GPU compilation
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILER_LLVMIR_TRANSFORMS_METALAIRPASS_H
#define KGEN_COMPILER_LLVMIR_TRANSFORMS_METALAIRPASS_H

#include "llvm/IR/PassManager.h"

namespace llvm {
class Module;
class ModulePass;
} // namespace llvm

namespace M::KGEN {

/// New pass manager pass that transforms LLVM IR to be compatible with Apple's
/// AIR (Apple Intermediate Representation) for Metal GPU compilation.
///
/// This pass:
/// - Transforms target triple to air64-apple-macosx
/// - Adds required AIR metadata for metallib compilation
/// - Converts GPU intrinsics to AIR intrinsics
/// - Adds air.kernel attributes to kernel functions
///
/// The pass only activates when targeting Metal GPU (air64-apple-macosx).
///
/// The Metal language version, AIR version, and macOS deployment target are
/// derived from the "target-features" function attribute present in the IR.
class MetalAIRPass : public llvm::RequiredPassInfoMixin<MetalAIRPass> {
public:
  MetalAIRPass() = default;

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
};

} // namespace M::KGEN

#endif // KGEN_COMPILER_LLVMIR_TRANSFORMS_METALAIRPASS_H
