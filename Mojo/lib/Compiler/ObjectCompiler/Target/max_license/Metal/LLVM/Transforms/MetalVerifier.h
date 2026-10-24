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
// MetalVerifier pass - Verify LLVM IR is compatible with Metal GPU compilation
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILER_LLVMIR_TRANSFORMS_METALVERIFIER_H
#define KGEN_COMPILER_LLVMIR_TRANSFORMS_METALVERIFIER_H

#include "llvm/IR/PassManager.h"

namespace M::KGEN {

/// Rejects Metal IR containing floating-point types wider than 32 bits
/// (f64, f80, f128); Metal only supports f32 and f16.
class MetalVerifierPass
    : public llvm::RequiredPassInfoMixin<MetalVerifierPass> {
public:
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
};

} // namespace M::KGEN

#endif // KGEN_COMPILER_LLVMIR_TRANSFORMS_METALVERIFIER_H
