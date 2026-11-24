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

#ifndef KGEN_TARGET_NVPTX_NVPTXTRAITS_H
#define KGEN_TARGET_NVPTX_NVPTXTRAITS_H

#include "Target/TargetTraits.h"

#include "llvm/TargetParser/Triple.h"

namespace M::KGEN {

struct NVPTXTraits final : TargetTraits {
  llvm::StringRef name() const override { return "nvptx"; }
  bool isGPU() const override { return true; }
  bool matches(const llvm::Triple &triple) const override {
    return triple.isNVPTX();
  }
  llvm::StringRef getAsmExtension() const override { return ".ptx"; }
  llvm::StringRef getLLVMExtension() const override { return ".nvptx.ll"; }
  llvm::StringRef getObjectExtension() const override { return ".cubin"; }

  llvm::StringRef getBitcodeExtension() const override { return ".bc"; }

  llvm::ArrayRef<EmissionKind> supportedEmissionKinds() const override {
    return commonEmissionKinds();
  }

  llvm::StringRef acceleratorSectionTitle() const override {
    return "NVIDIA (CUDA):";
  }
  llvm::ArrayRef<AcceleratorArch> supportedAcceleratorArchs() const override;
  std::optional<unsigned> requiredStackAllocationAddressSpace() const override;

  // This is max only target
  bool isBaseTarget() const override { return false; }

  /// Shared stateless instance for the backend/lowering `traits()`.
  static const NVPTXTraits &get();
};

} // namespace M::KGEN

#endif // KGEN_TARGET_NVPTX_NVPTXTRAITS_H
