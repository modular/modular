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

#ifndef KGEN_TARGET_AMDGPU_AMDGPUTRAITS_H
#define KGEN_TARGET_AMDGPU_AMDGPUTRAITS_H

#include "Target/TargetTraits.h"

#include "llvm/TargetParser/Triple.h"

namespace M::KGEN {

struct AMDGPUTraits final : TargetTraits {
  llvm::StringRef name() const override { return "amdgpu"; }
  bool isGPU() const override { return true; }
  bool matches(const llvm::Triple &triple) const override {
    return triple.isAMDGCN();
  }
  llvm::StringRef getAsmExtension() const override { return ".amdgcn"; }
  llvm::StringRef getLLVMExtension() const override { return ".amdgcn.ll"; }
  llvm::StringRef getObjectExtension() const override { return ".o"; }

  llvm::StringRef getBitcodeExtension() const override { return ".bc"; }

  llvm::ArrayRef<EmissionKind> supportedEmissionKinds() const override {
    return commonEmissionKinds();
  }

  llvm::StringRef acceleratorSectionTitle() const override {
    return "AMD (ROCm/HIP):";
  }
  llvm::ArrayRef<AcceleratorArch> supportedAcceleratorArchs() const override;
  std::optional<unsigned> requiredStackAllocationAddressSpace() const override;

  // This is max only target
  bool isBaseTarget() const override { return false; }

  /// Shared stateless instance for the backend/lowering `traits()`.
  static const AMDGPUTraits &get();
};

} // namespace M::KGEN

#endif // KGEN_TARGET_AMDGPU_AMDGPUTRAITS_H
