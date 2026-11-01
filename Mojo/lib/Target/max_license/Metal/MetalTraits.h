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

#ifndef KGEN_TARGET_METAL_METALTRAITS_H
#define KGEN_TARGET_METAL_METALTRAITS_H

#include "Target/TargetTraits.h"

#include "llvm/TargetParser/Triple.h"

namespace M::KGEN {

struct MetalTraits final : TargetTraits {
  llvm::StringRef name() const override { return "metal"; }
  bool isGPU() const override { return true; }
  // `xcrun metallib` is macOS-only, so a standalone object can't be produced
  // when cross-compiling; the emitted asm/IR file is aliased as the object.
  bool emitsOffloadObjectFile() const override { return false; }
  // Metal compiles through an arm64 triple; LLVM has no air64 target.
  std::string codegenTriple(llvm::StringRef triple) const override {
    std::string s = triple.str();
    auto pos = s.find("air64");
    if (pos != std::string::npos)
      s.replace(pos, 5, "arm64");
    return s;
  }
  // Metal AIR is emitted as LLVM-17-format bitcode.
  unsigned forcedBitcodeVersion() const override { return 17; }
  bool matches(const llvm::Triple &triple) const override {
    // Metal GPU targets compile through an ARM64/AIR triple spelled "air64-".
    return triple.str().starts_with("air64-");
  }
  // Metal uses LLVM IR text as its human-readable "assembly".
  llvm::StringRef getAsmExtension() const override { return ".ll"; }
  llvm::StringRef getLLVMExtension() const override { return ".metal.ll"; }
  llvm::StringRef getObjectExtension() const override { return ".metallib"; }
  // Metal AIR is LLVM bitcode in the forced version (see above).
  llvm::StringRef getBitcodeExtension() const override { return ".air"; }

  llvm::ArrayRef<EmissionKind> supportedEmissionKinds() const override {
    static const llvm::SmallVector<EmissionKind, 8> kinds = [] {
      llvm::SmallVector<EmissionKind, 8> result(commonEmissionKinds());
      // Optimized bitcode in the LLVM version Metal forces is AIR.
      result.push_back({"air", EmitAs::LLVM_OPT_BITCODE,
                        "one optimized Apple AIR bitcode .air file "
                        "per kernel, plus host LLVM bitcode"});
      return result;
    }();
    return kinds;
  }

  llvm::StringRef acceleratorSectionTitle() const override {
    return "Apple Silicon GPU:";
  }
  llvm::ArrayRef<AcceleratorArch> supportedAcceleratorArchs() const override;

  // This is max only target
  bool isBaseTarget() const override { return false; }

  /// Shared stateless instance for the backend/lowering `traits()`.
  static const MetalTraits &get();
};

} // namespace M::KGEN

#endif // KGEN_TARGET_METAL_METALTRAITS_H
