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

#include "Target/max_license/AMDGPU/AMDGPUTraits.h"

#include "llvm/Support/AMDGPUAddrSpace.h"

namespace M::KGEN {

const AMDGPUTraits &AMDGPUTraits::get() {
  static const AMDGPUTraits instance;
  return instance;
}

std::optional<unsigned>
AMDGPUTraits::requiredStackAllocationAddressSpace() const {
  return llvm::AMDGPUAS::PRIVATE_ADDRESS;
}

static constexpr TargetTraits::AcceleratorArch kAMDGPUAcceleratorArchs[] = {
    {"gfx90a", "CDNA2 (MI250X)"},
    {"mi250x", "(alias) -> gfx90a"},
    {"gfx942", "CDNA3 (MI300X)"},
    {"mi300a", "CDNA3 APU (MI300A)"},
    {"mi300x", "(alias) -> gfx942"},
    {"gfx950", "CDNA4 (MI355X)"},
    {"mi355x", "(alias) -> gfx950"},
    {"gfx1030", "RDNA2 (Radeon 6900)"},
    {"gfx1033", "RDNA2 (Van Gogh)"},
    {"gfx1100", "RDNA3 (Radeon 7900)"},
    {"gfx1101", "RDNA3 (Radeon 7800/7700)"},
    {"gfx1102", "RDNA3 (Radeon 7600)"},
    {"gfx1103", "RDNA3 (Radeon 780M)"},
    {"gfx1150", "RDNA3+ (Radeon 880M)"},
    {"gfx1151", "RDNA3+ (Radeon 8060S)"},
    {"gfx1152", "RDNA3+ (Radeon 860M)"},
    {"gfx1200", "RDNA4 (Radeon 9060)"},
    {"gfx1201", "RDNA4 (Radeon 9070)"},
    {"gfx1250", "CDNA5 (MI455X)"},
    {"mi455x", "(alias) -> gfx1250"},
    {"amdgpu:<arch>", "(alias) e.g., amdgpu:gfx942"},
};

llvm::ArrayRef<TargetTraits::AcceleratorArch>
AMDGPUTraits::supportedAcceleratorArchs() const {
  return kAMDGPUAcceleratorArchs;
}

namespace {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
RegisterTargetTraits<AMDGPUTraits> registerAMDGPUTraits;
#pragma GCC diagnostic pop
} // namespace

} // namespace M::KGEN
