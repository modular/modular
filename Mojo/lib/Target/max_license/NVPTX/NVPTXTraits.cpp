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

#include "Target/max_license/NVPTX/NVPTXTraits.h"

#include "llvm/Support/NVPTXAddrSpace.h"

namespace M::KGEN {

const NVPTXTraits &NVPTXTraits::get() {
  static const NVPTXTraits instance;
  return instance;
}

std::optional<unsigned>
NVPTXTraits::requiredStackAllocationAddressSpace() const {
  return llvm::NVPTXAS::AddressSpace::ADDRESS_SPACE_LOCAL;
}

static constexpr TargetTraits::AcceleratorArch kNVPTXAcceleratorArchs[] = {
    {"sm_52", "Maxwell (GTX 970)"},
    {"sm_60", "Pascal (Tesla P100)"},
    {"sm_61", "Pascal (GTX 1060, GTX 1080 Ti)"},
    {"sm_75", "Turing (RTX 2060)"},
    {"sm_80", "Ampere datacenter (A100)"},
    {"sm_86", "Ampere workstation (A10)"},
    {"sm_87", "Ampere embedded (Jetson Orin)"},
    {"sm_89", "Ada Lovelace (L4, RTX 4090)"},
    {"sm_90", "Hopper (H100)"},
    {"sm_90a", "Hopper with extensions"},
    {"sm_100", "Blackwell datacenter (B100, B200)"},
    {"sm_100a", "Blackwell datacenter with extensions"},
    {"sm_103", "Blackwell datacenter (B300)"},
    {"sm_103a", "Blackwell datacenter with extensions"},
    {"sm_110", "Blackwell embedded (Jetson Thor)"},
    {"sm_110a", "Blackwell embedded with extensions"},
    {"sm_120", "Blackwell consumer (RTX 5090)"},
    {"sm_120a", "Blackwell consumer with extensions"},
    {"sm_121", "Blackwell (DGX Spark)"},
    {"sm_121a", "Blackwell with extensions"},
    {"cuda", "Generic CUDA"},
    {"nvidia:<arch>", "(alias) e.g., nvidia:sm_90a, nvidia:80"},
};

llvm::ArrayRef<TargetTraits::AcceleratorArch>
NVPTXTraits::supportedAcceleratorArchs() const {
  return kNVPTXAcceleratorArchs;
}

namespace {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
RegisterTargetTraits<NVPTXTraits> registerNVPTXTraits;
#pragma GCC diagnostic pop
} // namespace

} // namespace M::KGEN
