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

#include "Target/max_license/Metal/MetalTraits.h"

namespace M::KGEN {

const MetalTraits &MetalTraits::get() {
  static const MetalTraits instance;
  return instance;
}

static constexpr TargetTraits::AcceleratorArch kMetalAcceleratorArchs[] = {
    {"apple-m1", "Apple M1"},
    {"apple-m1-metal4", "Apple M1 (Metal 4.0 / macOS 26+)"},
    {"apple-m2", "Apple M2"},
    {"apple-m2-metal4", "Apple M2 (Metal 4.0 / macOS 26+)"},
    {"apple-m3", "Apple M3"},
    {"apple-m3-metal4", "Apple M3 (Metal 4.0 / macOS 26+)"},
    {"apple-m4", "Apple M4"},
    {"apple-m4-metal4", "Apple M4 (Metal 4.0 / macOS 26+)"},
    {"apple-m5", "Apple M5"},
    {"apple-m5-metal4", "Apple M5 (Metal 4.0 / macOS 26+)"},
    {"apple-m6", "Apple M6"},
    {"apple-m6-metal4", "Apple M6 (Metal 4.0 / macOS 26+)"},
};

llvm::ArrayRef<TargetTraits::AcceleratorArch>
MetalTraits::supportedAcceleratorArchs() const {
  return kMetalAcceleratorArchs;
}

namespace {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
RegisterTargetTraits<MetalTraits> registerMetalTraits;
#pragma GCC diagnostic pop
} // namespace

} // namespace M::KGEN
