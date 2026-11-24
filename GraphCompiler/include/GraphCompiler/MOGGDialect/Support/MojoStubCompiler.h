//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGSUPPORT_MOJOSTUBCOMPILER_H
#define GENERICML_GRAPHCOMPILER_MOGGSUPPORT_MOJOSTUBCOMPILER_H

#include "mlir/IR/BuiltinAttributes.h"

#include "Support/Buffer.h"
#include "Support/MDialect/MAttrs.h"

#include <string>

namespace M::MOGG {

/// `name` becomes part of the attribute name.
DenseResourceElementsAttr createArchiveAttr(MLIRContext *ctx, BufferRef buffer,
                                            StringRef name);

/// Returns the host target info, or the `set_virtual_cpu_target` override
/// when set.
///
/// A GPU accelerator override (per the triple) replaces host GPU probing, so a
/// target GPU (e.g. `sm_100a` / `nvptx64-nvidia-cuda`) can be compiled for on a
/// machine without it. A non-GPU override falls back to host detection.
///
/// `forHostExecution` ignores the virtual-CPU override: an artifact that runs
/// in the compiling process (constant folding) must target this machine.
ErrorOr<TargetInfoAttr> getTargetInfo(MLIRContext *ctx,
                                      StringRef acceleratorArchOverride = "",
                                      StringRef acceleratorTripleOverride = "",
                                      bool forHostExecution = false);

/// A resolved, validated virtual-CPU-target request, letting a host-warmed
/// cache target a different CPU or architecture.
struct VirtualCpuTargetSpec {
  /// Empty => use the host triple.
  std::string triple;
  /// Resolved CPU model name (de-sugared from "generic").
  std::string cpu;
  /// Verbatim feature delta; only meaningful when `hasExplicitFeatures`.
  std::string features;
  /// False: derive features from the `cpu` name only, without host CPUID.
  bool hasExplicitFeatures = false;
};

/// Accepts a bare arch name (host triple kept, features name-only) or a
/// descriptor "triple=<t>;cpu=<c>;features=<f>" (any order; `cpu` required).
/// `hostTriple` de-sugars "generic" and defaults a missing descriptor triple.
/// Rejects "native". Precondition: `knob` is non-empty.
ErrorOr<VirtualCpuTargetSpec> parseVirtualCpuTarget(StringRef knob,
                                                    StringRef hostTriple);

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGSUPPORT_MOJOSTUBCOMPILER_H
