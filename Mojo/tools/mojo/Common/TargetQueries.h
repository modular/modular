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
// This file implements the `--print-*` target queries shared by the `mojo`
// subcommands that accept target options.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_COMMON_TARGETQUERIES_H
#define KGEN_TOOLS_MOJO_COMMON_TARGETQUERIES_H

#include "Support/LLVMForwardDecls.h"

namespace M {
class TargetInfoAttr;

/// Print the targets this build can generate code for.
int printSupportedTargets();

/// Print valid CPU names for a target triple.
int printSupportedCpus(StringRef userTriple);

/// Print the supported accelerator architectures declared by the
/// registered `TargetTraits`.
int printSupportedAccelerators();

/// Print effective target configuration as command-line options.
int printEffectiveTarget(TargetInfoAttr targetInfo);

} // namespace M

#endif // KGEN_TOOLS_MOJO_COMMON_TARGETQUERIES_H
