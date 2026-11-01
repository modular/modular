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
//
// Operand inference: maps an MO op's operands and results to a kernel's
// arguments and results.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_OPERANDINFERENCE_H
#define GENERICML_GRAPHCOMPILER_CONVERSION_OPERANDINFERENCE_H

#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"

#include "llvm/ADT/DenseSet.h"

#include <optional>

namespace M::MOGG {

/// Maps an `Operation`'s operands and results to the kernel's operands; the
/// mirror of parameter mapping. Mapping doubles as validation: an op matches
/// its kernel iff `mapMOOpToKernelImplementation` succeeds.
using KernelDeclIndexToOperandAdaptorMapping =
    llvm::MapVector<uint64_t, KernelDeclOperandAdaptor>;

struct KernelDeclOperandMapping {
  KernelDeclIndexToOperandAdaptorMapping operandMapping;
  KernelDeclIndexToOperandAdaptorMapping resultMapping;
};

/// The signature position of the kernel argument a mapping error is about,
/// when one is to blame.
using KernelArgBlame = std::optional<uint64_t>;

/// Maps a kernel call's `operands` and `results` to the kernel's arguments.
/// Needs no `Operation`, so it also serves destination-passing-style calls
/// such as `mogg.call.opaque`. `opName` names the caller in diagnostics; the
/// first two flags mirror the `SameVariadicOperandSize` /
/// extra-output-shape-argument op properties. Set `resultLeavesByValue` when
/// the kernel returns its result rather than writing a destination, so
/// `results` has no entry for it. `allowMutToImmut` lets a mutable operand
/// satisfy an immutable kernel operand (never the converse), for an operand
/// that a fused mutable-load made mutable; ConformCallOperands and MAPToMOGG's
/// epilogue classification opt in.
ErrorOr<KernelDeclOperandMapping> mapToKernelImplementation(
    ValueRange operands, ValueRange results,
    const KernelDeclFunctionAdaptor &adaptor, StringRef opName,
    bool hasSameVariadicOperandSize, bool hasExtraOutputShapeArgument,
    bool resultLeavesByValue = false, bool allowMutToImmut = false);

/// Same, and on an error about one kernel argument, sets `blame` to its
/// position.
ErrorOr<KernelDeclOperandMapping> mapToKernelImplementation(
    ValueRange operands, ValueRange results,
    const KernelDeclFunctionAdaptor &adaptor, StringRef opName,
    bool hasSameVariadicOperandSize, bool hasExtraOutputShapeArgument,
    bool resultLeavesByValue, bool allowMutToImmut, KernelArgBlame &blame);

/// Convenience overload that reads the operands/results and op properties off
/// `op`.
ErrorOr<KernelDeclOperandMapping>
mapMOOpToKernelImplementation(Operation *op,
                              const KernelDeclFunctionAdaptor &adaptor);

/// Same, and on an error about one kernel argument, sets `blame` to its
/// position.
ErrorOr<KernelDeclOperandMapping>
mapMOOpToKernelImplementation(Operation *op,
                              const KernelDeclFunctionAdaptor &adaptor,
                              KernelArgBlame &blame);

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_OPERANDINFERENCE_H
